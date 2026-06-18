"""Reclassify all chunks in the DB with the new two-level taxonomy.

Strategy (two passes):
  Pass 1 — fast legacy map (SQL UPDATE, no API calls)
    Uses LEGACY_CATEGORY_MAP to bulk-fill domain/subdomain from the old
    category_legacy column. Covers ~95% of chunks correctly in seconds.

  Pass 2 — LLM reclassification (Haiku, batched, parallel)
    Targets chunks where the legacy map was ambiguous or wrong:
      - category_legacy IN ('porezi', 'računovodstvo', 'ostalo')
        because these were catch-all buckets that now need splitting.
      - Any chunk where domain IS NULL after pass 1.
    Sends chunk_text to Haiku in batches of 20. Haiku returns
    {domain, subdomain} for each. Results written back in batches.

Usage:
    python scripts/reclassify_all.py                  # full run
    python scripts/reclassify_all.py --pass1-only     # legacy map only, no API
    python scripts/reclassify_all.py --pass2-only     # LLM pass only
    python scripts/reclassify_all.py --dry-run        # print counts, no writes
    python scripts/reclassify_all.py --workers 10     # more parallelism
    python scripts/reclassify_all.py --limit 100      # test on 100 chunks
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import psycopg
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.taxonomy import (
    LEGACY_CATEGORY_MAP,
    SUBDOMAIN_TO_DOMAIN,
    VALID_DOMAINS,
    VALID_SUBDOMAINS,
    prompt_taxonomy_block,
)

load_dotenv()
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5"
BATCH_SIZE  = 20   # chunks per API call
DEFAULT_WORKERS = 5


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""Ti si klasifikator tekstualnih isječaka iz hrvatskog računovodstvenog i poreznog znanja.

Svaki isječak je dio članka, zakona ili stručnog teksta iz oblasti računovodstva, poreza i financija.

Tvoj zadatak: za svaki isječak odredi domain i subdomain prema ovoj taksonomiji:

{prompt_taxonomy_block()}

Pravila:
1. Odaberi NAJTOČNIJI subdomain. Domain se automatski određuje iz subdomaina.
2. Ako isječak pokriva više tema, odaberi PRIMARNU temu (onu koja zauzima više prostora).
3. Za kratke isječke bez jasnog sadržaja (npr. zaglavlja, kazala, oglasi) koristi domain="ostalo", subdomain="ostalo".
4. Nikad ne izmišljaj nove domaine ili subdomaine — koristi ISKLJUČIVO vrijednosti iz taksonomije.

Vrati ISKLJUČIVO JSON array bez ikakvog teksta prije ili nakon, u ovom obliku:
[
  {{"id": "<chunk_id>", "domain": "<domain>", "subdomain": "<subdomain>"}},
  ...
]
"""


# ── Pass 1: legacy map ────────────────────────────────────────────────────────

def run_pass1(conn: psycopg.Connection, dry_run: bool) -> dict[str, int]:
    """Bulk-fill domain/subdomain from legacy category map. Returns counts."""
    counts: dict[str, int] = {}
    with conn.cursor() as cur:
        for legacy_cat, (domain, subdomain) in LEGACY_CATEGORY_MAP.items():
            if dry_run:
                cur.execute(
                    "SELECT count(*) FROM chunks WHERE category_legacy = %s",
                    (legacy_cat,),
                )
                n = cur.fetchone()[0]
            else:
                cur.execute(
                    """
                    UPDATE chunks
                    SET domain = %s, subdomain = %s
                    WHERE category_legacy = %s
                    """,
                    (domain, subdomain, legacy_cat),
                )
                n = cur.rowcount
            counts[legacy_cat] = n
        if not dry_run:
            conn.commit()
    return counts


# ── Pass 2: LLM reclassification ──────────────────────────────────────────────

def _chunks_needing_llm(conn: psycopg.Connection, limit: int | None) -> list[tuple]:
    """Return (id, chunk_text) for chunks that need LLM reclassification.

    Targets the ambiguous legacy categories where splitting is needed:
      - porezi     → may be dohodak / dobit / paušal / PDV / porezi
      - računovodstvo → may be knjiženje / fin_izv / računovodstvo
      - ostalo     → may be any domain

    Also catches any chunk still NULL after pass 1.
    """
    ambiguous = ("porezi", "računovodstvo", "ostalo")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id::text, chunk_text
            FROM chunks
            WHERE category_legacy = ANY(%s)
               OR domain IS NULL
            ORDER BY id
            """ + (f" LIMIT {limit}" if limit else ""),
            (list(ambiguous),),
        )
        return cur.fetchall()


def _classify_batch(
    client: anthropic.Anthropic,
    batch: list[tuple],  # list of (id, chunk_text)
) -> list[dict]:
    """Send one batch to Haiku. Returns list of {id, domain, subdomain}."""
    payload = [
        {"id": cid, "text": text[:600]}  # truncate to 600 chars — enough for classification
        for cid, text in batch
    ]
    user_msg = json.dumps(payload, ensure_ascii=False)

    try:
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        results = json.loads(raw)
        # Validate each result
        validated = []
        for r in results:
            subdomain = r.get("subdomain", "ostalo")
            domain    = r.get("domain", "ostalo")
            # Correct domain if it doesn't match subdomain
            if subdomain in SUBDOMAIN_TO_DOMAIN:
                domain = SUBDOMAIN_TO_DOMAIN[subdomain]
            # Fallback for unknown values
            if subdomain not in VALID_SUBDOMAINS:
                log.warning("Unknown subdomain %r, falling back to ostalo", subdomain)
                subdomain, domain = "ostalo", "ostalo"
            validated.append({"id": r["id"], "domain": domain, "subdomain": subdomain})
        return validated
    except Exception as e:
        log.warning("Batch failed: %s", e)
        # Return safe fallback for the whole batch
        return [{"id": cid, "domain": "ostalo", "subdomain": "ostalo"} for cid, _ in batch]


def _write_batch(conn: psycopg.Connection, results: list[dict]) -> int:
    """Write a batch of classification results to the DB. Returns rows updated."""
    with conn.cursor() as cur:
        for r in results:
            cur.execute(
                "UPDATE chunks SET domain = %s, subdomain = %s WHERE id = %s::uuid",
                (r["domain"], r["subdomain"], r["id"]),
            )
        conn.commit()
    return len(results)


def run_pass2(
    chunks: list[tuple],
    dry_run: bool,
    workers: int,
) -> dict[str, int]:
    """LLM reclassification pass. Returns subdomain distribution."""
    if not chunks:
        print("  No chunks need LLM reclassification.")
        return {}

    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        timeout=30.0,
    )

    batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
    print(f"  {len(chunks)} chunks → {len(batches)} batches × {BATCH_SIZE}")

    all_results: list[dict] = []
    subdomain_counts: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_classify_batch, client, batch): batch
            for batch in batches
        }
        with tqdm(total=len(batches), desc="  LLM classify") as pbar:
            for future in as_completed(futures):
                results = future.result()
                all_results.extend(results)
                for r in results:
                    subdomain_counts[r["subdomain"]] = (
                        subdomain_counts.get(r["subdomain"], 0) + 1
                    )
                pbar.update(1)

    if not dry_run:
        db_url = os.environ["DATABASE_URL"]
        written = 0
        with psycopg.connect(db_url) as conn:
            write_batches = [
                all_results[i:i + 200]
                for i in range(0, len(all_results), 200)
            ]
            for wb in tqdm(write_batches, desc="  Writing to DB"):
                written += _write_batch(conn, wb)
        print(f"  Written {written} rows.")
    else:
        print(f"  DRY RUN — would update {len(all_results)} rows.")

    return subdomain_counts


# ── Verification ──────────────────────────────────────────────────────────────

def print_distribution(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT domain, subdomain, count(*)
            FROM chunks
            GROUP BY domain, subdomain
            ORDER BY domain, count(*) DESC
            """
        )
        rows = cur.fetchall()

    print("\n=== DISTRIBUTION AFTER RECLASSIFICATION ===")
    current_domain = None
    domain_total = 0
    for domain, subdomain, cnt in rows:
        if domain != current_domain:
            if current_domain is not None:
                print(f"  {'TOTAL':30s} {domain_total:6d}")
            print(f"\n  {domain}")
            current_domain = domain
            domain_total = 0
        subdomain_label = subdomain or "(null)"
        print(f"    {subdomain_label:28s} {cnt:6d}")
        domain_total += cnt
    if current_domain:
        print(f"  {'TOTAL':30s} {domain_total:6d}")

    cur2 = conn.cursor()
    cur2.execute("SELECT count(*) FROM chunks WHERE domain IS NULL")
    nulls = cur2.fetchone()[0]
    print(f"\n  Chunks with domain=NULL: {nulls}  {'✅' if nulls == 0 else '⚠️  run again'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Reclassify chunks with new taxonomy")
    parser.add_argument("--pass1-only",  action="store_true")
    parser.add_argument("--pass2-only",  action="store_true")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--workers",     type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit",       type=int, default=None,
                        help="Process only N chunks in pass 2 (for testing)")
    args = parser.parse_args()

    db_url = os.environ["DATABASE_URL"]

    print(f"\n{'='*60}")
    print(f"  RRiF Chunk Reclassification")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"{'='*60}\n")

    with psycopg.connect(db_url) as conn:

        # ── Pass 1 ────────────────────────────────────────────────────────────
        if not args.pass2_only:
            print("PASS 1 — legacy category map (no API calls)")
            t0 = time.perf_counter()
            counts = run_pass1(conn, args.dry_run)
            elapsed = time.perf_counter() - t0
            for legacy_cat, n in sorted(counts.items(), key=lambda x: -x[1]):
                print(f"  {legacy_cat:35s} → {n:5d} chunks")
            print(f"  Done in {elapsed:.1f}s\n")

        # ── Pass 2 ────────────────────────────────────────────────────────────
        if not args.pass1_only:
            print("PASS 2 — LLM reclassification (ambiguous categories)")
            t0 = time.perf_counter()
            chunks = _chunks_needing_llm(conn, args.limit)
            print(f"  Found {len(chunks)} chunks to reclassify")

            if chunks:
                subdomain_counts = run_pass2(chunks, args.dry_run, args.workers)
                elapsed = time.perf_counter() - t0
                print(f"\n  Subdomain breakdown from LLM pass:")
                for sd, cnt in sorted(subdomain_counts.items(), key=lambda x: -x[1]):
                    print(f"    {sd:28s} {cnt:5d}")
                print(f"  Done in {elapsed:.1f}s")

        # ── Verification ──────────────────────────────────────────────────────
        if not args.dry_run:
            print_distribution(conn)


if __name__ == "__main__":
    main()
