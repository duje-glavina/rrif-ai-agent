r"""Ingest the full archive from a manifest, and measure itself doing it.

WHAT IS DIFFERENT FROM ingest_rrif_articles.py
──────────────────────────────────────────────
**It ingests a list, not a directory tree.** `collect_pdfs()` walked the disk
and recursed exactly one level, which silently dropped all 857 files under
`_prilozi/<Naziv>/<ISSUE>/` and silently included `RRIF1603/Backup/` — 36
re-exported copies of an issue that is already there, of which md5 catches
exactly one. Both failures are invisible: no error, just a corpus that is
quietly wrong in both directions. Here the input is `manifest.csv`, the rows
are filtered on `folder_ok` and `file_ok`, and **you can read what will be
ingested before you ingest it**.

**It batches.** The old path called `ingest()` once per article: one DB
connection, one embed call of ~30 chunks, and one Haiku round-trip per 20
chunks, 7,900 times over. On a 5090 a batch of 30 is most of a millisecond of
compute wrapped in a lot of waiting. Chunks are accumulated across articles and
flushed in blocks.

**It records what it did.** One JSONL line per article — timings, chunk count,
where the title came from, whether an author was found, why it was skipped —
and phase totals at the end. When this has to be run again, or explained to a
client asking "how long until it's live", the answer is a number.

**It can be stopped.** `--resume` reads the unit_refs already in the target
database and skips them, so a run that dies at article 6,000 costs you nothing
but the time already spent.

COST WARNING
────────────
Classification is a Haiku call per 20 chunks. At ~155,000 chunks that is ~7,750
calls and on the order of $80–100, plus most of an hour of wall clock. For a
sizing probe you do not need it: `--no-classify` leaves domain/subdomain NULL,
which `scripts/reclassify_all.py` can fill later if the probe DB is ever
promoted. Measure classification separately on a sample instead:

    python scripts/ingest_v3.py --manifest ... --sample 200 --db rrif_rag_v3_probe

USAGE
─────
    # prepare the target
    createdb rrif_rag_v3_probe
    psql -d rrif_rag_v3_probe -f db/schema_full.sql

    # see what would be ingested, extract and chunk, write nothing
    python scripts/ingest_v3.py --manifest data/manifest_rrif.csv \
                                --manifest data/manifest_pip.csv --dry-run

    # calibration: 200 articles spread across every publication and year
    python scripts/ingest_v3.py --manifest data/manifest_rrif.csv \
                                --manifest data/manifest_pip.csv \
                                --db rrif_rag_v3_probe --sample 200 --no-classify

    # the real thing
    python scripts/ingest_v3.py --manifest data/manifest_rrif.csv \
                                --manifest data/manifest_pip.csv \
                                --db rrif_rag_v3_probe --no-classify --resume
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

LOG_DIR = Path("eval/results")


# ── Manifest ──────────────────────────────────────────────────────────────────

def load_manifest(path: Path, include_mismatch: bool) -> list[dict]:
    """Rows that name a real article, with an absolute path attached.

    A row is ingestable when both `folder_ok` and `file_ok` are true — which is
    what keeps Backup/, ispravak_*/, whole-issue PDFs and stray a.pdf out
    without a single hard-coded exclusion. `issue_mismatch` rows (the file name
    disagrees with the folder about which issue it is) are held back by
    default: one misfiled article is better left out than dated wrong.
    """
    with path.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if rows and "root" not in rows[0]:
        raise SystemExit(
            f"{path} has no 'root' column — rebuild it with the current "
            f"scripts/build_manifest.py, which writes one.")

    out = []
    for r in rows:
        if r.get("folder_ok") != "True" or r.get("file_ok") != "True":
            continue
        if r.get("issue_mismatch") == "True" and not include_mismatch:
            continue
        r["path"] = Path(r["root"]) / r["rel_path"]
        out.append(r)
    return out


def stratify(rows: list[dict], n: int, seed: int = 20260914) -> list[dict]:
    """Take n rows spread across (publication, year), not the first n.

    The first n of a sorted manifest is one publication and two years, which
    measures the wrong thing twice: extraction speed varies with layout era,
    and chunk counts vary with article length, and both differ by publication.
    """
    if n >= len(rows):
        return rows
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["pub_code"], r["year"])].append(r)
    rnd = random.Random(seed)
    picked: list[dict] = []
    keys = sorted(buckets)
    # Proportional, at least one per bucket, then top up round-robin.
    for k in keys:
        share = max(1, round(n * len(buckets[k]) / len(rows)))
        picked.extend(rnd.sample(buckets[k], min(share, len(buckets[k]))))
    rnd.shuffle(picked)
    return picked[:n]


# ── Database ──────────────────────────────────────────────────────────────────

INSERT_SQL = """
INSERT INTO chunks (
    chunk_text, embedding,
    category_legacy, source_type, source,
    law_name, article_number, nn_reference,
    valid_from, valid_to, status, citable,
    chunk_index, total_chunks, extra_metadata,
    domain, subdomain
) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
"""


def resolve_dsn(db: str) -> str:
    """--db takes either a bare database name or a full DSN.

    A bare name is grafted onto DATABASE_URL, so the probe inherits the same
    host, user and password as the real thing and differs in exactly one
    component. Being explicit here is the whole point: the failure mode this
    guards against is a forgotten .env writing a measurement run into v2.
    """
    if not db:
        raise SystemExit("--db is required for a live run (e.g. --db rrif_rag_v3_probe)")
    if "://" in db:
        return db
    base = os.environ.get("DATABASE_URL", "")
    if not base:
        raise SystemExit("No DATABASE_URL in the environment to derive --db from; "
                         "pass a full DSN instead.")
    head, _, _tail = base.rpartition("/")
    return f"{head}/{db}"


def existing_unit_refs(dsn: str) -> set[str]:
    import psycopg
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT extra_metadata->>'unit_ref' FROM chunks "
                    "WHERE extra_metadata ? 'unit_ref'")
        return {r[0] for r in cur.fetchall() if r[0]}


# ── Run ───────────────────────────────────────────────────────────────────────

class Timers:
    def __init__(self):
        self.t = Counter()

    def add(self, phase: str, seconds: float):
        self.t[phase] += seconds

    def report(self) -> dict:
        """Raw seconds, biggest first, plus the sum. Rounding happens at print
        time — rounding here made a phase read as 104 % of a short run."""
        return dict(sorted(self.t.items(), key=lambda kv: -kv[1])) | {
            "ukupno (faze)": sum(self.t.values())}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="append", required=True,
                    help="manifest.csv (repeatable)")
    ap.add_argument("--db", default="",
                    help="Target database name or full DSN. Required unless --dry-run.")
    ap.add_argument("--sample", type=int, default=0,
                    help="Ingest this many articles, spread across publication and year")
    ap.add_argument("--limit", type=int, default=0, help="Hard cap, applied after sampling")
    ap.add_argument("--batch", type=int, default=512,
                    help="Chunks accumulated before an embed+insert flush (default 512)")
    ap.add_argument("--embed-batch", type=int, default=128,
                    help="Chunks per forward pass (default 128; lower it on CUDA OOM)")
    ap.add_argument("--no-classify", action="store_true",
                    help="Leave domain/subdomain NULL. Saves ~7,750 Haiku calls.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip unit_refs already present in the target database")
    ap.add_argument("--include-mismatch", action="store_true",
                    help="Also ingest articles whose file name disagrees with their folder")
    ap.add_argument("--dry-run", action="store_true",
                    help="Extract and chunk only — no embedding, no database")
    args = ap.parse_args()

    rows: list[dict] = []
    for m in args.manifest:
        rows.extend(load_manifest(Path(m), args.include_mismatch))
    if not rows:
        print("Manifest nema nijedan redak koji zadovoljava uvjete.")
        return 1
    rows.sort(key=lambda r: r["unit_ref"])

    dsn = "" if args.dry_run else resolve_dsn(args.db)
    skip: set[str] = set()
    if args.resume and not args.dry_run:
        skip = existing_unit_refs(dsn)
        print(f"  --resume: {len(skip)} članaka je već u bazi")

    rows = [r for r in rows if r["unit_ref"] not in skip]
    if args.sample:
        rows = stratify(rows, args.sample)
        rows.sort(key=lambda r: r["unit_ref"])
    if args.limit:
        rows = rows[:args.limit]

    pubs = Counter(r["pub_code"] for r in rows)
    print(f"\n{'='*70}")
    print("  INGEST v3" + ("  — DRY RUN" if args.dry_run else f"  →  {args.db}"))
    print(f"{'='*70}")
    print(f"  Članaka:      {len(rows)}")
    print(f"  Po izdanju:   " + "  ".join(f"{k} {v}" for k, v in sorted(pubs.items())))
    print(f"  Klasifikacija: {'NE' if args.no_classify or args.dry_run else 'DA (Haiku)'}")
    print(f"{'='*70}\n")

    # Imports are deferred so that --dry-run neither loads a 2 GB model nor
    # needs a database to be reachable.
    from rag.ingest.article_loader import load_article
    if not args.dry_run:
        import numpy as np
        import psycopg
        from pgvector.psycopg import register_vector
        from rag.embedder import embed_passages
        from rag.classifier import STATUS_VALID
        if not args.no_classify:
            from rag.ingest.pipeline import _classify_chunks
    else:
        STATUS_VALID = "vazeci"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{stamp}_ingest_v3.jsonl"
    log = log_path.open("w", encoding="utf-8")

    timers = Timers()
    pending: list[tuple] = []          # (text, article_number, chunk_index, total, meta_tuple)
    n_articles = n_skipped = n_chunks = n_inserted = 0
    skip_reasons = Counter()
    title_src = Counter()
    no_author = 0
    errors: list[tuple[str, str]] = []
    t0 = time.perf_counter()

    conn = None
    if not args.dry_run:
        conn = psycopg.connect(dsn)
        register_vector(conn)

    def flush():
        nonlocal n_inserted, pending
        if not pending or args.dry_run:
            pending = []
            return
        texts = [p[0] for p in pending]

        t = time.perf_counter()
        vecs = embed_passages(texts, batch_size=args.embed_batch, show_progress=False)
        timers.add("embedding", time.perf_counter() - t)

        if args.no_classify:
            cls = [(None, None)] * len(texts)
        else:
            t = time.perf_counter()
            cls = _classify_chunks(texts)
            timers.add("klasifikacija", time.perf_counter() - t)

        t = time.perf_counter()
        params = []
        for (text, art_no, idx, total, meta), vec, (dom, sub) in zip(pending, vecs, cls):
            params.append((
                text, vec,
                meta["category"], "članak", meta["source"],
                None, art_no, None,
                meta["valid_from"], None, STATUS_VALID, True,
                idx, total, psycopg.types.json.Jsonb(meta["extra"]),
                dom, sub,
            ))
        with conn.cursor() as cur:
            cur.executemany(INSERT_SQL, params)
        conn.commit()
        timers.add("insert", time.perf_counter() - t)

        n_inserted += len(params)
        pending = []

    for i, r in enumerate(rows, 1):
        pdf = r["path"]
        rec: dict = {"unit_ref": r["unit_ref"], "pub": r["pub_code"],
                     "year": int(r["year"]), "month": int(r["month"]),
                     "rel_path": r["rel_path"]}

        t = time.perf_counter()
        try:
            chunks = load_article(pdf, verbose=False)
        except Exception as e:                                  # noqa: BLE001
            timers.add("ekstrakcija", time.perf_counter() - t)
            errors.append((r["unit_ref"], f"{type(e).__name__}: {e}"))
            rec |= {"ok": False, "error": str(e)[:200]}
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue
        dt = time.perf_counter() - t
        timers.add("ekstrakcija", dt)
        rec["ms_extract"] = round(dt * 1000, 1)

        if not chunks:
            n_skipped += 1
            # load_article prints its own reason when verbose; here we only know
            # that nothing came back, which in practice means ads or too short.
            skip_reasons["prazno / prekratko / oglas"] += 1
            rec |= {"ok": False, "skipped": True}
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue

        first = chunks[0]
        n_articles += 1
        n_chunks += len(chunks)
        if not first.author:
            no_author += 1
        title_src[bool(first.title)] += 1

        rec |= {
            "ok": True, "chunks": len(chunks),
            "chars": sum(c.char_count for c in chunks),
            "tokens": sum(c.token_count for c in chunks),
            "over_508": sum(1 for c in chunks if c.token_count > 508),
            "title": first.title[:120], "author": first.author or "",
            "sections": len({c.section_num for c in chunks if c.section_num}),
            "citation": first.source_citation[:160],
        }
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")

        meta = {
            "category": first.default_category,
            "source": first.source_citation,
            "valid_from": date(first.year, first.month, 1),
            "extra": {
                "pub_type": first.pub_type,
                "pub_label": first.pub_label,
                "article_num": first.article_num,
                "author": first.author or "",
                "title": first.title,
                # The join key for RRiF's metadata spreadsheet. Everything else
                # in this dict is derived from the PDF; this one is derived from
                # the file's place in the archive, which is what their XLS also
                # keys on.
                "unit_ref": r["unit_ref"],
            },
        }
        for c in chunks:
            pending.append((c.text, c.section_num, c.chunk_index, len(chunks), meta))

        if len(pending) >= args.batch:
            flush()

        if i % 50 == 0 or i == len(rows):
            el = time.perf_counter() - t0
            rate = i / el if el else 0
            eta = (len(rows) - i) / rate if rate else 0
            print(f"  {i}/{len(rows)}  {n_chunks} chunkova  "
                  f"{rate:.1f} čl/s  preostalo ~{eta/60:.0f} min", end="\r", flush=True)

    flush()
    if conn:
        conn.close()
    log.close()
    elapsed = time.perf_counter() - t0

    print(" " * 78, end="\r")
    print(f"\n{'='*70}")
    print("  REZULTAT")
    print(f"{'='*70}")
    print(f"  Članaka učitano:   {n_articles}")
    print(f"  Preskočeno:        {n_skipped}")
    if n_articles:
        print(f"  Chunkova:          {n_chunks}   ({n_chunks / n_articles:.1f} po članku)")
    if not args.dry_run:
        print(f"  Redaka upisano:    {n_inserted}")
    if n_articles:
        print(f"  Bez autora:        {no_author}  "
              f"({no_author / n_articles * 100:.1f} %)")
    print(f"  Ukupno vrijeme:    {elapsed/60:.1f} min")
    if n_articles:
        print(f"  Po članku:         {elapsed / n_articles * 1000:.0f} ms")

    print("\n  Po fazi (s):")
    for phase, secs in timers.report().items():
        share = f"{min(secs / elapsed, 1.0) * 100:4.1f} %" if elapsed else ""
        print(f"    {phase:<16} {secs:>9.1f}   {share}")
    print(f"    {'ostalo':<16} {max(elapsed - sum(timers.t.values()), 0):>9.1f}"
          "   (I/O, popis, čekanje)")

    if skip_reasons:
        print("\n  Razlozi preskakanja:")
        for why, n in skip_reasons.most_common():
            print(f"    {n:>5}  {why}")
    if errors:
        print(f"\n  Greške ({len(errors)}):")
        for ref, err in errors[:15]:
            print(f"    {ref}: {err}")
        if len(errors) > 15:
            print(f"    … i još {len(errors) - 15}")

    # Extrapolation is the point of a sample run, so do the arithmetic here
    # rather than leaving it to be done wrong later.
    if args.sample and n_articles:
        total_rows = sum(len(load_manifest(Path(m), args.include_mismatch))
                         for m in args.manifest)
        f = total_rows / n_articles
        print(f"\n  Projekcija na cijeli korpus ({total_rows} članaka, ×{f:.1f}):")
        print(f"    chunkova   ~{int(n_chunks * f):,}".replace(",", "."))
        print(f"    vrijeme    ~{elapsed * f / 60:.0f} min")
        if args.no_classify:
            print("    (bez klasifikacije — s njom računaj na ~7.750 Haiku poziva)")

    print(f"\n  → {log_path}")
    print(f"{'='*70}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
