"""Why does the answer land in the top 5 but not in one chunk?

    bar jedan pojam, top-5   100.0%    the material is always retrieved
    ODGOVOR top-5             81.1%    ...whole, in a single chunk, 81% of the time

Seven questions live in that gap. The assumption has been "chunk boundaries cut
the answer in half", but that is a hypothesis, and there are three other things
it could be:

  SPLIT      the keywords are spread across two or more retrieved chunks —
             the chunker cut mid-answer. This is the one we can fix.
  ADJACENT   same, and the two chunks are consecutive in the same document,
             which makes it a chunk-size or overlap problem specifically.
  MISSING    a keyword appears in no retrieved chunk at all. Then it is a
             retrieval failure, or the golden set is demanding a phrasing the
             corpus never uses — check the keyword before blaming the chunker.

Run it against any results JSON produced with --skip-generation (retrieved_meta
carries the chunk text, so no database is needed for the diagnosis):

    python -m eval.diag_split eval/results/<run>.json

Add --db to also pull the neighbouring chunks of the best-matching chunk out of
Postgres, which is what tells you whether the missing half was one chunk away.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import yaml

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.yaml"


def _norm(s: str) -> str:
    """Identical to the grader in run_eval.py — must not drift from it."""
    s = s.lower().replace(" ", " ").replace("­", "")
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"\s+%", "%", s)


def load_keywords() -> dict[str, list[str]]:
    raw = yaml.safe_load(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    items = raw["questions"] if isinstance(raw, dict) and "questions" in raw else raw
    return {
        it["id"]: [k for k in (it.get("expected_keywords") or []) if k]
        for it in items
    }


def _pos(m: dict) -> int | None:
    """Magazine chunks store their position within the article in article_number."""
    try:
        return int(str(m.get("article_number")).strip())
    except (TypeError, ValueError):
        return None


def _fmt_chunk(rank: int, m: dict) -> str:
    return (f"#{rank} {str(m.get('chunk_id'))[:8]}  {m.get('source_type')}  "
            f"{m.get('source')}  pos={m.get('article_number')}  "
            f"score={m.get('rerank_score')}")


def analyse(q: dict, keywords: list[str]) -> None:
    metas = q.get("retrieved_meta") or []
    kws = [(k, _norm(k)) for k in keywords]
    texts = [_norm(m.get("chunk_text") or "") for m in metas]

    # keyword × rank hit matrix
    hits: dict[str, list[int]] = {
        k: [i for i, t in enumerate(texts) if nk in t] for k, nk in kws
    }
    missing = [k for k, where in hits.items() if not where]
    present = {k: w for k, w in hits.items() if w}

    print("=" * 78)
    print(f"{q['id']}   {q.get('query', '')}")
    print("-" * 78)
    width = max((len(k) for k in hits), default=10)
    for k, where in hits.items():
        marks = "".join("█" if i in where else "·" for i in range(len(metas)))
        print(f"  {k:<{width}}  {marks}  {'—' if not where else where}")
    print(f"  {'':<{width}}  {''.join(str(i+1) for i in range(len(metas)))}  ← rank")
    print()

    if missing:
        print(f"  MISSING: {missing}")
        print("  Not in any retrieved chunk. Either retrieval missed the passage,")
        print("  or the golden set expects a phrasing the corpus doesn't use.")
        print("  Read the keyword before blaming the chunker.")
    if len(present) >= 2:
        carriers = sorted({w[0] for w in present.values()})
        if len(carriers) >= 2:
            a, b = metas[carriers[0]], metas[carriers[1]]
            pa, pb = _pos(a), _pos(b)
            same_doc = a.get("source") == b.get("source")
            adjacent = same_doc and pa is not None and pb is not None and abs(pa - pb) == 1
            print(f"  {'ADJACENT' if adjacent else 'SPLIT'}: keywords spread across "
                  f"ranks {[c + 1 for c in carriers]}")
            print(f"    {_fmt_chunk(carriers[0] + 1, a)}")
            print(f"    {_fmt_chunk(carriers[1] + 1, b)}")
            if adjacent:
                print("    → consecutive chunks of the same document. This is a")
                print("      chunk-size / overlap problem, and the cheapest fix on")
                print("      the board.")
            elif same_doc:
                print("    → same document, non-consecutive positions.")
            else:
                print("    → different documents; the answer may legitimately need both.")
            lo = (a.get("chunk_text") or "")[-260:]
            hi = (b.get("chunk_text") or "")[:260]
            print(f"\n    …tail of #{carriers[0]+1}: …{lo.strip()}")
            print(f"    head of #{carriers[1]+1}: {hi.strip()}…")
    print()


def db_neighbours(chunk_id: str) -> None:
    import psycopg
    from dotenv import load_dotenv
    load_dotenv()
    sql = """
        WITH target AS (SELECT source, article_number FROM chunks WHERE id = %s::uuid)
        SELECT c.id, c.article_number, left(c.chunk_text, 300)
        FROM chunks c, target t
        WHERE c.source = t.source
        ORDER BY nullif(regexp_replace(c.article_number, '\\D', '', 'g'), '')::int
        """
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute(sql, (chunk_id,))
        for cid, pos, text in cur.fetchall():
            mark = "→" if str(cid) == chunk_id else " "
            print(f"{mark} pos={pos}  {str(cid)[:8]}  {text!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="A --skip-generation results JSON")
    ap.add_argument("--db", metavar="CHUNK_ID",
                    help="Instead of the report, dump every chunk of that chunk's "
                         "document in order, so you can see the boundary.")
    ap.add_argument("--also-top1", action="store_true",
                    help="Also report questions that make top-5 but miss top-1.")
    args = ap.parse_args()

    if args.db:
        db_neighbours(args.db)
        return

    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    kw_by_id = load_keywords()

    split = [q for q in data["per_question"]
             if q.get("gradeable_content")
             and q.get("content_any_top_k") and not q.get("content_top_k")]
    print(f"\n{len(split)} questions retrieve the material but never whole in one chunk\n")
    for q in split:
        analyse(q, kw_by_id.get(q["id"], []))

    if args.also_top1:
        near = [q for q in data["per_question"]
                if q.get("gradeable_content")
                and q.get("content_top_k") and not q.get("content_top_1")]
        print(f"\n{len(near)} more have it whole, but not at rank 1 "
              f"(reranker, not chunking)\n")
        for q in near:
            analyse(q, kw_by_id.get(q["id"], []))


if __name__ == "__main__":
    main()
