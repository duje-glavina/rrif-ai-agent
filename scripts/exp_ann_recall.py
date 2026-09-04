"""Measure HNSW recall against exact nearest-neighbour search.

This is the diagnostic that separates the three things which all look like
"retrieval got worse":

  1. ANN recall loss  — the index stopped finding what brute force finds
  2. Ranking failure  — the right chunk IS retrieved, but ranks below others
  3. Temporal ambiguity — the right chunk is retrieved, wrong year

Only (1) is fixed by tuning m / ef_construction / ef_search. If recall is
high and top-1 is still bad, no amount of index tuning will help and you are
looking at a ranking problem instead.

At ~12k chunks expect recall near 100% — that is the baseline worth recording
now, so the number at 100k / 575k means something later.

Run from the project root:

    python scripts/exp_ann_recall.py
    python scripts/exp_ann_recall.py --k 20 --ef-search 100
    python scripts/exp_ann_recall.py --golden eval/golden_set_v3.yaml
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import psycopg
import yaml
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.embedder import embed_query  # noqa: E402

load_dotenv()

DEFAULT_GOLDEN = "eval/golden_set_legacy1.yaml"


def load_queries(path: str) -> list[tuple[str, str]]:
    """Return (id, query) pairs from a golden set file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [(entry["id"], entry["query"]) for entry in data]


def search(conn, qvec, k: int, exact: bool, ef_search: int | None) -> tuple[list, float]:
    """Top-k chunk ids by cosine distance, plus elapsed seconds.

    exact=True disables index and bitmap scans for this transaction, forcing a
    sequential scan — which is the true nearest-neighbour answer to compare
    the index against.
    """
    with conn.transaction():
        with conn.cursor() as cur:
            if exact:
                cur.execute("SET LOCAL enable_indexscan = off")
                cur.execute("SET LOCAL enable_bitmapscan = off")
            elif ef_search is not None:
                cur.execute("SET LOCAL hnsw.ef_search = %s", (ef_search,))

            started = time.perf_counter()
            cur.execute(
                "SELECT id FROM chunks WHERE embedding IS NOT NULL "
                "ORDER BY embedding <=> %s LIMIT %s",
                (qvec, k),
            )
            rows = [r[0] for r in cur.fetchall()]
            return rows, time.perf_counter() - started


def confirm_plan(conn, qvec, k: int) -> None:
    """Sanity check that the non-exact path really uses the HNSW index.

    If this reports a Seq Scan, every recall number below is 1.0 by
    construction and the run is meaningless.
    """
    with conn.cursor() as cur:
        cur.execute(
            "EXPLAIN SELECT id FROM chunks WHERE embedding IS NOT NULL "
            "ORDER BY embedding <=> %s LIMIT %s",
            (qvec, k),
        )
        plan = "\n".join(r[0] for r in cur.fetchall())
    uses_index = "idx_chunks_embedding_hnsw" in plan or "Index Scan" in plan
    print("Index in use: " + ("yes" if uses_index else "NO — results below are meaningless"))
    if not uses_index:
        print(plan)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--k", type=int, default=20,
                    help="Candidate pool size to compare (default 20, matching CANDIDATES).")
    ap.add_argument("--ef-search", type=int, default=None,
                    help="Override hnsw.ef_search for the approximate run.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    queries = load_queries(args.golden)
    print(f"Golden set: {args.golden}  ({len(queries)} queries)")
    print(f"k = {args.k}" + (f", hnsw.ef_search = {args.ef_search}" if args.ef_search else ""))
    print()

    recalls: list[float] = []
    t_exact: list[float] = []
    t_ann: list[float] = []

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
            print(f"Corpus: {cur.fetchone()[0]:,} embedded chunks\n")

        confirm_plan(conn, embed_query(queries[0][1]), args.k)

        for qid, question in queries:
            qvec = embed_query(question)
            exact_ids, te = search(conn, qvec, args.k, exact=True, ef_search=None)
            ann_ids, ta = search(conn, qvec, args.k, exact=False, ef_search=args.ef_search)

            overlap = len(set(exact_ids) & set(ann_ids))
            recall = overlap / len(exact_ids) if exact_ids else 0.0
            recalls.append(recall)
            t_exact.append(te)
            t_ann.append(ta)

            flag = "  <-- LOW" if recall < 0.95 else ""
            print(f"  {qid:28s} recall@{args.k} = {recall:5.1%}   "
                  f"ann {ta*1000:6.1f}ms  exact {te*1000:7.1f}ms{flag}")

            if args.verbose and recall < 1.0:
                missed = [str(i) for i in exact_ids if i not in set(ann_ids)]
                print(f"      missed by the index: {', '.join(missed)}")

    print()
    print(f"  mean recall@{args.k}   {statistics.mean(recalls):.1%}")
    print(f"  min  recall@{args.k}   {min(recalls):.1%}")
    print(f"  mean ann latency    {statistics.mean(t_ann)*1000:.1f} ms")
    print(f"  mean exact latency  {statistics.mean(t_exact)*1000:.1f} ms")
    print()

    mean = statistics.mean(recalls)
    if mean >= 0.98:
        print("  Index is not your problem. Any top-1 weakness is ranking or")
        print("  temporal ambiguity — tuning m/ef_construction will not help.")
    elif mean >= 0.90:
        print("  Mild recall loss. Try raising hnsw.ef_search first (session")
        print("  variable, free to test) before rebuilding the index.")
    else:
        print("  Real recall loss. Rebuild with m=32, ef_construction=128 and")
        print("  re-measure. Some of your missing top-1 is the index, not ranking.")


if __name__ == "__main__":
    main()
