"""Populate the stemmed text column and index it (Route B of the FTS experiment).

Adds `chunk_text_stem`, fills it from `chunk_text` using rag.stem_hr, and
builds a GIN index over it with the 'simple' config. Nothing here needs
superuser or filesystem access, which is the point: if this moves retrieval,
Croatian morphology is not a reason to prefer self-hosted Postgres.

Run from the project root:

    python scripts/exp_stem_corpus.py --dry-run     # sample before/after, no writes
    python scripts/exp_stem_corpus.py               # populate + index
    python scripts/exp_stem_corpus.py --recreate    # re-stem rows already done

Pick the backend explicitly when it matters:

    STEM_BACKEND=crude   python scripts/exp_stem_corpus.py    # fast, laptop
    STEM_BACKEND=classla python scripts/exp_stem_corpus.py    # real, on the GPU box

The query side is already wired: rag/query.py reads FTS_CONFIG=simple|stem and
puts the question through the same normaliser. Use the same STEM_BACKEND for
both — mismatched vocabularies collapse lexical recall without raising
anything.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.stem_hr import backend_name, stem_many  # noqa: E402

load_dotenv()

BATCH = 500

# chunks.id is a uuid, so paging works off an ordered id cursor. Starting from
# the zero uuid avoids a NULL parameter in the WHERE clause, which psycopg
# cannot assign a type to ("could not determine data type of parameter").
ZERO_UUID = "00000000-0000-0000-0000-000000000000"


def add_column(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_text_stem TEXT")
    conn.commit()


def total_rows(conn: psycopg.Connection, recreate: bool) -> int:
    sql = ("SELECT count(*) FROM chunks" if recreate else
           "SELECT count(*) FROM chunks WHERE chunk_text_stem IS NULL")
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()[0]


def fetch_batch(conn: psycopg.Connection, recreate: bool, after_id: str) -> list[tuple]:
    """One page of rows past `after_id`, oldest id first.

    Both modes page by id. The non-recreate mode could instead re-query for
    NULLs each time, but paging keeps the two paths identical and avoids a
    full re-scan per batch as the remaining set shrinks.
    """
    extra = "" if recreate else "AND chunk_text_stem IS NULL"
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id, chunk_text FROM chunks "
            f"WHERE id > %s::uuid {extra} ORDER BY id LIMIT %s",
            (after_id, BATCH),
        )
        return cur.fetchall()


def populate(conn: psycopg.Connection, recreate: bool) -> int:
    """Stem in batches, committing each one so an interrupted run resumes."""
    total = total_rows(conn, recreate)
    if total == 0:
        print("Nothing to do — every row already has chunk_text_stem.")
        return 0

    print(f"Stemming {total:,} rows using the '{backend_name()}' backend...")
    done, started, cursor_id = 0, time.time(), ZERO_UUID

    while True:
        rows = fetch_batch(conn, recreate, cursor_id)
        if not rows:
            break

        ids = [r[0] for r in rows]
        stemmed = stem_many([r[1] for r in rows])

        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE chunks SET chunk_text_stem = %s WHERE id = %s",
                list(zip(stemmed, ids)),
            )
        conn.commit()

        cursor_id = str(ids[-1])
        done += len(rows)
        elapsed = max(time.time() - started, 0.001)
        eta = (total - done) / (done / elapsed) if done else 0
        print(f"  {done:,}/{total:,}  ({done/elapsed:.0f} rows/s, "
              f"~{eta/60:.0f} min left)   ", end="\r", flush=True)

    print(f"\nStemmed {done:,} rows in {(time.time() - started)/60:.1f} min.")
    return done


def build_index(dsn: str) -> None:
    """GIN index matching the expression the query will use.

    CREATE INDEX CONCURRENTLY cannot run inside a transaction block, so this
    opens its own autocommit connection rather than fighting the main one.

    The expression must match the query EXACTLY, cast included, or the planner
    quietly falls back to a sequential scan and you benchmark the wrong thing.
    """
    print("Building GIN index...")
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_fts_stem "
            "ON chunks USING gin (to_tsvector('simple'::regconfig, chunk_text_stem))"
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_size_pretty(pg_relation_size('idx_chunks_fts_stem'))"
            )
            print(f"Index built: {cur.fetchone()[0]}")


def dry_run(conn: psycopg.Connection, n: int) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_text FROM chunks ORDER BY random() LIMIT %s", (n,))
        rows = [r[0] for r in cur.fetchall()]
    print(f"backend: {backend_name()}\n")
    for original, stemmed in zip(rows, stem_many(rows)):
        print(f"BEFORE: {original[:160]}")
        print(f"AFTER : {stemmed[:160]}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Show sample before/after pairs, change nothing.")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--recreate", action="store_true",
                    help="Re-stem rows that already have a value.")
    ap.add_argument("--skip-index", action="store_true")
    args = ap.parse_args()

    dsn = os.environ["DATABASE_URL"]

    with psycopg.connect(dsn) as conn:
        if args.dry_run:
            dry_run(conn, args.samples)
            return
        add_column(conn)
        populate(conn, args.recreate)

    if not args.skip_index:
        build_index(dsn)

    print(
        "\nDone. Next: stem the query side too, then\n"
        "  FTS_CONFIG=stem python -m eval.run_eval --skip-generation --note stem_b\n"
    )


if __name__ == "__main__":
    main()


# ── NOTE: verify the index is actually used ───────────────────────────────────
#
# The query side lives in rag/query.py: FTS_CONFIG picks the column, and
# _build_tsquery runs stem_query() over the question when it is set to 'stem'.
# The column name is interpolated into the SQL string, not bound, because the
# planner only matches an index expression against a constant.
#
# Before trusting any number this produces, confirm the GIN index is hit:
#
#   EXPLAIN (ANALYZE, BUFFERS)
#   SELECT id FROM chunks
#   WHERE to_tsvector('simple', chunk_text_stem)
#         @@ to_tsquery('simple', 'stop | pdv');
#
# You want "Bitmap Index Scan on idx_chunks_fts_stem". A Seq Scan means the
# expression doesn't match the index — usually a cast difference — and at 12k
# rows it will still be fast enough that nothing looks wrong.
