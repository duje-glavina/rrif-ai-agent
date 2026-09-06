"""Measure how badly the corpus was ingested, before deciding to redo it.

Re-ingesting is a day's work and it invalidates every baseline we have, so it
should be a decision with numbers behind it rather than a hunch. This counts
the defects that are already visible in retrieved chunks:

  * font-encoding damage — pages extracted with a shifted font map, e.g.
    "QSPTJOBD öOBODJKF" which is "PROSINAC FINANCIJE" with every letter moved
    one place. Such a chunk is unretrievable and unreadable, and it embeds to
    noise.
  * soft hyphens and hyphen-newline pairs left by the PDF's line breaking,
    which split words in the middle: "djelat­nostima", "zahti-jeva".
  * page furniture inside the body — source URLs, "9 of 71", export
    timestamps — which the reranker scores as content.
  * `source` labels that are a paragraph of body text rather than a title.
  * missing chunk positions, duplicate chunks, and the length distribution.

Run it from the project root:

    python scripts/audit_corpus.py
    python scripts/audit_corpus.py --samples 5     # show example chunks

Nothing here writes. Run it again after any re-ingest and compare.
"""
from __future__ import annotations

import argparse
import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

# Croatian function words. Any long passage of real Croatian contains several;
# a long chunk containing none of them is almost certainly not Croatian text.
FUNCTION_WORDS = r'\m(je|se|na|za|od|iz|koji|koja|nije|te|ili|ako)\M'

CHECKS: list[tuple[str, str, str]] = [
    (
        "font-encoding damage",
        f"""SELECT count(*) FROM chunks
            WHERE length(chunk_text) > 400
              AND chunk_text !~* '{FUNCTION_WORDS}'""",
        "Long chunks containing no Croatian function word at all. These embed "
        "to noise and can never be retrieved correctly.",
    ),
    (
        "soft hyphens (U+00AD)",
        r"SELECT count(*) FROM chunks WHERE chunk_text LIKE '%' || chr(173) || '%'",
        "Words split by the PDF's line breaking. Breaks both FTS and keyword "
        "grading; invisible when you print the text.",
    ),
    (
        "hyphen-newline splits",
        r"SELECT count(*) FROM chunks WHERE chunk_text ~ '\w-\n\w'",
        "Same problem, different encoding of it.",
    ),
    (
        "page furniture: URLs",
        "SELECT count(*) FROM chunks WHERE chunk_text ILIKE '%narodne-novine.nn.hr%'"
        " OR chunk_text ~ 'https?://'",
        "Source links inside the body text.",
    ),
    (
        "page furniture: 'N of M'",
        r"SELECT count(*) FROM chunks WHERE chunk_text ~ '\m\d+ of \d+\M'",
        "PDF viewer pagination captured as content.",
    ),
    (
        "page furniture: export timestamp",
        r"SELECT count(*) FROM chunks WHERE chunk_text ~ '\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2} [AP]M'",
        "The moment someone printed the PDF, now part of the knowledge base.",
    ),
    (
        "source label is body text",
        "SELECT count(*) FROM chunks WHERE length(source) > 120",
        "The `source` field should be a citable label. Over ~120 characters it "
        "is a paragraph, which means the title was never extracted.",
    ),
    (
        "article_number NULL (članak)",
        "SELECT count(*) FROM chunks WHERE source_type = 'članak' AND article_number IS NULL",
        "Magazine chunks with no position, so neighbouring chunks cannot be "
        "found and adjacency cannot be reasoned about.",
    ),
    (
        "duplicate chunk_text",
        """SELECT coalesce(sum(c - 1), 0) FROM (
               SELECT count(*) AS c FROM chunks GROUP BY md5(chunk_text) HAVING count(*) > 1
           ) d""",
        "Redundant copies competing for the same top-20 slots.",
    ),
    (
        "very short chunks (<200 chars)",
        "SELECT count(*) FROM chunks WHERE length(chunk_text) < 200",
        "Too little context to answer anything, but they still occupy "
        "candidate slots.",
    ),
    (
        "very long chunks (>4000 chars)",
        "SELECT count(*) FROM chunks WHERE length(chunk_text) > 4000",
        "Dilute embeddings — one vector covering several unrelated topics.",
    ),
]

SAMPLE_SQL = {
    "font-encoding damage":
        f"""SELECT left(chunk_text, 200) FROM chunks
            WHERE length(chunk_text) > 400 AND chunk_text !~* '{FUNCTION_WORDS}'
            LIMIT %s""",
    "source label is body text":
        "SELECT left(source, 200) FROM chunks WHERE length(source) > 120 LIMIT %s",
    "page furniture: 'N of M'":
        r"""SELECT left(chunk_text, 200) FROM chunks
            WHERE chunk_text ~ '\m\d+ of \d+\M' LIMIT %s""",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=0,
                    help="Show this many example chunks for the checks that have them.")
    args = ap.parse_args()

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks")
        total = cur.fetchone()[0]

        print(f"\n{total:,} chunks\n")
        print(f"  {'check':<34} {'count':>8} {'share':>8}")
        print("  " + "-" * 52)

        results = []
        for name, sql, why in CHECKS:
            cur.execute(sql)
            n = cur.fetchone()[0] or 0
            results.append((name, n, why))
            share = f"{100 * n / total:.1f}%" if total else "—"
            flag = "  ←" if n and (n / total) > 0.02 else ""
            print(f"  {name:<34} {n:>8,} {share:>8}{flag}")

        print("\n  ← marks anything affecting more than 2% of the corpus.\n")

        for name, n, why in results:
            if n:
                print(f"  {name} ({n:,})")
                print(f"    {why}")
        print()

        cur.execute("""
            SELECT source_type,
                   count(*),
                   min(length(chunk_text)),
                   percentile_disc(0.5) WITHIN GROUP (ORDER BY length(chunk_text)),
                   percentile_disc(0.95) WITHIN GROUP (ORDER BY length(chunk_text)),
                   max(length(chunk_text))
            FROM chunks GROUP BY source_type ORDER BY 2 DESC
        """)
        print(f"  {'source_type':<14} {'n':>8} {'min':>7} {'p50':>7} {'p95':>7} {'max':>8}")
        print("  " + "-" * 56)
        for st, n, mn, p50, p95, mx in cur.fetchall():
            print(f"  {str(st):<14} {n:>8,} {mn:>7} {p50:>7} {p95:>7} {mx:>8}")

        cur.execute("""
            SELECT source_type, status, citable, count(*)
            FROM chunks GROUP BY 1, 2, 3 ORDER BY 4 DESC
        """)
        print(f"\n  {'source_type':<14} {'status':<12} {'citable':<8} {'n':>8}")
        print("  " + "-" * 46)
        for st, status, citable, n in cur.fetchall():
            print(f"  {str(st):<14} {str(status):<12} {str(citable):<8} {n:>8,}")

        if args.samples:
            for name, sql in SAMPLE_SQL.items():
                cur.execute(sql, (args.samples,))
                rows = cur.fetchall()
                if not rows:
                    continue
                print(f"\n  ── samples: {name} " + "─" * 30)
                for (txt,) in rows:
                    print(f"    {txt!r}")
        print()


if __name__ == "__main__":
    main()
