"""Compare semantic-only vs hybrid retrieval on the same questions.

For each query, prints the top 5 results from each strategy side by side
so you can see whether RRF actually helps. The questions are the same as
in retrieval_smoke_test.py, so improvements (or regressions) are directly
comparable to the earlier baseline.
"""
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.embedder import embed_query
from rag.retrieve.hybrid import hybrid_search

load_dotenv()


QUESTIONS = [
    "Kolika je opća stopa PDV-a?",
    "Koja je porezna osnovica kod uvoza dobara?",
    "Što su porezni obveznici i tko nije porezni obveznik?",
    "Kada se PDV obračunava po sniženoj stopi?",
]


def semantic_only(query: str, k: int = 5):
    qvec = embed_query(query)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT article_number, chunk_text,
                       1 - (embedding <=> %s) AS similarity
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (qvec, qvec, k),
            )
            return cur.fetchall()


def main():
    for q in QUESTIONS:
        print("=" * 100)
        print(f"Q: {q}")
        print("=" * 100)

        print("\n--- Semantic only ---")
        for art, text, sim in semantic_only(q):
            label = f"Članak {art}" if art else "[preamble]"
            preview = text[:120].replace("\n", " ")
            print(f"  {sim:.4f}  {label:14s}  {preview}...")

        print("\n--- Hybrid (semantic + FTS, RRF-fused) ---")
        hits = hybrid_search(q, k=5)
        for hit in hits:
            label = f"Članak {hit.article_number}" if hit.article_number else "[preamble]"
            preview = hit.chunk_text[:120].replace("\n", " ")
            sem = f"sem#{hit.semantic_rank}" if hit.semantic_rank else "sem—"
            kw = f"kw#{hit.keyword_rank}" if hit.keyword_rank else "kw—"
            print(f"  rrf={hit.rrf_score:.4f}  {sem:6s} {kw:6s}  {label:14s}  {preview}...")
        print()


if __name__ == "__main__":
    main()
