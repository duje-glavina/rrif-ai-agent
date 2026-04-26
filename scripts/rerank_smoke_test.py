"""Compare semantic-only vs semantic+rerank on the same questions.

For each query:
  1. Pull top-20 by semantic search (the candidate pool)
  2. Rerank those 20 with BGE-reranker-v2-m3
  3. Print top-5 from each strategy side by side

Tests whether reranking pulls the actually-best chunk to #1 even when
plain semantic search ranks it #2 or #3.
"""
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.embedder import embed_query
from rag.retrieve.rerank import rerank

load_dotenv()


CANDIDATE_POOL = 20  # how many candidates the reranker re-scores
TOP_K = 5            # how many we ultimately show


QUESTIONS = [
    "Kolika je opća stopa PDV-a?",
    "Koja je porezna osnovica kod uvoza dobara?",
    "Što su porezni obveznici i tko nije porezni obveznik?",
    "Kada se PDV obračunava po sniženoj stopi?",
]


def semantic_candidates(query: str, n: int = CANDIDATE_POOL):
    """Pull n nearest chunks by cosine distance. Returns (id, article_no, text, sim)."""
    qvec = embed_query(query)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, article_number, chunk_text,
                       1 - (embedding <=> %s) AS similarity
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (qvec, qvec, n),
            )
            return cur.fetchall()


def main():
    for q in QUESTIONS:
        print("=" * 100)
        print(f"Q: {q}")
        print("=" * 100)

        candidates = semantic_candidates(q, n=CANDIDATE_POOL)

        # Semantic-only top 5 = first 5 of the candidate pool
        print("\n--- Semantic only (top 5 of 20) ---")
        for cid, art, text, sim in candidates[:TOP_K]:
            label = f"Članak {art}" if art else "[preamble]"
            preview = text[:120].replace("\n", " ")
            print(f"  {sim:.4f}  {label:14s}  {preview}...")

        # Reranked top 5
        print(f"\n--- Reranked (rerank top 20 → top {TOP_K}) ---")
        # Build (id, text) tuples for the reranker, also keep article_no for printing
        article_lookup = {cid: art for cid, art, _, _ in candidates}
        rerank_input = [(cid, text) for cid, _, text, _ in candidates]
        ranked = rerank(q, rerank_input, k=TOP_K)
        for cid, text, score in ranked:
            art = article_lookup.get(cid)
            label = f"Članak {art}" if art else "[preamble]"
            preview = text[:120].replace("\n", " ")
            print(f"  {score:.4f}  {label:14s}  {preview}...")
        print()


if __name__ == "__main__":
    main()
