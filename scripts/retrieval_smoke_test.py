"""Smoke test: ask a question, retrieve nearest chunks, print top 5.

No classifier, no BM25, no rerank. Just plain semantic search to verify
embeddings + pgvector are returning sensible results.
"""
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.embedder import embed_query

load_dotenv()


QUESTIONS = [
    "Kolika je opća stopa PDV-a?",
    "Koja je porezna osnovica kod uvoza dobara?",
    "Što su porezni obveznici i tko nije porezni obveznik?",
    "Kada se PDV obračunava po sniženoj stopi?",
]


def search(query: str, k: int = 5) -> None:
    print("=" * 80)
    print(f"Q: {query}")
    print("-" * 80)

    qvec = embed_query(query)

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            # Cosine distance: smaller = more similar
            # 1 - distance = cosine similarity (higher = more similar)
            cur.execute(
                """
                SELECT
                    article_number,
                    chunk_text,
                    1 - (embedding <=> %s) AS similarity
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (qvec, qvec, k),
            )
            for art, text, sim in cur.fetchall():
                preview = text[:200].replace("\n", " ")
                label = f"Članak {art}" if art else "[preamble]"
                print(f"  {sim:.4f}  {label:14s}  {preview}...")
    print()


if __name__ == "__main__":
    for q in QUESTIONS:
        search(q)