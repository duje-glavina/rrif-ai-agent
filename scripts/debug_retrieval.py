"""Debug script — tests retrieval directly without the full pipeline."""
import os, sys
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

import psycopg
from pgvector.psycopg import register_vector
from rag.embedder import embed_query

question = "Koji je rok za predaju PDV obrasca?"
qvec = embed_query(question)
print(f"Query vector shape: {qvec.shape}")

# Test 1: no filters at all
with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, category, source_type, chunk_text FROM chunks ORDER BY embedding <=> %s LIMIT 5",
            (qvec,)
        )
        rows = cur.fetchall()
        print(f"\nTest 1 — no filter, top 5 by similarity:")
        for r in rows:
            print(f"  cat={r[1]} type={r[2]} text={r[3][:80]!r}")

# Test 2: with category=PDV filter
with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM chunks WHERE citable = TRUE AND category = %s AND status = 'važeći'",
            ("PDV",)
        )
        print(f"\nTest 2 — count with category=PDV AND status=važeći: {cur.fetchone()[0]}")

        cur.execute(
            "SELECT count(*) FROM chunks WHERE citable = TRUE AND category = %s",
            ("PDV",)
        )
        print(f"Test 3 — count with category=PDV (no status filter): {cur.fetchone()[0]}")

        cur.execute("SELECT DISTINCT status FROM chunks WHERE category = 'PDV'")
        print(f"Test 4 — status values for PDV chunks: {[r[0] for r in cur.fetchall()]}")
