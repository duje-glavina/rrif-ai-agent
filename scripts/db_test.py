"""Quick check: connect to Postgres, register vector type, round-trip a vector."""
import os
from dotenv import load_dotenv
import psycopg
from pgvector.psycopg import register_vector

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

with psycopg.connect(DATABASE_URL) as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT '[1,2,3]'::vector;")
        result = cur.fetchone()
        print(f"Got vector back from Postgres: {result[0]}")
        print(f"Type: {type(result[0])}")