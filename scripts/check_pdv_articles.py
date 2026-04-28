"""Quick script to inspect PDV chunks in the DB and find correct article numbers.
Run from project root: python scripts\check_pdv_articles.py
"""
import psycopg, os
from dotenv import load_dotenv
load_dotenv()

with psycopg.connect(os.environ['DATABASE_URL']) as conn:
    with conn.cursor() as cur:

        print("=== PDV zakon articles (stopa / rate) ===")
        cur.execute("""
            SELECT article_number, chunk_text
            FROM chunks
            WHERE category='PDV' AND source_type='zakon'
            AND chunk_text ILIKE '%stopa%'
            ORDER BY article_number
            LIMIT 10
        """)
        for row in cur.fetchall():
            print(f"art={row[0]}: {row[1][:150]}")

        print("\n=== PDV zakon articles (uvoz / import) ===")
        cur.execute("""
            SELECT article_number, chunk_text
            FROM chunks
            WHERE category='PDV' AND source_type='zakon'
            AND chunk_text ILIKE '%uvoz%'
            ORDER BY article_number
            LIMIT 10
        """)
        for row in cur.fetchall():
            print(f"art={row[0]}: {row[1][:150]}")

        print("\n=== PDV zakon articles (porezni obveznik) ===")
        cur.execute("""
            SELECT article_number, chunk_text
            FROM chunks
            WHERE category='PDV' AND source_type='zakon'
            AND chunk_text ILIKE '%porezni obveznik%'
            ORDER BY article_number
            LIMIT 10
        """)
        for row in cur.fetchall():
            print(f"art={row[0]}: {row[1][:150]}")

        print("\n=== All PDV zakon article numbers ===")
        cur.execute("""
            SELECT DISTINCT article_number
            FROM chunks
            WHERE category='PDV' AND source_type='zakon'
            ORDER BY article_number
        """)
        print([row[0] for row in cur.fetchall()])
