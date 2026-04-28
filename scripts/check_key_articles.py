"""Inspect specific PDV articles to verify golden set expected_articles.
Run: python scripts\check_key_articles.py
"""
import psycopg, os
from dotenv import load_dotenv
load_dotenv()

ARTICLES_TO_CHECK = ['6', '7', '35', '38', '85']

with psycopg.connect(os.environ['DATABASE_URL']) as conn:
    with conn.cursor() as cur:
        for art in ARTICLES_TO_CHECK:
            cur.execute("""
                SELECT article_number, chunk_text
                FROM chunks
                WHERE category='PDV' AND source_type='zakon'
                AND article_number = %s
            """, (art,))
            rows = cur.fetchall()
            print(f"\n{'='*60}")
            print(f"ARTICLE {art} ({len(rows)} chunks)")
            for row in rows:
                print(row[1][:300])

        # Also check what the 'porezi' articles look like for dohodak
        print(f"\n{'='*60}")
        print("SAMPLE 'porezi' category chunks (source types):")
        cur.execute("""
            SELECT DISTINCT source_type, source
            FROM chunks
            WHERE category='porezi'
            LIMIT 10
        """)
        for row in cur.fetchall():
            print(f"  type={row[0]}  source={row[1][:80]}")

        # Check plaće category
        print(f"\n{'='*60}")
        print("SAMPLE 'plaće' category chunks:")
        cur.execute("""
            SELECT DISTINCT source_type, source
            FROM chunks
            WHERE category='plaće'
            LIMIT 5
        """)
        for row in cur.fetchall():
            print(f"  type={row[0]}  source={row[1][:80]}")
