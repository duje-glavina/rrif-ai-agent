import psycopg, os
from dotenv import load_dotenv
load_dotenv()

with psycopg.connect(os.environ['DATABASE_URL']) as conn:
    with conn.cursor() as cur:
        print("=== CATEGORIES ===")
        cur.execute("SELECT category, count(*) FROM chunks GROUP BY category ORDER BY count(*) DESC")
        for row in cur.fetchall():
            print(row)

        print("\n=== SOURCE_TYPES ===")
        cur.execute("SELECT source_type, count(*) FROM chunks GROUP BY source_type ORDER BY count(*) DESC")
        for row in cur.fetchall():
            print(row)

        print("\n=== SAMPLE CHUNK (first PDV-ish) ===")
        cur.execute("SELECT id, category, source_type, source, chunk_text FROM chunks LIMIT 3")
        for row in cur.fetchall():
            print(f"id={row[0]}, cat={row[1]}, type={row[2]}, source={row[3]}")
            print(f"  text: {row[4][:100]}")
