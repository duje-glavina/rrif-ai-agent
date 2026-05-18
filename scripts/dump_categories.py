# scripts/dump_categories.py
import psycopg, os
from dotenv import load_dotenv
load_dotenv()

SUBDOMAINS = ['PDV', 'dohodak', 'dobit', 'plaće', 'knjiženje', 'porezi']

with psycopg.connect(os.environ['DATABASE_URL']) as conn:
    with conn.cursor() as cur:
        for sd in SUBDOMAINS:
            cur.execute("""
                SELECT source, article_number, chunk_text
                FROM chunks
                WHERE subdomain = %s
                  AND status = 'vazeci'
                  AND source_type = 'članak'
                ORDER BY source
                LIMIT 20
            """, (sd,))
            rows = cur.fetchall()
            with open(f'dump_{sd}.txt', 'w', encoding='utf-8') as f:
                for source, art, text in rows:
                    f.write(f"=== {source} | art={art} ===\n")
                    f.write(text[:500])
                    f.write("\n\n")
            print(f"{sd}: {len(rows)} chunks dumped")