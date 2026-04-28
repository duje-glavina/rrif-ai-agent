"""Fix PDV zakon chunks — mark the 2013 law as currently valid.

The law was ingested with status='nevazeci' (historical marker from the
ingest script), but the 2013 Zakon o PDV-u is still the base law in force
(with amendments). This script corrects the status so the retriever can
find it when filtering on status='vazeci'.

Run from project root:
    python scripts\fix_pdv_status.py
"""
import psycopg, os
from dotenv import load_dotenv
load_dotenv()

with psycopg.connect(os.environ['DATABASE_URL']) as conn:
    with conn.cursor() as cur:
        # Check current state
        cur.execute("SELECT status, count(*) FROM chunks WHERE category='PDV' AND source_type='zakon' GROUP BY status")
        print("Before:")
        for row in cur.fetchall():
            print(f"  status={row[0]}  count={row[1]}")

        # Fix it
        cur.execute("""
            UPDATE chunks
            SET status = 'vazeci', valid_to = NULL
            WHERE category = 'PDV' AND source_type = 'zakon'
        """)
        print(f"\nUpdated {cur.rowcount} rows.")

        # Verify
        cur.execute("SELECT status, count(*) FROM chunks WHERE category='PDV' AND source_type='zakon' GROUP BY status")
        print("\nAfter:")
        for row in cur.fetchall():
            print(f"  status={row[0]}  count={row[1]}")

    conn.commit()
    print("\nDone. Re-run debug_retrieval.py to verify.")
