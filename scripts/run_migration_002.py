import psycopg, os
from dotenv import load_dotenv
load_dotenv()

sql = open('rag/migrations/002_add_domain_subdomain.sql', encoding='utf-8').read()
with psycopg.connect(os.environ['DATABASE_URL']) as conn:
    conn.execute(sql)
    conn.commit()
    print('Migration complete.')

    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='chunks' ORDER BY ordinal_position"
        )
        cols = [r[0] for r in cur.fetchall()]
        print('Columns now:', cols)
