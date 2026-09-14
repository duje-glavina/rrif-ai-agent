# Building `rrif_rag_v2` alongside `rrif_rag`

*Duje · 7 Sep 2026 · companion to `claude/retrieval_experiments_runbook.md`*

The re-ingest changes seven things at once — chunk size, title extraction, text
cleanup, page-furniture removal, classification input, the ads filter, chunk
positions. If any of them makes retrieval worse, the only way to find out is to
still have the old corpus to compare against. So v2 is built as a **separate
database**, not in place.

Everything in the codebase reads `DATABASE_URL` from the environment, so
switching between the two costs no code change at all — just a variable in
front of the command.

Disk cost is about 250 MB. That is the entire price of not being able to make
an irreversible mistake tonight.

---

## 0. Before you start: one blocker

`scripts/ingest_rrif_articles.py` line 34:

```python
from rag.classifier import STATUS_VALID, STATUS_INVALID
```

`rag/classifier.py` defines `STATUS_VALID = "vazeci"` and **nothing else**.
There is no `STATUS_INVALID`. The ingest script raises `ImportError` before it
reads a single PDF.

Add to `rag/classifier.py`, next to `STATUS_VALID`:

```python
STATUS_INVALID = "nevazeci"
```

Check the string against what is already in the corpus before trusting it:

```bash
psql "$V1" -c "SELECT DISTINCT status FROM chunks;"
```

It must match exactly — `_build_where` compares against `status = 'vazeci'`, so
a typo here quietly removes rows from every current-state query.

---

## 1. Work out the two connection strings

Never edit `.env` for this. `.env` stays pointed at v1 until v2 has won, so
that anything you run without thinking about it hits the known-good corpus.

```bash
cd ~/projekti/rrif-ai-agent

V1=$(grep -m1 '^DATABASE_URL=' .env | cut -d= -f2- | tr -d '"'"'"' )
V2="${V1%/*}/rrif_rag_v2"

echo "v1: $V1"
echo "v2: $V2"
```

`${V1%/*}` strips everything after the last `/`, i.e. the database name.
**If your URL has query parameters** (`?sslmode=require`, `?options=...`) this
will mangle it — in that case just write `V2` out by hand.

These two variables live only in your current shell. Open a new terminal and
you have to set them again; that is deliberate.

---

## 2. Create the database, owned by `ai_user`

```bash
sudo -u postgres createdb -O ai_user rrif_rag_v2
```

The `-O ai_user` matters. In v1 the tables ended up owned by `postgres`, which
is why `ALTER TABLE chunks ADD COLUMN` failed with `must be owner of table
chunks` and needed a superuser to fix. Setting the owner at creation avoids
repeating that.

---

## 3. Load the schema

```bash
sudo -u postgres psql -d rrif_rag_v2 < db/schema_full.sql
```

Note the `<` redirect rather than `psql -f db/schema_full.sql`. With `-f`, the
**postgres** user opens the file, and it usually cannot traverse
`/home/kron/...`, so you get a confusing permission error about a file that is
plainly there. With `<`, your own shell opens it and feeds it on stdin.

Run it as `postgres` because the dump contains `CREATE EXTENSION vector` and
`ALTER ... OWNER TO postgres`, both of which need superuser.

The dump begins and ends with `\restrict` / `\unrestrict` — a pg_dump 17.9
safety wrapper. Fine on psql 17.x; on an older psql it errors on the first
line, in which case delete those two lines.

Then hand everything to `ai_user`:

```bash
sudo -u postgres psql -d rrif_rag_v2 <<'SQL'
ALTER TABLE chunks   OWNER TO ai_user;
ALTER TABLE queries  OWNER TO ai_user;
ALTER TABLE feedback OWNER TO ai_user;
ALTER FUNCTION set_updated_at() OWNER TO ai_user;
GRANT ALL ON SCHEMA public TO ai_user;
SQL
```

---

## 4. Check the schema actually matches

```bash
psql "$V1" -c '\d chunks' > /tmp/v1_schema.txt
psql "$V2" -c '\d chunks' > /tmp/v2_schema.txt
diff /tmp/v1_schema.txt /tmp/v2_schema.txt
```

Expect exactly two differences, both **v1-only**:

* the `chunk_text_stem` column
* the `idx_chunks_fts_stem` index

Those came from the stemming experiment, which measured no effect. Leaving
them out of v2 is intended. If stemming is ever worth retrying on v2, run
`scripts/exp_stem_corpus.py` against it then.

Anything else in that diff means the committed `db/schema_full.sql` has drifted
from the live v1 database — stop and reconcile before ingesting, because you
would be comparing two corpora with different shapes.

---

## 5. Drop the HNSW index before loading

```bash
psql "$V2" -c "DROP INDEX IF EXISTS idx_chunks_embedding_hnsw;"
```

The schema dump creates the index on an empty table. Every subsequent `INSERT`
then updates the HNSW graph one vector at a time, which is both much slower
across ~30k inserts and produces a worse graph than one built in a single pass
over the finished set. Drop it, load, rebuild.

Leave the GIN full-text index in place — it is cheap to maintain incrementally.

---

## 6. Ingest

```bash
DATABASE_URL="$V2" python scripts/ingest_rrif_articles.py "/path/to/Arhiva" --dry-run
DATABASE_URL="$V2" python scripts/ingest_rrif_articles.py "/path/to/Arhiva"
```

Do the dry run first. It prints chunk counts and the extracted title per file
without touching the database, which is the fastest way to see whether the new
title extractor works — the titles are right there in the output.

Also ingest the law, so v2 is a complete corpus rather than magazine-only:

```bash
DATABASE_URL="$V2" python scripts/ingest_pdv_2013.py
```

Budget: two to three hours. Your remembered 90 minutes was for a run that
discarded half its tokens at the embedder; the new chunker keeps them, and
there will be roughly twice as many chunks. Classification is ~1,300–1,800
Haiku calls at 20 chunks per call — **check the API credits before starting**,
not forty minutes in.

---

## 7. Rebuild the index and update the statistics

```bash
psql "$V2" <<'SQL'
SET maintenance_work_mem = '4GB';
CREATE INDEX idx_chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
ANALYZE chunks;
SQL
```

Both statements must be in the **same session** for `SET` to apply — hence the
heredoc rather than two `-c` calls.

If the Postgres log says `hnsw graph no longer fits into maintenance_work_mem`,
the build spilled to disk and took far longer than it needed to. Raise the
value and rebuild. (This is the same parameter that the Aiven probe in the
other runbook is about — worth noting what value you actually needed.)

`ANALYZE` is not optional. Without fresh statistics the planner may choose a
sequential scan over the new table and the latency comparison becomes
meaningless.

---

## 8. Sanity-check before spending an eval on it

```bash
psql "$V2" <<'SQL'
SELECT source_type, status, count(*) FROM chunks GROUP BY 1,2 ORDER BY 3 DESC;
SELECT pg_size_pretty(pg_database_size('rrif_rag_v2'));
SQL

DATABASE_URL="$V2" python scripts/audit_corpus.py
DATABASE_URL="$V2" PYTHONPATH=. python scripts/check_embed_truncation.py
```

The audit is the direct read on whether the re-ingest did what it was for.
Targets: `source label is body text` near 0% (was 96.6%), soft hyphens near 0%
(was 54.2%), truncation near 0% (was 55.6%).

If truncation is still material, the token cap is not being applied and there
is no point running the eval yet.

---

## 9. The A/B

```bash
python -m eval.run_eval --skip-generation --note v1_base
DATABASE_URL="$V2" python -m eval.run_eval --skip-generation --note v2_base

python -m eval.compare_runs eval/results/*v1_base*.json eval/results/*v2_base*.json
```

The summary and the results JSON both record which database produced them, so
these cannot be mixed up later.

Re-run the retrieval-mode ablation on v2 as well. The `tight` filter currently
loses to `domain` because chunk subdomains were assigned by Haiku reading only
`text[:600]`; if v2 classifies from the whole chunk, that finding may reverse:

```bash
DATABASE_URL="$V2" RETRIEVAL_MODE=domain python -m eval.run_eval --skip-generation --note v2_domain
```

---

## 10. Switching over, and rolling back

Nothing switches automatically. When v2 wins on the numbers, change the one
line in `.env`:

```
DATABASE_URL=postgresql://ai_user:...@localhost/rrif_rag_v2
```

Rolling back is putting the old line back. That is the whole point of this
exercise.

Keep `rrif_rag` until v2 has survived a full evaluation *and* a generation run
with Sonnet in the loop — retrieval improving does not by itself prove answers
improved. Only then:

```bash
sudo -u postgres dropdb rrif_rag        # not before
```

---

## What NOT to do

* **Do not edit `.env` before the A/B.** Any command you run without thinking
  should hit the corpus you trust.
* **Do not `TRUNCATE chunks` in v1** to "start clean". That destroys the
  baseline in the same step that changes seven variables.
* **Do not ingest into v2 twice.** There is no de-duplication in
  `pipeline.ingest()` — it inserts unconditionally, so a re-run doubles the
  corpus silently. If a run fails halfway, drop and recreate the database
  rather than resuming.
* **Do not compare a v2 run against a v1 results file from before 6 Sep.** The
  grader changed twice that day (morphology-aware matching, ČLANCI/ZAKONI
  split). Re-run the v1 baseline fresh, in the same session as the v2 one.
