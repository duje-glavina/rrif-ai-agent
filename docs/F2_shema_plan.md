# F2 schema: laws alongside articles

*Duje · 7 Sep 2026 · design note, not a work order*

Two content types, one table. Magazine articles chunk at ~400 tokens and answer
questions on their own. Laws do not: the unit of meaning is the **članak**, and
half a članak is often worse than useless, because the exception lives in
stavak (3) and the rule in stavak (1).

This note says what the schema should become, what to do about it **now** (very
little, on purpose), and what to leave until F2. Laws are out of F1 scope by
agreement; nothing here is a reason to delay F1.

---

## What the measurement says

Zakon o radu, 109 pages, taken as a fair worst case — it is large and heavily
amended. Measured directly on the PDF:

```
članci                      272
chars       min 42   p50 791   p90 2325   p95 2718   max 18142
tokens      p50 200  p75 382   p90 587    p95 686    p99 2535   max 4581
over the 508-token embedding window        42 / 272  = 15.4%
stavak markers present                     222 / 272 = 82%   (42/42 of the oversized)
split ONLY the oversized ones at stavak → 607 units, largest remaining 3249 tokens
```

Four things follow.

**1. Whole-članak chunking is affordable.** 85% of članci fit the embedding
window untouched. This is not a case where the natural unit is hopelessly
larger than the model — it mostly already fits.

**2. The tail needs a split, and stavak is the right seam.** Every oversized
članak has stavak markers, so the split point always exists. It is also the
seam a lawyer would choose, which matters when the excerpt is shown to an
advisor.

**3. Stavak alone is not enough.** One 3,249-token fragment survives the split.
There must be a fallback to the sentence packer we already have, or a chunk
goes into the embedder and comes out silently truncated — the defect that cost
48.7% of v1's tokens.

**4. Article numbers are not unique inside one law file.** The PDF is the
consolidated text *plus* the transitional provisions of each amending act, and
those restart their numbering. Ten article numbers appear twice. So `čl. 11` is
ambiguous **within a single document**, and any schema that keys on the article
label alone will silently merge two unrelated provisions.

Heading shapes, for whoever writes the loader — all three occur, and the letter
suffix puts the dot **before** the letter:

```
Članak 1.
Članak 2. (NN 93/14, 151/22)
Članak 221.a (NN 151/22)
```

My first heading regex matched 183 of 272 and glued 89 članci onto their
predecessors without complaining. Count the matches against a hand count before
trusting a loader.

---

## The schema: two tables, three logical levels

The three levels are *document → unit → chunk*. Only two of them need a table.

### `documents` — one row per source publication

```sql
CREATE TABLE documents (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_type        text NOT NULL,          -- 'clanak' | 'zakon'
    title           text NOT NULL,
    -- magazine
    publication     text,                   -- 'RRiF' | 'PiP'
    issue           int,
    year            int,
    author          text,
    -- statute
    law_name        text,
    nn_reference    text,                   -- 'NN 93/14, 151/22, …'
    -- both
    valid_from      date,
    valid_to        date,                   -- NULL = in force
    status          text NOT NULL DEFAULT 'vazeci',
    source_file     text,                   -- the PDF this came from
    source_sha256   text,                   -- so a re-ingest is detectable
    ingested_at     timestamptz NOT NULL DEFAULT now()
);
```

`status` and `valid_to` finally mean something here. In v1 they were stamped on
every magazine chunk and encoded a false claim — that a 2018 article on how to
book a leasing contract stops being true on 31 December 2018. Validity is a
property of a *legal text*, which is a property of a *document*, which is why
it belongs on this table and not on 33,000 chunk rows.

### `chunks` — the embedded units, as now, plus provenance

```sql
ALTER TABLE chunks
    ADD COLUMN document_id uuid REFERENCES documents(id),
    ADD COLUMN unit_ref    text,   -- display label: '221.a', 'Prijelazne odredbe'
    ADD COLUMN unit_seq    int,    -- position of the unit within the document
    ADD COLUMN chunk_index int;    -- position of this chunk within the unit

CREATE INDEX ON chunks (document_id, unit_seq, chunk_index);
```

**`unit_seq` is the key; `unit_ref` is the label.** This is the whole answer to
the duplicate-article-number problem. Two članci both labelled `11.` — one in
the consolidated text, one in a transitional block — get different `unit_seq`,
so they never merge, and both still display as "članak 11.".

A whole članak is then reconstructed with an ordinary query:

```sql
SELECT chunk_text
FROM   chunks
WHERE  document_id = %s AND unit_seq = %s
ORDER  BY chunk_index;
```

### Why no third table

A `units` table would hold an id, a parent, a label, an ordinal and nothing
else — every real field is either on the document or on the chunks. It buys one
join's worth of tidiness and costs a join on every retrieval path, plus a
second place for `unit_ref` to drift out of sync. The two-table version answers
every question we actually have. If a unit later needs its own attributes
(marginal notes, per-article amendment history, cross-references), add the
table then; `unit_seq` is already the foreign key it would need.

---

## Chunking policy, by source type

| | `clanak` (magazine) | `zakon` (statute) |
|---|---|---|
| unit | none — the article is chunked directly | one članak |
| target | ~400 tokens, 60-token overlap | whole članak |
| when oversized | sentence packer (as now) | split at stavak `(1) (2) (3)` |
| when *still* oversized | — | sentence packer, with overlap |
| hard cap | 508 tokens | 508 tokens |

The hard cap is not negotiable in either column. `sentence-transformers`
truncates past e5-large's 512-token window **silently** — no warning, no error,
no log line. That is how half the v1 corpus became invisible to semantic
search. Any new loader prints a token histogram in its dry run and refuses to
proceed if anything exceeds the cap.

Magazine chunking does not change. The union metric — whether the answer needs
more than one chunk — was measured three separate times on this golden set and
found nothing split across chunks. There is no evidence to act on.

---

## Retrieval policy

**Expand statute hits, don't expand magazine hits.** When a `zakon` chunk
survives reranking, fetch its siblings and hand the generator the whole članak.
When a magazine chunk survives, hand over the chunk. The asymmetry is not
elegance, it is what the measurements support: for laws the surrounding stavak
changes the meaning; for magazine text, three measurements say it does not.

**One HNSW index over everything**, until something forces otherwise. A second
index is a second thing to keep in sync and a second recall profile to reason
about, in exchange for a speed-up nobody has asked for.

**The real F2 problem is dilution, not chunking.** In v2, 188 statute chunks
sit against 33,076 magazine chunks. Article 38 of the VAT law is present,
correctly flagged, and loses on ratio — ZAKONI top-K went from 66.7% on v1 to
33.3% on v2 for exactly this reason, while every article metric improved. A
better statute chunker will not fix that. Two things to measure when F2 starts:

- a **reserved quota** in the candidate pool — when the classifier's domain is
  legal, guarantee N slots to `doc_type = 'zakon'` before fusion;
- **separate retrieval then fuse** — run the two corpora as two queries and
  merge with RRF, which is already the fusion we use.

Measure both against the same golden set before choosing. And note the golden
set has six law questions, so one question is 16.7 points — that comparison
needs more law questions before it means anything.

---

## What to do now, in F1

**One thing: start writing `document_id` at ingest.**

Add the `documents` table and the `document_id` column now, populate them
during the ingest we already run, and leave every other column of the design
unbuilt. The reason is narrow and specific: re-embedding 33,000 chunks costs
about an hour and a half of GPU time, and retrofitting provenance later means
re-deriving it from `source` strings by regex — which is precisely the kind of
"parse it back out of a display string" that produced the article-number mess
we are still cleaning up. Writing an id at ingest costs nothing and removes
that whole class of problem.

Concretely:

1. `CREATE TABLE documents (…)` as above.
2. `ALTER TABLE chunks ADD COLUMN document_id uuid REFERENCES documents(id);`
3. In `rag/ingest/pipeline.py`, insert one `documents` row per PDF before the
   chunks and carry the id onto every chunk from that file.
4. Leave `unit_ref`, `unit_seq`, `chunk_index` out for now — F1 has no units.

**Do not do now:** the article-aware law loader, the expansion path, the
dilution experiments, `unit_seq`, or a second index. All of it is F2 work, none
of it is on the critical path, and every hour spent on it is an hour not spent
on the two things F1 is actually short of — an end-to-end accuracy number and a
validation set worth the name.

---

## Before committing to any of this

Zakon o radu is one law and probably a worst case. **ZPDV is already in the
corpus and should be measured the same way** — članak count, token
distribution, share over 508, stavak coverage, duplicate numbering — before the
loader is written against a sample of one. If ZPDV looks materially different,
the policy table above is a hypothesis, not a design.
