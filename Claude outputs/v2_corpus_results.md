# v2 corpus: results, and what the day actually established

*Duje · 7 Sep 2026 · companion to `claude/v2_corpus_runbook.md`*

## The number

**F1 (članci) retrieval: 93.5% — v2 corpus, TOP_K=10, laws_only temporal mode.**

Measured on the 31 magazine questions of the 41-question golden set, graded on
whether the retrieved chunks contain the expected key terms, with Croatian
morphology normalised on both sides.

| | v1 @ K=5 | v2 @ K=10 |
|---|---|---|
| ČLANCI odgovor u uniji | 90.3% | **93.5%** |
| ČLANCI source top-K | 84.6% | **92.3%** |
| ZAKONI article top-K | 66.7% | 33.3% |
| chunks needed for answer | 1.09 | 1.54 |
| avg latency | 3,991 ms | **1,949 ms** |
| corpus size | 12,749 | 33,264 |

v2 wins on every article metric at comparable context volume (v1: 5 × ~649
tokens; v2: 10 × ~349) and halves latency. It loses two law questions.

**Recommendation: adopt v2 with `TOP_K=10`.** `.env` is unchanged and still
points at v1, so this is a deliberate one-line switch, not a fait accompli.

---

## Why the system barely changed and the numbers moved a lot

The retrieval pipeline is nearly what it was this morning. What changed is the
measuring instrument, four times, and the corpus underneath it.

| reported | what it actually measured |
|---|---|
| 8.1% | citations, via a denominator that made 16.2% the ceiling |
| 67.6% | retrieval, exact substring, single chunk |
| 78.4% | + Croatian morphology on both sides of the comparison |
| 83.9% | + ČLANCI/ZAKONI split, so F1 isn't dragged by out-of-scope questions |
| 93.5% | + v2 corpus at matched context volume |

Only the last two lines involved changing the system. The rest was the ruler.

---

## Findings that outlive today

**Half the knowledge base was invisible to semantic search.** e5-large has a
512-token window and `sentence-transformers` truncates past it silently. The v1
median chunk was 649 tokens; 55.6% of magazine chunks exceeded the window and
48.7% of all corpus tokens never reached the embedder. v2: 0.1%. This was the
single largest defect found, and nothing in the code or logs pointed at it —
the check had to be written.

**The chunker's size cap was unenforceable.** `MAX_CHARS_PER_CHUNK = 1_500`
split only on blank lines, while the extractor joined blocks with single
newlines. Pages taking the table-aware path had no split point at all, so a
1,500-character cap produced a 9,728-character maximum.

**96.6% of citations were paragraphs of body text.** `_extract_title_and_author`
joined the first four surviving lines, which on a two-column page is prose. That
field is what an advisor sees under an answer. Now 8%.

**Titles are found by type size, not position.** Three templates in this archive
disagree about where the title sits: 2024 puts it after the byline with every
line drawn twice; 2014 puts it before, undrawn-twice, sometimes beneath a
full-page advert. Largest type on page one works for all three and will
probably work for the fourth.

**`status` is a legal property and articles were being stamped with it.**
`HISTORICAL_CUTOFF = 2019` marked everything older as `nevazeci`, hiding 5,174
chunks of still-correct method. Worse, `valid_to` was set to 31 December of the
publication year — so a question about 2025 excluded the December 2024 issue,
which is exactly where RRiF publishes next year's changes. `pdv_008` returned
nothing at all under that filter and five correct chunks at 0.999 without it.

**The classifier ran at temperature 1.** Two eval runs of identical code
classified the same question differently and moved a metric by 7.7 points.
Pinned to 0.

---

## Three of my own hypotheses that measurement killed

**The temporal filter.** I read the magazine exemption as an operator-precedence
bug and made the filter apply to everything. Measured, that was worse on every
metric — the exemption was correct and the bug was in the data model.

**Chunk boundaries splitting answers.** Assumed for hours. `diag_split` found
zero adjacent cases among the seven suspects, and the union metric later added
nothing over the single-chunk one in either corpus. No answer in this golden set
is split across chunks. The real cause was declension in the grader.

**Croatian stemming.** Built, indexed, wired through the query path, measured:
no effect. Five questions move, net zero. The stemmer turned out to belong in
the grader, not the index.

---

## Still open

- **Law retrieval.** 188 statute chunks against 33,076 magazine chunks. Article
  38 of the VAT law is present, correctly flagged, and loses on ratio.
  Legitimate F2 work, and the open scope question in the TehSpec.
- **Recency as ranking rather than filtering.** `recency_boost` is emitted by
  the classifier and read by nothing. The corpus is 2014 and 2024 with a
  ten-year hole, which flatters a crude year cutoff; a full archive will not.
- **The summary labels say "top-5" when TOP_K is 10.** Cosmetic, but it will
  mislead someone quoting the table.
- **`article_number` NULL for 36.9% of chunks** and 2.0% duplicate text — both
  worse in v2 than v1, neither yet understood.
- **The golden set is 41 questions.** Every result above rests on one or two
  questions moving. RRiF's validation set is the fix, and 150–200 questions
  rather than 100 is the ask.
