"""End-to-end query pipeline for the RRiF AI Agent.

  1. Classify question (Haiku)         → domain, subdomains, time_period
  2. Hybrid retrieval, ONE CTE query   → top-N candidates (pgvector + FTS, RRF in SQL)
  3. Cross-encoder reranking (BGE)     → top-K chunks
  4. Answer generation (Sonnet)        → answer + citations + temporal basis
  5. Return structured QueryResponse


THREE FIXES IN THIS REVISION
────────────────────────────

1. ONE RERANK CALL PER QUESTION (was up to four)

   The old `_retrieve` ran a three-level fallback and called
   `_top_rerank_score` at every level before deciding whether to widen, then
   reranked again at the end — up to four sequential invocations, ~35 scored
   pairs. Locally that is one batched pass and nearly free; over an API it is
   four network round-trips, which does not fit the 200 ms retrieval+rerank
   budget in the VOICE sheet or the 2 s first-token criterion in F4.

   Now: candidates are gathered per level, and each level reranks only the
   chunks it newly contributed, with scores cached. The common case (tight
   filter is good enough) is exactly one call over ~20 pairs. The worst case
   is three calls but still ~20 pairs total, because nothing is scored twice.

2. TEMPORAL FILTER NOW APPLIES TO MAGAZINE CHUNKS TOO

   The old WHERE clause read:

       (source_type != 'članak' AND ({sql_time}) OR source_type = 'članak')

   Operator precedence makes that `(not članak AND time_ok) OR is članak`, so
   every magazine chunk bypassed the time filter while statute had to satisfy
   it. With 5,174 of 12,561 magazine chunks marked `nevazeci`, 41% of the
   corpus competed on current-state questions with no temporal constraint —
   which is why "Kolika je opća stopa PDV-a?" returned a 2014 article about
   the old 13% rate, scoring 0.99.

   The exemption existed because every magazine chunk has `valid_to` set
   (none NULL), so a `valid_to IS NULL` test would delete the whole corpus.
   But `sql_time` for a current question is `status = 'vazeci'`, and 7,387
   magazine chunks satisfy that. The exemption was never needed.

   TEMPORAL_MODE=strict (default) applies the filter to everything.
   TEMPORAL_MODE=legacy restores the exemption, for A/B comparison.

3. CONNECTION POOLING AND A SINGLE CTE

   Was: a fresh `psycopg.connect()` per call, two queries per retrieval
   attempt, up to three attempts, plus one more connection per HTTP request
   in the API layer — up to seven new TLS connections per question, and an
   N+1 pattern the tech spec explicitly forbids.

   Now: one process-wide `ConnectionPool` (pgvector registered once per
   connection), and semantic + FTS + RRF fusion in a single round trip.

   Requires `psycopg_pool` — add to requirements.txt.


EXPERIMENT HOOKS (all default to production behaviour)

  RETRIEVAL_MODE=tight|domain|wide   skip straight to a filter level
  TEMPORAL_MODE=strict|legacy        see fix 2
  ask(..., skip_generation=True)     stop after reranking, no Sonnet call
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from rag.classifier import ClassifierResult, classify
from rag.embedder import embed_query
from rag.retrieve.rerank import rerank
from rag.generate.answerer import answer as _generate_answer
from rag.rewrite.rewriter import rewrite

load_dotenv()

log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

GENERATOR_MODEL         = "claude-sonnet-4-6"
POOL_PER_RANKER         = 50   # rows each CTE branch contributes before fusion
CANDIDATES              = 20   # candidates surviving fusion, sent to the reranker
TOP_K                   = 5    # chunks handed to the generator
RRF_K                   = 60
MIN_FALLBACK_CANDIDATES = 5
RERANK_THRESHOLD        = 0.75

RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "tight").lower()
if RETRIEVAL_MODE not in {"tight", "domain", "wide"}:
    raise ValueError(f"RETRIEVAL_MODE must be tight|domain|wide, got {RETRIEVAL_MODE!r}")

TEMPORAL_MODE = os.getenv("TEMPORAL_MODE", "strict").lower()
if TEMPORAL_MODE not in {"strict", "legacy"}:
    raise ValueError(f"TEMPORAL_MODE must be strict|legacy, got {TEMPORAL_MODE!r}")


# ── Connection pool ───────────────────────────────────────────────────────────

_POOL: ConnectionPool | None = None


def _pool() -> ConnectionPool:
    """One pool for the process, with pgvector registered per connection.

    register_vector() rewrites type adapters on the connection, which is why
    the old code opened a separate plain connection for FTS. Doing it once in
    the pool's configure hook removes that constraint entirely.
    """
    global _POOL
    if _POOL is None:
        _POOL = ConnectionPool(
            conninfo=os.environ["DATABASE_URL"],
            min_size=int(os.getenv("PG_POOL_MIN", "1")),
            max_size=int(os.getenv("PG_POOL_MAX", "10")),
            configure=register_vector,
            open=True,
        )
    return _POOL


def close_pool() -> None:
    """For clean shutdown in the API layer (FastAPI lifespan)."""
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL = None


# ── Response data classes ─────────────────────────────────────────────────────

@dataclass
class Citation:
    source: str
    article_number: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    excerpt: str = ""
    author: str | None = None


@dataclass
class QueryResponse:
    answer: str
    citations: list[Citation]
    temporal_basis: str
    confidence: Literal["high", "low"]
    referred_to_advisor: bool
    classifier: ClassifierResult = field(repr=False)
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    original_query: str = ""
    rewritten_query: str = ""
    rewrite_changed: bool = False
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_scores: list[float] = field(default_factory=list)
    # What retrieval returned, before the generator had any say. Grade
    # retrieval on this, not on citations.
    retrieved_meta: list[dict] = field(default_factory=list)
    generation_skipped: bool = False
    n_rerank_calls: int = 0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": [
                {
                    "source": c.source,
                    "article_number": c.article_number,
                    "valid_from": c.valid_from,
                    "valid_to": c.valid_to,
                    "excerpt": c.excerpt,
                    "author": c.author,
                }
                for c in self.citations
            ],
            "temporal_basis": self.temporal_basis,
            "confidence": self.confidence,
            "referred_to_advisor": self.referred_to_advisor,
            "meta": {
                "domain": self.classifier.domain,
                "subdomains": self.classifier.subdomains,
                "time_period": self.classifier.time_period.type,
                "latency_ms": self.latency_ms,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "original_query": self.original_query,
                "rewritten_query": self.rewritten_query,
                "rewrite_changed": self.rewrite_changed,
                "retrieval_mode": RETRIEVAL_MODE,
                "temporal_mode": TEMPORAL_MODE,
                "n_rerank_calls": self.n_rerank_calls,
                "generation_skipped": self.generation_skipped,
            },
        }


# ── Retrieval ─────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _build_tsquery(question: str) -> str | None:
    """OR-joined tokens, sanitised.

    The old version did `" | ".join(question.split())`, which hands raw
    punctuation to to_tsquery — 'PDV-a' is a syntax error, and the whole FTS
    branch was wrapped in try/except to swallow it. Extracting \\w+ runs makes
    the query always valid, so a silent FTS failure can no longer look like
    "the keyword search just didn't match anything".

    OR rather than AND is deliberate: recall first, then let RRF and the
    reranker sort it out.
    """
    words = [w for w in _WORD_RE.findall(question) if len(w) >= 3]
    return " | ".join(words) if words else None


def _build_where(
    domain: str | None,
    subdomains: list[str] | None,
    sql_time: str | None,
    params: dict,
) -> str:
    """WHERE fragment using NAMED placeholders, so it can appear in both CTE
    branches without duplicating positional parameters."""
    clauses = ["citable = TRUE"]

    if domain and subdomains:
        clauses.append("domain = %(domain)s AND subdomain = ANY(%(subdomains)s)")
        params["domain"] = domain
        params["subdomains"] = list(subdomains)
    elif domain:
        clauses.append("domain = %(domain)s")
        params["domain"] = domain

    if sql_time:
        if TEMPORAL_MODE == "strict":
            # Applies to every source type. See fix 2 in the module docstring.
            clauses.append(f"({sql_time})")
        else:
            clauses.append(
                f"(source_type <> 'članak' AND ({sql_time}) OR source_type = 'članak')"
            )

    return " AND ".join(clauses)


_SELECT_COLS = """c.id, c.chunk_text, c.source, c.article_number,
                  c.valid_from::text, c.valid_to::text,
                  c.source_type, c.extra_metadata, c.status"""


def _hybrid_search(
    question: str,
    qvec,
    domain: str | None,
    subdomains: list[str] | None,
    sql_time: str | None,
) -> list[tuple]:
    """Semantic + FTS + RRF fusion in a single round trip.

    Each branch is limited BEFORE numbering, so the vector branch still uses
    the HNSW index and the FTS branch still uses the GIN index; row_number()
    then runs over 50 rows rather than the whole table.
    """
    params: dict = {}
    where = _build_where(domain, subdomains, sql_time, params)
    tsq = _build_tsquery(question)

    params.update({
        "qvec": qvec,
        "pool": POOL_PER_RANKER,
        "rrf_k": RRF_K,
        "candidates": CANDIDATES,
    })

    vec_cte = f"""
        vec AS (
            SELECT id, row_number() OVER (ORDER BY dist) AS rnk
            FROM (
                SELECT id, embedding <=> %(qvec)s AS dist
                FROM chunks
                WHERE {where} AND embedding IS NOT NULL
                ORDER BY embedding <=> %(qvec)s
                LIMIT %(pool)s
            ) t
        )"""

    if tsq:
        params["tsq"] = tsq
        fts_cte = f""",
        fts AS (
            SELECT id, row_number() OVER (ORDER BY score DESC) AS rnk
            FROM (
                SELECT id,
                       ts_rank_cd(to_tsvector('simple', chunk_text),
                                  to_tsquery('simple', %(tsq)s)) AS score
                FROM chunks
                WHERE {where}
                  AND to_tsvector('simple', chunk_text)
                      @@ to_tsquery('simple', %(tsq)s)
                ORDER BY score DESC
                LIMIT %(pool)s
            ) t
        )"""
        fused = """
        fused AS (
            SELECT COALESCE(v.id, f.id) AS id,
                   COALESCE(1.0 / (%(rrf_k)s + v.rnk), 0.0)
                 + COALESCE(1.0 / (%(rrf_k)s + f.rnk), 0.0) AS rrf
            FROM vec v FULL OUTER JOIN fts f ON f.id = v.id
        )"""
    else:
        # No usable keyword tokens — vector branch only.
        fts_cte = ""
        fused = """
        fused AS (
            SELECT id, 1.0 / (%(rrf_k)s + rnk) AS rrf FROM vec
        )"""

    sql = f"""
        WITH {vec_cte}{fts_cte},{fused}
        SELECT {_SELECT_COLS}
        FROM fused fu JOIN chunks c ON c.id = fu.id
        ORDER BY fu.rrf DESC
        LIMIT %(candidates)s
    """

    with _pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def _levels(f: dict) -> list[tuple[str | None, list[str] | None]]:
    """Filter levels to try, in order, per RETRIEVAL_MODE."""
    if RETRIEVAL_MODE == "wide":
        return [(None, None)]
    if RETRIEVAL_MODE == "domain":
        return [(f["domain"], None), (None, None)]
    return [(f["domain"], f["subdomains"]), (f["domain"], None), (None, None)]


def _retrieve_and_rank(question: str, clf: ClassifierResult) -> tuple[list[tuple], dict, int]:
    """Gather candidates, widening only when needed, and rerank incrementally.

    Returns (top_chunks, scores_by_chunk_id, n_rerank_calls).

    Each level reranks only what it newly contributed; scores are cached, so
    no (query, passage) pair is ever scored twice. One call in the common case.
    """
    qvec = embed_query(question)
    f = clf.to_retrieval_filter()
    sql_time = f["sql_time_filter"]

    pool: dict = {}            # chunk_id -> row
    scores: dict[str, float] = {}
    n_calls = 0
    levels = _levels(f)

    for i, (domain, subdomains) in enumerate(levels):
        rows = _hybrid_search(question, qvec, domain, subdomains, sql_time)
        fresh = [r for r in rows if r[0] not in pool]
        for r in rows:
            pool.setdefault(r[0], r)

        if fresh:
            scored = rerank(question, [(r[0], r[1]) for r in fresh], k=len(fresh))
            n_calls += 1
            for cid, _, s in scored:
                scores[str(cid)] = float(s)

        if not pool:
            continue

        ordered = sorted(
            pool.values(),
            key=lambda r: scores.get(str(r[0]), 0.0),
            reverse=True,
        )
        best = scores.get(str(ordered[0][0]), 0.0)

        enough = len(pool) >= MIN_FALLBACK_CANDIDATES
        good = best >= RERANK_THRESHOLD
        if (enough and good) or i == len(levels) - 1:
            log.debug(
                "_retrieve level=%d pool=%d best=%.3f rerank_calls=%d",
                i, len(pool), best, n_calls,
            )
            return ordered[:TOP_K], scores, n_calls

        log.info(
            "_retrieve widening after level %d (pool=%d, best=%.3f)",
            i, len(pool), best,
        )

    return [], scores, n_calls


# ── Generation ────────────────────────────────────────────────────────────────

def _generate(question: str, top_chunks: list[tuple], clf: ClassifierResult) -> QueryResponse:
    import time
    from datetime import date as _date

    chunks_for_answerer: list[dict] = []
    for row in top_chunks:
        em = row[7] if len(row) > 7 and row[7] else {}
        chunks_for_answerer.append({
            "chunk_id":       row[0],
            "chunk_text":     row[1],
            "source":         row[2],
            "law_name":       None,
            "nn_reference":   None,
            "article_number": row[3],
            "valid_from":     row[4],
            "valid_to":       row[5],
            "status":         row[8] if len(row) > 8 else "nevazeci",
            "source_type":    row[6],
            "pub_label":      em.get("pub_label", ""),
            "title":          em.get("title", ""),
            "author":         em.get("author", ""),
            "article_num":    em.get("article_num", ""),
        })

    t0 = time.perf_counter()
    gen = _generate_answer(question, chunks_for_answerer, model=GENERATOR_MODEL)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    citations = [
        Citation(
            # law_name is hardcoded None above, so without the fallback every
            # citation came back with an empty source.
            source=c.get("law_name") or c.get("source") or "",
            article_number=c.get("article_number"),
            valid_from=None,
            valid_to=None,
            excerpt="",
            author=c.get("author"),
        )
        for c in gen.citations
    ]

    today = _date.today().isoformat()
    return QueryResponse(
        answer=gen.answer,
        citations=citations,
        temporal_basis=gen.temporal_note or f"Odgovor generiran {today}.",
        confidence="low" if gen.refused else "high",
        referred_to_advisor=gen.refused,
        classifier=clf,
        latency_ms=latency_ms,
        tokens_in=gen.input_tokens,
        tokens_out=gen.output_tokens,
    )


def _meta(top_chunks: list[tuple], scores: dict) -> list[dict]:
    return [
        {
            "chunk_id":       str(row[0]),
            "source":         row[2],
            "article_number": row[3],
            "source_type":    row[6],
            "status":         row[8] if len(row) > 8 else None,
            "rerank_score":   scores.get(str(row[0])),
            # Carried so the eval can check whether a retrieved chunk actually
            # contains the expected answer. Publication-issue matching turned
            # out to be the wrong instrument: recurring figures like the 2024
            # minimum wage appear across eleven issues of the same year, so
            # "which issue" has no single right answer.
            "chunk_text":     row[1],
        }
        for row in top_chunks
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def ask(
    question: str,
    verbose: bool = False,
    enable_rewrite: bool = False,
    skip_generation: bool = False,
) -> QueryResponse:
    """Full RAG pipeline: classify → retrieve → rerank → generate.

    skip_generation: stop after reranking. `retrieved_meta` is still populated,
                     which is what the eval harness grades retrieval on.
    """
    import time
    t_start = time.perf_counter()

    original_query = question
    rewritten_query = question
    rewrite_changed = False
    if enable_rewrite:
        rw = rewrite(question)
        rewritten_query = rw.rewritten
        rewrite_changed = rw.changed
        question = rw.rewritten
        if verbose:
            marker = "(changed)" if rw.changed else "(unchanged)"
            print(f"[rewriter] {marker} → {rw.rewritten!r}")

    clf = classify(question)
    if verbose:
        print(f"[classifier] domain={clf.domain} subdomains={clf.subdomains} | "
              f"time={clf.time_period.type} | recency={clf.recency_boost}")

    top_chunks, scores, n_calls = _retrieve_and_rank(question, clf)
    if verbose:
        print(f"[retrieval] {len(top_chunks)} chunks "
              f"(mode={RETRIEVAL_MODE}, temporal={TEMPORAL_MODE}, "
              f"rerank_calls={n_calls})")

    if not top_chunks:
        from datetime import date as _date
        return QueryResponse(
            answer=(
                "Nažalost, nisam pronašao relevantne informacije u bazi znanja. "
                "Molim Vas obratite se RRiF savjetničkoj liniji za točan odgovor."
            ),
            citations=[],
            temporal_basis=f"Pretraživanje provedeno {_date.today().isoformat()}.",
            confidence="low",
            referred_to_advisor=True,
            classifier=clf,
            latency_ms=int((time.perf_counter() - t_start) * 1000),
            original_query=original_query,
            rewritten_query=rewritten_query,
            rewrite_changed=rewrite_changed,
            generation_skipped=skip_generation,
            n_rerank_calls=n_calls,
        )

    if skip_generation:
        result = QueryResponse(
            answer="",
            citations=[],
            temporal_basis="",
            confidence="high",
            referred_to_advisor=False,
            classifier=clf,
            generation_skipped=True,
        )
    else:
        result = _generate(question, top_chunks, clf)

    result.latency_ms = int((time.perf_counter() - t_start) * 1000)
    result.original_query = original_query
    result.rewritten_query = rewritten_query
    result.rewrite_changed = rewrite_changed
    result.retrieved_chunk_ids = [str(r[0]) for r in top_chunks]
    result.retrieved_scores = [scores.get(str(r[0]), 0.0) for r in top_chunks]
    result.retrieved_meta = _meta(top_chunks, scores)
    result.n_rerank_calls = n_calls

    if verbose and not skip_generation:
        print(f"[generator] confidence={result.confidence} | "
              f"citations={len(result.citations)} | "
              f"latency={result.latency_ms}ms | "
              f"tokens={result.tokens_in}in/{result.tokens_out}out")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    # WARNING at the root keeps httpx/huggingface chatter out of the way;
    # our own retrieval messages still come through.
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("rag").setLevel(logging.INFO)

    args = sys.argv[1:]
    enable_rewrite = "--rewrite" in args
    skip_gen = "--no-generation" in args
    args = [a for a in args if not a.startswith("--")]

    question = " ".join(args) if args else None
    if not question:
        print("Usage: python -m rag.query [--rewrite] [--no-generation] <question>")
        sys.exit(1)

    print(f"\nPitanje: {question}\n{'='*60}")
    result = ask(question, verbose=True, enable_rewrite=enable_rewrite,
                 skip_generation=skip_gen)
    print(f"\n{'='*60}")
    if skip_gen:
        print(json.dumps(result.retrieved_meta, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    close_pool()
