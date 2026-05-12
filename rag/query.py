"""End-to-end query pipeline for the RRiF AI Agent.

Orchestrates the full RAG flow:
  1. Classify question (Haiku)         → domain, subdomains, time_period
  2. Hybrid retrieval with filters     → top 20 candidates (semantic + FTS, RRF fused)
  3. Cross-encoder reranking (BGE)     → top 5 most relevant chunks
  4. Answer generation (Sonnet)        → answer + citations + temporal basis
  5. Return structured QueryResponse

Retrieval uses a three-level fallback:
  tight  → domain + subdomain(s)
  medium → domain only
  wide   → no filter (last resort)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from rag.classifier import ClassifierResult, classify
from rag.embedder import embed_query
from rag.retrieve.rerank import rerank
from rag.generate.answerer import answer as _generate_answer
from rag.rewrite.rewriter import rewrite

load_dotenv()

log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

GENERATOR_MODEL         = "claude-sonnet-4-6"
CANDIDATES              = 20
TOP_K                   = 5
RRF_K                   = 60
MIN_FALLBACK_CANDIDATES = 5   # widen filter if fewer candidates than this


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
            },
        }

# ── Retrieval helpers ─────────────────────────────────────────────────────────

def _build_where(
    domain: str | None,
    subdomains: list[str] | None,
    sql_time: str | None,
    params: list,
) -> str:
    clauses = ["citable = TRUE"]

    if domain and subdomains:
        clauses.append("domain = %s AND subdomain = ANY(%s)")
        params.extend([domain, subdomains])
    elif domain:
        clauses.append("domain = %s")
        params.append(domain)

    if sql_time:
        clauses.append(f"({sql_time})")

    return " AND ".join(clauses)


def _semantic_fts_search(
    question: str,
    qvec,
    where_sql: str,
    params: list,
    n: int,
) -> tuple[dict, dict]:
    """Run semantic + FTS search. Returns (sem_rows, fts_rows) dicts keyed by chunk id."""
    fts_rows: dict = {}
    words = [w for w in question.split() if len(w) >= 3]
    if words:
        tsquery = " | ".join(words)
        try:
            with psycopg.connect(os.environ["DATABASE_URL"]) as conn_fts:
                with conn_fts.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT id, chunk_text, source, article_number,
                            valid_from::text, valid_to::text,
                            source_type, extra_metadata
                        FROM chunks
                        WHERE {where_sql}
                          AND to_tsvector('simple', chunk_text)
                              @@ to_tsquery('simple', %s)
                        ORDER BY ts_rank_cd(
                            to_tsvector('simple', chunk_text),
                            to_tsquery('simple', %s)
                        ) DESC
                        LIMIT %s
                        """,
                        params + [tsquery, tsquery, n],
                    )
                    fts_rows = {row[0]: row for row in cur.fetchall()}
        except Exception as e:
            log.warning("FTS search failed: %s", e)

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn_vec:
        register_vector(conn_vec)
        with conn_vec.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, chunk_text, source, article_number,
                       valid_from::text, valid_to::text,
                       source_type, extra_metadata
                FROM chunks
                WHERE {where_sql}
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                params + [qvec, n],
            )
            sem_rows = {row[0]: row for row in cur.fetchall()}

    return sem_rows, fts_rows


def _rrf_merge(sem_rows: dict, fts_rows: dict) -> list[tuple]:
    """Reciprocal Rank Fusion. Returns rows sorted by RRF score."""
    sem_ranks = {rid: i + 1 for i, rid in enumerate(sem_rows)}
    fts_ranks = {rid: i + 1 for i, rid in enumerate(fts_rows)}
    all_ids = set(sem_ranks) | set(fts_ranks)

    scored = []
    for rid in all_ids:
        score = 0.0
        if rid in sem_ranks:
            score += 1.0 / (RRF_K + sem_ranks[rid])
        if rid in fts_ranks:
            score += 1.0 / (RRF_K + fts_ranks[rid])
        row = sem_rows.get(rid) or fts_rows.get(rid)
        scored.append((
            score,
            row[0], row[1], row[2], row[3], row[4], row[5],
            row[6] if len(row) > 6 else None,
            row[7] if len(row) > 7 else None,
        ))

    scored.sort(reverse=True)
    return [(r[1], r[2], r[3], r[4], r[5], r[6],r[7], r[8]) for r in scored[:CANDIDATES]]


def _retrieve(question: str, clf: ClassifierResult) -> list[tuple]:
    """Hybrid retrieval with three-level fallback.

    1. tight  — domain + subdomains
    2. medium — domain only        (if tight < MIN_FALLBACK_CANDIDATES)
    3. wide   — no filter          (if medium < MIN_FALLBACK_CANDIDATES)
    """
    qvec = embed_query(question)
    filters = clf.to_retrieval_filter()
    sql_time = filters["sql_time_filter"]

    # Attempt 1: tight
    params: list = []
    where_sql = _build_where(filters["domain"], filters["subdomains"], sql_time, params)
    sem_rows, fts_rows = _semantic_fts_search(question, qvec, where_sql, params, CANDIDATES)
    candidates = _rrf_merge(sem_rows, fts_rows)

    if len(candidates) >= MIN_FALLBACK_CANDIDATES:
        log.debug("_retrieve tight: %d (domain=%s subdomains=%s)",
                  len(candidates), filters["domain"], filters["subdomains"])
        return candidates

    # Attempt 2: domain-only
    log.info("_retrieve → domain-only fallback (tight=%d, domain=%s)",
             len(candidates), filters["domain"])
    fb = clf.fallback_filter()
    params2: list = []
    where_sql2 = _build_where(fb["domain"], None, sql_time, params2)
    sem2, fts2 = _semantic_fts_search(question, qvec, where_sql2, params2, CANDIDATES)
    candidates2 = _rrf_merge(sem2, fts2)

    if len(candidates2) >= MIN_FALLBACK_CANDIDATES:
        return candidates2

    # Attempt 3: unfiltered
    log.info("_retrieve → unfiltered fallback (domain-level=%d)", len(candidates2))
    params3: list = []
    where_sql3 = _build_where(None, None, sql_time, params3)
    sem3, fts3 = _semantic_fts_search(question, qvec, where_sql3, params3, CANDIDATES)
    return _rrf_merge(sem3, fts3)


# ── Generation ────────────────────────────────────────────────────────────────

def _generate(question: str, top_chunks: list[tuple], clf: ClassifierResult) -> QueryResponse:
    import time
    from datetime import date as _date

    chunks_for_answerer: list[dict] = []
    for row in top_chunks:
        em = row[7] if len(row) > 7 and row[7] else {}
        chunks_for_answerer.append({
            "chunk_id":       row[0] if len(row) > 0 else None,
            "chunk_text":     row[1] if len(row) > 1 else "",
            "source":         row[2] if len(row) > 2 else "",
            "law_name":       None,
            "nn_reference":   None,
            "article_number": row[3] if len(row) > 3 else None,
            "valid_from":     row[4] if len(row) > 4 else None,
            "valid_to":       row[5] if len(row) > 5 else None,
            "status":         "vazeci" if not (row[5] if len(row) > 5 else None) else "nevazeci",
            "source_type":    row[6] if len(row) > 6 else None,
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
            source=c.get("law_name") or "",
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


# ── Public API ────────────────────────────────────────────────────────────────

def ask(
    question: str,
    verbose: bool = False,
    enable_rewrite: bool = False,
) -> QueryResponse:
    """Full RAG pipeline: classify → retrieve → rerank → generate.

    enable_rewrite: if True, run the Haiku query rewriter before classification
                    to clean up colloquial/abbreviated input. Default off for
                    backward compatibility.
    """
    import time
    t_start = time.perf_counter()

    # Step 0: Rewrite (optional)
    original_query = question
    rewritten_query = question
    rewrite_changed = False
    if enable_rewrite:
        rw = rewrite(question)
        rewritten_query = rw.rewritten
        rewrite_changed = rw.changed
        question = rw.rewritten  # downstream stages see the rewritten query
        if verbose:
            marker = "(changed)" if rw.changed else "(unchanged)"
            print(f"[rewriter] {marker} → {rw.rewritten!r}")

    # Step 1: Classify
    clf = classify(question)
    if verbose:
        print(f"[classifier] domain={clf.domain} subdomains={clf.subdomains} | "
              f"time={clf.time_period.type} | recency={clf.recency_boost}")

    # Step 2: Retrieve
    candidates = _retrieve(question, clf)
    if verbose:
        print(f"[retrieval] {len(candidates)} candidates returned")

    if not candidates:
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
        )

    # Step 3: Rerank
    rerank_input = [(row[0], row[1]) for row in candidates]
    reranked = rerank(question, rerank_input, k=TOP_K)
    if verbose:
        print(f"[reranker] top {len(reranked)} chunks selected")

    meta_lookup = {row[0]: row for row in candidates}
    top_chunks = [
        meta_lookup[chunk_id]
        for chunk_id, _, _ in reranked
        if chunk_id in meta_lookup
    ]

    # Step 4: Generate
    result = _generate(question, top_chunks, clf)
    result.latency_ms = int((time.perf_counter() - t_start) * 1000)
    result.original_query = original_query
    result.rewritten_query = rewritten_query
    result.rewrite_changed = rewrite_changed

    if verbose:
        print(f"[generator] confidence={result.confidence} | "
              f"citations={len(result.citations)} | "
              f"latency={result.latency_ms}ms | "
              f"tokens={result.tokens_in}in/{result.tokens_out}out")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.WARNING)

    args = sys.argv[1:]
    enable_rewrite = False
    if "--rewrite" in args:
        enable_rewrite = True
        args.remove("--rewrite")

    question = " ".join(args) if args else None
    if not question:
        print("Usage: python -m rag.query [--rewrite] <question>")
        print('Example: python -m rag.query "Koji je rok za predaju PDV obrasca?"')
        sys.exit(1)

    print(f"\nPitanje: {question}\n{'='*60}")
    result = ask(question, verbose=True, enable_rewrite=enable_rewrite)
    print(f"\n{'='*60}")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
