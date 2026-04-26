"""Run-time tracer: orchestrates the full pipeline and logs every stage.

Pipeline:
    semantic retrieval → reranking → Claude generation

Each call to `run_query()` produces:
  - The generated answer + retrieved chunks (returned to the caller)
  - A JSON file in logs/runs/ capturing the full trace, including LLM stage
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from rag.embedder import embed_query, MODEL_NAME as EMBEDDER_NAME
from rag.retrieve.rerank import rerank, MODEL_NAME as RERANKER_NAME
from rag.generate.answerer import answer, system_prompt_hash, DEFAULT_MODEL

load_dotenv()


LOGS_DIR = Path("logs/runs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# Cost lookup. Values are USD per 1K tokens. Update when pricing changes.
# If a model isn't here, cost is reported as None instead of guessed.
PRICING_PER_1K = {
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
    "claude-opus-4-7":   {"input": 0.015, "output": 0.075},
    "claude-haiku-4-5-20251001": {"input": 0.001, "output": 0.005},
}


# ---------------------------------------------------------------------------
# Trace data structures
# ---------------------------------------------------------------------------

@dataclass
class StageTrace:
    name: str
    latency_ms: int
    data: dict = field(default_factory=dict)


@dataclass
class RunTrace:
    timestamp: str
    query: str
    config: dict
    stages: list[StageTrace] = field(default_factory=list)
    total_latency_ms: int = 0
    error: str | None = None

    def add_stage(self, name: str, latency_ms: int, data: dict) -> None:
        self.stages.append(StageTrace(name=name, latency_ms=latency_ms, data=data))

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "query": self.query,
            "config": self.config,
            "stages": [asdict(s) for s in self.stages],
            "total_latency_ms": self.total_latency_ms,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]


def _log_path(query: str, when: datetime) -> Path:
    stamp = when.strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"{stamp}_{_query_hash(query)}.json"


def _now_ms() -> float:
    return time.perf_counter() * 1000


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    pricing = PRICING_PER_1K.get(model)
    if not pricing:
        return None
    return round(
        (input_tokens / 1000) * pricing["input"]
        + (output_tokens / 1000) * pricing["output"],
        6,
    )


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

CANDIDATE_POOL = 20
TOP_K = 5


def _semantic_candidates(query: str, n: int) -> list[dict]:
    qvec = embed_query(query)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, article_number, chunk_text,
                       1 - (embedding <=> %s) AS similarity,
                       law_name, source, valid_from, valid_to, status, citable, nn_reference
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (qvec, qvec, n),
            )
            rows = cur.fetchall()
    return [
        {
            "chunk_id": r[0],
            "article_number": r[1],
            "chunk_text": r[2],
            "semantic_similarity": float(r[3]),
            "law_name": r[4],
            "source": r[5],
            "valid_from": r[6].isoformat() if r[6] else None,
            "valid_to": r[7].isoformat() if r[7] else None,
            "status": r[8],
            "citable": r[9],
            "nn_reference": r[10],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    query: str
    top_chunks: list[dict]
    answer_text: str
    refused: bool
    citations: list[dict]
    temporal_note: str | None
    trace_path: Path
    total_latency_ms: int
    estimated_cost_usd: float | None


def run_query(
    query: str,
    *,
    candidate_pool: int = CANDIDATE_POOL,
    top_k: int = TOP_K,
    persist: bool = True,
    skip_generation: bool = False,
) -> RunResult:
    """Run query through retrieval + generation, log everything, return results.

    `skip_generation=True` returns retrieval-only results — useful for
    debugging the retrieval layer without spending API tokens.
    """
    started_at = datetime.now(timezone.utc)
    pipeline_start = _now_ms()

    trace = RunTrace(
        timestamp=started_at.isoformat(),
        query=query,
        config={
            "embedder": EMBEDDER_NAME,
            "reranker": RERANKER_NAME,
            "generator": DEFAULT_MODEL if not skip_generation else None,
            "system_prompt_hash": system_prompt_hash() if not skip_generation else None,
            "candidate_pool": candidate_pool,
            "top_k": top_k,
            "git_sha": _git_sha(),
        },
    )

    answer_text = ""
    refused = False
    citations: list[dict] = []
    temporal_note: str | None = None
    estimated_cost: float | None = None

    log_path = _log_path(query, started_at)

    try:
        # Stage 1: semantic retrieval
        t0 = _now_ms()
        candidates = _semantic_candidates(query, n=candidate_pool)
        trace.add_stage(
            name="semantic_retrieval",
            latency_ms=int(_now_ms() - t0),
            data={
                "n_candidates": len(candidates),
                "candidates": [
                    {
                        "rank": i + 1,
                        "chunk_id": c["chunk_id"],
                        "article_number": c["article_number"],
                        "similarity": c["semantic_similarity"],
                        "preview": c["chunk_text"][:120],
                    }
                    for i, c in enumerate(candidates)
                ],
            },
        )

        # Stage 2: reranking
        t0 = _now_ms()
        rerank_input = [(c["chunk_id"], c["chunk_text"]) for c in candidates]
        ranked = rerank(query, rerank_input, k=top_k)
        by_id = {c["chunk_id"]: c for c in candidates}
        top_chunks = []
        for cid, text, score in ranked:
            chunk = dict(by_id[cid])
            chunk["rerank_score"] = score
            top_chunks.append(chunk)
        trace.add_stage(
            name="rerank",
            latency_ms=int(_now_ms() - t0),
            data={
                "top_k": [
                    {
                        "rank": i + 1,
                        "chunk_id": c["chunk_id"],
                        "article_number": c["article_number"],
                        "rerank_score": c["rerank_score"],
                        "preview": c["chunk_text"][:120],
                    }
                    for i, c in enumerate(top_chunks)
                ],
                "max_score": top_chunks[0]["rerank_score"] if top_chunks else None,
            },
        )

        # Stage 3: generation (optional)
        if not skip_generation:
            t0 = _now_ms()
            gen = answer(query, top_chunks)
            estimated_cost = _estimate_cost(
                gen.model, gen.input_tokens, gen.output_tokens,
            )
            answer_text = gen.answer
            refused = gen.refused
            citations = gen.citations
            temporal_note = gen.temporal_note
            trace.add_stage(
                name="generation",
                latency_ms=int(_now_ms() - t0),
                data={
                    "model": gen.model,
                    "input_tokens": gen.input_tokens,
                    "output_tokens": gen.output_tokens,
                    "estimated_cost_usd": estimated_cost,
                    "refused": gen.refused,
                    "answer": gen.answer,
                    "citations": gen.citations,
                    "temporal_note": gen.temporal_note,
                    "raw_response": gen.raw_response,
                },
            )

    except Exception as exc:
        trace.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        trace.total_latency_ms = int(_now_ms() - pipeline_start)
        if persist:
            log_path.write_text(
                json.dumps(trace.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    return RunResult(
        query=query,
        top_chunks=top_chunks,
        answer_text=answer_text,
        refused=refused,
        citations=citations,
        temporal_note=temporal_note,
        trace_path=log_path if persist else Path("/dev/null"),
        total_latency_ms=trace.total_latency_ms,
        estimated_cost_usd=estimated_cost,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage: python -m rag.retrieve.runner "<query>"')
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    result = run_query(q)
    print(f"\n{'=' * 70}")
    print(f"Query:   {result.query}")
    print(f"Latency: {result.total_latency_ms} ms")
    if result.estimated_cost_usd is not None:
        print(f"Cost:    ${result.estimated_cost_usd:.5f}")
    print(f"Trace:   {result.trace_path}")
    print(f"Refused: {result.refused}")
    print(f"\nAnswer:\n{result.answer_text}")
    if result.citations:
        print(f"\nCitations:")
        for c in result.citations:
            print(f"  - {c.get('law_name')}, {c.get('nn_reference')}, čl. {c.get('article_number')}")
    if result.temporal_note:
        print(f"\nTemporal note: {result.temporal_note}")