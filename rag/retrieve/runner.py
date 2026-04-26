"""Run-time tracer: wraps the retrieval pipeline and logs every stage to disk.

Each call to `run_query()` produces:
  - The answer / retrieved chunks (returned to the caller)
  - A JSON file in logs/runs/ capturing the full trace

The trace schema is intentionally extensible: new pipeline stages
(classifier, generation, etc.) can be added without breaking older logs.
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
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from rag.embedder import embed_query, MODEL_NAME as EMBEDDER_NAME
from rag.retrieve.rerank import rerank, MODEL_NAME as RERANKER_NAME

load_dotenv()


LOGS_DIR = Path("logs/runs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Trace data structures
# ---------------------------------------------------------------------------

@dataclass
class StageTrace:
    """One pipeline stage's worth of timing + payload."""
    name: str
    latency_ms: int
    data: dict = field(default_factory=dict)


@dataclass
class RunTrace:
    """Full trace for one query going through the pipeline."""
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
    """Best-effort capture of current commit. Returns 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _query_hash(query: str) -> str:
    """Short stable hash, used in log filenames so re-runs of the same
    query don't collide."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]


def _log_path(query: str, when: datetime) -> Path:
    stamp = when.strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"{stamp}_{_query_hash(query)}.json"


def _now_ms() -> float:
    return time.perf_counter() * 1000


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

CANDIDATE_POOL = 20
TOP_K = 5


def _semantic_candidates(query: str, n: int) -> list[dict]:
    """Pull n nearest chunks by cosine distance."""
    qvec = embed_query(query)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, article_number, chunk_text,
                       1 - (embedding <=> %s) AS similarity,
                       law_name, source, valid_from, valid_to, status, citable
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
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """What `run_query()` returns to the caller. Mirrors what the trace contains
    but in convenient form for downstream code (eval harness, future API)."""
    query: str
    top_chunks: list[dict]
    trace_path: Path
    total_latency_ms: int


def run_query(
    query: str,
    *,
    candidate_pool: int = CANDIDATE_POOL,
    top_k: int = TOP_K,
    persist: bool = True,
) -> RunResult:
    """Run query through the full retrieval pipeline, log the trace, return results.

    `persist=False` skips writing the trace to disk — useful for tests.
    """
    started_at = datetime.now(timezone.utc)
    pipeline_start = _now_ms()

    trace = RunTrace(
        timestamp=started_at.isoformat(),
        query=query,
        config={
            "embedder": EMBEDDER_NAME,
            "reranker": RERANKER_NAME,
            "candidate_pool": candidate_pool,
            "top_k": top_k,
            "git_sha": _git_sha(),
        },
    )

    try:
        # Stage 1: semantic candidates
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
        # build a lookup so we can attach metadata to the reranked output
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

    except Exception as exc:
        trace.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        trace.total_latency_ms = int(_now_ms() - pipeline_start)
        if persist:
            log_path = _log_path(query, started_at)
            log_path.write_text(
                json.dumps(trace.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    return RunResult(
        query=query,
        top_chunks=top_chunks,
        trace_path=log_path if persist else Path("/dev/null"),
        total_latency_ms=trace.total_latency_ms,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage: python -m rag.retrieve.runner "<query>"')
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    result = run_query(q)
    print(f"\nTrace written to: {result.trace_path}")
    print(f"Total latency: {result.total_latency_ms} ms\n")
    for i, chunk in enumerate(result.top_chunks, start=1):
        art = chunk["article_number"] or "[preamble]"
        score = chunk["rerank_score"]
        preview = chunk["chunk_text"][:120].replace("\n", " ")
        print(f"  #{i}  {score:.4f}  Članak {art:6s}  {preview}...")
