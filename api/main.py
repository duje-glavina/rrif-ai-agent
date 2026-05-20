"""FastAPI web interface for the RRiF AI Agent."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.query import ask

load_dotenv()

app = FastAPI(title="RRiF AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DB ────────────────────────────────────────────────────────────────────────

def get_db():
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    try:
        yield conn
    finally:
        conn.close()

# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    advisor_id: str

class FeedbackRequest(BaseModel):
    query_id: str
    advisor_id: str
    rating: int                          # 1 = thumbs up, -1 = thumbs down
    accuracy_verdict: str | None = None  # correct | partially_correct | incorrect | cannot_evaluate
    would_send_to_client: str | None = None  # yes | no | with_edits
    failure_mode: str | None = None      # wrong_category | missing_source | hallucination | wrong_article | outdated | other
    comment: str | None = None
    suggested_answer: str | None = None

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/query")
def query_endpoint(req: QueryRequest, db=Depends(get_db)):
    try:
        result = ask(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    query_id = str(uuid.uuid4())
    result_dict = result.to_dict()

    # Estimate cost (claude-sonnet-4-6: $3/MTok in, $15/MTok out)
    estimated_cost = (result.tokens_in * 3 + result.tokens_out * 15) / 1_000_000

    with db.cursor() as cur:
        cur.execute("""
            INSERT INTO queries (
                query_id, advisor_id, question_text,
                classified_category,
                answer_text, citations, confidence,
                referred_to_advisor, model_used,
                tokens_in, tokens_out, latency_ms,
                estimated_cost_usd, error_flag,
                retrieved_chunk_ids, retrieved_scores,
                trace_json
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            query_id,
            req.advisor_id,
            req.question,
            result.classifier.category,
            result.answer,
            Jsonb(result_dict["citations"]),
            result.confidence,
            result.referred_to_advisor,
            "claude-sonnet-4-6",
            result.tokens_in,
            result.tokens_out,
            result.latency_ms,
            estimated_cost,
            False,
            [str(c) for c in result.retrieved_chunk_ids] or None,
            result.retrieved_scores or None,
            Jsonb(result_dict["meta"]),
        ))
    db.commit()

    return {
        "query_id": query_id,
        "answer": result.answer,
        "citations": result_dict["citations"],
        "confidence": result.confidence,
        "referred_to_advisor": result.referred_to_advisor,
        "temporal_basis": result.temporal_basis,
    }


@app.post("/api/feedback")
def feedback_endpoint(req: FeedbackRequest, db=Depends(get_db)):
    # Verify the query_id exists
    with db.cursor() as cur:
        cur.execute("SELECT 1 FROM queries WHERE query_id = %s", (req.query_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="query_id not found")

        cur.execute("""
            INSERT INTO feedback (
                query_id, advisor_id, rating,
                accuracy_verdict, would_send_to_client,
                failure_mode, comment, suggested_answer
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            req.query_id, req.advisor_id, req.rating,
            req.accuracy_verdict, req.would_send_to_client,
            req.failure_mode, req.comment, req.suggested_answer,
        ))
    db.commit()
    return {"status": "ok"}


@app.get("/api/admin/stats")
def admin_stats(db=Depends(get_db)):
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM queries")
        total_queries = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM feedback")
        total_feedback = cur.fetchone()[0]

        cur.execute("""
            SELECT accuracy_verdict, COUNT(*)
            FROM feedback
            WHERE accuracy_verdict IS NOT NULL
            GROUP BY accuracy_verdict
            ORDER BY COUNT(*) DESC
        """)
        verdict_counts = dict(cur.fetchall())

        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE accuracy_verdict = 'correct') AS correct,
                COUNT(*) FILTER (WHERE accuracy_verdict IN ('correct','partially_correct')) AS acceptable,
                COUNT(*) FILTER (WHERE accuracy_verdict IS NOT NULL) AS total_rated
            FROM feedback
        """)
        row = cur.fetchone()
        correct, acceptable, total_rated = row
        accuracy_pct = round(correct / total_rated * 100, 1) if total_rated else None
        acceptable_pct = round(acceptable / total_rated * 100, 1) if total_rated else None

        cur.execute("""
            SELECT classified_category, COUNT(*)
            FROM queries
            GROUP BY classified_category
            ORDER BY COUNT(*) DESC
        """)
        by_category = dict(cur.fetchall())

    return {
        "total_queries": total_queries,
        "total_feedback": total_feedback,
        "verdict_counts": verdict_counts,
        "accuracy_strict_pct": accuracy_pct,      # % rated 'correct'
        "accuracy_acceptable_pct": acceptable_pct, # % rated 'correct' or 'partially_correct'
        "queries_by_category": by_category,
    }


# ── Static frontend (mount last so API routes take priority) ──────────────────

frontend_dir = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")