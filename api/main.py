"""FastAPI web interface for the RRiF AI Agent."""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import bcrypt
import psycopg
from psycopg.types.json import Jsonb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
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

# ── JWT config ────────────────────────────────────────────────────────────────

JWT_SECRET = os.environ["JWT_SECRET"]
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 12

security = HTTPBearer()

def create_token(username: str, name: str) -> str:
    payload = {
        "sub": username,
        "name": name,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {"username": payload["sub"], "name": payload["name"]}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ── DB ────────────────────────────────────────────────────────────────────────

def get_db():
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    try:
        yield conn
    finally:
        conn.close()

# ── Models ────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class QueryRequest(BaseModel):
    question: str

class FeedbackRequest(BaseModel):
    query_id: str
    rating: int
    accuracy_verdict: str | None = None
    would_send_to_client: str | None = None
    failure_mode: str | None = None
    comment: str | None = None
    suggested_answer: str | None = None

# ── Auth endpoint ─────────────────────────────────────────────────────────────

@app.post("/api/login")
def login_endpoint(req: LoginRequest, db=Depends(get_db)):
    with db.cursor() as cur:
        cur.execute(
            "SELECT password_hash, name FROM users WHERE username = %s AND is_active = TRUE",
            (req.username.strip(),)
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Pogrešno korisničko ime ili lozinka.")

    password_hash, name = row
    if not bcrypt.checkpw(req.password.encode(), password_hash.encode()):
        raise HTTPException(status_code=401, detail="Pogrešno korisničko ime ili lozinka.")

    token = create_token(req.username.strip(), name)
    return {"token": token, "username": req.username.strip(), "name": name}

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/query")
def query_endpoint(req: QueryRequest, db=Depends(get_db), user=Depends(verify_token)):
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
            user["username"],
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
def feedback_endpoint(req: FeedbackRequest, db=Depends(get_db), user=Depends(verify_token)):
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
            req.query_id, user["username"], req.rating,
            req.accuracy_verdict, req.would_send_to_client,
            req.failure_mode, req.comment, req.suggested_answer,
        ))
    db.commit()
    return {"status": "ok"}


@app.get("/api/admin/stats")
def admin_stats(db=Depends(get_db), user=Depends(verify_token)):
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
        "accuracy_strict_pct": accuracy_pct,
        "accuracy_acceptable_pct": acceptable_pct,
        "queries_by_category": by_category,
    }


# ── Static frontend (mount last so API routes take priority) ──────────────────

frontend_dir = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")