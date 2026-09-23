"""
CLAIMS AGENT - REST API with a human review workflow.

This closes two gaps:
  1. The agent is now SERVED as an API, not just runnable in batch.
  2. Drafts go into a review QUEUE with an approve/reject action and an
     audit trail - instead of just landing in a folder.

    POST /claims/{claim_id}/resolve   run the agent NOW, wait for the draft
    GET  /reviews                     the specialist's queue
    GET  /reviews/{id}                one draft
    POST /reviews/{id}/decision       approve / reject  (recorded, auditable)
    GET  /health                      liveness

SYNCHRONOUS by choice: the caller waits ~20s for the agent to finish. Simple
to reason about and to debug. The async version (return an id, poll later) is
the alternative we will look at afterwards.

STORAGE is SQLite, two tables:
    reviews    - current state of each draft   (mutable)
    decisions  - every decision ever made      (append-only audit log)
That split matters: you can change a review's status, but you can never
rewrite the history of who decided what.

START THE SERVER (from the ai-lab folder):
    .venv/bin/uvicorn claims_agent_api:app --reload --port 8100

THEN open the interactive docs:
    http://127.0.0.1:8100/docs
"""
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import review_store

# Anchor every path to THIS file's folder, so the script works
# no matter which directory it is run from (Airflow runs it from elsewhere).
HERE = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(HERE, "claims_reviews.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    review_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id    TEXT NOT NULL,
    draft       TEXT NOT NULL,
    status      TEXT NOT NULL,          -- PENDING / APPROVED / REJECTED
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id   INTEGER NOT NULL,
    decision    TEXT NOT NULL,          -- APPROVED / REJECTED
    reviewer    TEXT NOT NULL,
    notes       TEXT,
    decided_at  TEXT NOT NULL,
    FOREIGN KEY (review_id) REFERENCES reviews(review_id)
);
"""


def db():
    """Open a connection. check_same_thread=False because FastAPI serves
    requests on a threadpool."""
    con = sqlite3.connect(DB_FILE, check_same_thread=False)
    con.row_factory = sqlite3.Row          # rows behave like dicts
    return con


# ---------------------------------------------------------------------------
# STARTUP - load the agent ONCE. Importing it pulls in the embedding model and
# the policy index, which takes a few seconds. Doing that per request would be
# unusable.
# ---------------------------------------------------------------------------
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the claims agent (model + policy index)...")
    # We keep the FUNCTION, not the data. Claims are re-read on every request.
    #
    # The first version cached `CLAIMS` here at startup - so a claim added to
    # claims.csv while the server was running returned 404 forever. A real
    # service queries the claims database per request; a cached snapshot is
    # both a bug and unrealistic.
    from claims_agent import resolve_claim, load_claims
    state["resolve_claim"] = resolve_claim
    state["load_claims"] = load_claims
    review_store.init()
    print(f"Ready. {len(load_claims())} claims currently in claims.csv.")
    yield
    state.clear()


app = FastAPI(
    title="Claims Resolution Agent",
    description="Drafts a resolution for a rejected pharmacy claim, "
                "then holds it for human review.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# THE CONTRACT - what goes in, what comes out.
# ---------------------------------------------------------------------------
class ResolveResponse(BaseModel):
    review_id: int
    claim_id: str
    status: str
    draft: str
    latency_ms: int
    reused: bool = False        # True = we returned an existing review, the
                                # agent did NOT run again (see idempotency below)


class ReviewSummary(BaseModel):
    review_id: int
    claim_id: str
    status: str
    created_at: str


class ReviewDetail(ReviewSummary):
    draft: str


class DecisionRequest(BaseModel):
    decision: str = Field(..., pattern="^(APPROVED|REJECTED)$",
                          description="APPROVED or REJECTED")
    reviewer: str = Field(..., min_length=2,
                          description="Who is making this decision")
    notes: Optional[str] = Field(None, description="Optional rationale")


class DecisionResponse(BaseModel):
    review_id: int
    claim_id: str
    status: str
    reviewer: str
    decided_at: str


# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok",
            "claims_loaded": len(state["load_claims"]()),   # read fresh
            "pending_reviews": review_store.pending_count()}


@app.post("/claims/{claim_id}/resolve", response_model=ResolveResponse)
def resolve(claim_id: str):
    """Run the agent on one claim and queue the draft for review.

    SYNCHRONOUS: this call blocks until the agent finishes (~20 seconds).
    """
    claim = state["load_claims"]().get(claim_id)   # fresh every request
    if claim is None:
        # 404 = you asked for something that does not exist.
        raise HTTPException(status_code=404,
                            detail=f"{claim_id} not found in the claims system")

    # --- GAP 2 FIX: only REJECTED claims need resolving -------------------
    # Without this, a PAID claim (e.g. CLM1003) could be "resolved", wasting
    # money on an LLM call and putting a meaningless draft in front of a
    # specialist. Validate the business precondition, not just existence.
    if claim["status"].strip().upper() != "REJECTED":
        raise HTTPException(
            status_code=409,
            detail=f"{claim_id} has status {claim['status']} - "
                   f"only REJECTED claims need resolution")

    # --- GAP 3 FIX: idempotency ------------------------------------------
    # If a draft for this claim is already awaiting review, hand back THAT one
    # rather than running the agent again and creating a duplicate. Repeating
    # the same request must not repeat the side effects.
    for r in review_store.list_reviews("PENDING"):
        if r["claim_id"] == claim_id:
            full = review_store.get_review(r["review_id"])
            return ResolveResponse(
                review_id=r["review_id"], claim_id=claim_id, status="PENDING",
                draft=full["draft"], latency_ms=0, reused=True)
    # NOTE: if the previous review was already APPROVED or REJECTED we DO
    # allow a fresh one - that is a legitimate re-resolution after new
    # information, not an accidental duplicate.

    started = time.perf_counter()
    draft = state["resolve_claim"](
        f"Claim {claim_id} was rejected. Explain why and how to fix it.")
    if not draft:
        # 502 = an upstream dependency (the model) failed us.
        raise HTTPException(status_code=502, detail="agent produced no draft")

    review_id, _ = review_store.add_review(claim_id, draft, source="api")

    return ResolveResponse(
        review_id=review_id, claim_id=claim_id, status="PENDING",
        draft=draft, latency_ms=int((time.perf_counter() - started) * 1000))


@app.get("/reviews", response_model=list[ReviewSummary])
def list_reviews(status: Optional[str] = None):
    """The specialist's queue. Filter with ?status=PENDING."""
    con = db()
    if status:
        rows = con.execute(
            "SELECT review_id, claim_id, status, created_at FROM reviews "
            "WHERE status = ? ORDER BY review_id", (status.upper(),)).fetchall()
    else:
        rows = con.execute(
            "SELECT review_id, claim_id, status, created_at FROM reviews "
            "ORDER BY review_id").fetchall()
    con.close()
    return [ReviewSummary(**dict(r)) for r in rows]


@app.get("/reviews/{review_id}", response_model=ReviewDetail)
def get_review(review_id: int):
    con = db()
    row = con.execute("SELECT * FROM reviews WHERE review_id = ?",
                      (review_id,)).fetchone()
    con.close()
    if row is None:
        raise HTTPException(status_code=404, detail="no such review")
    return ReviewDetail(**{k: row[k] for k in
                           ("review_id", "claim_id", "status", "created_at", "draft")})


@app.post("/reviews/{review_id}/decision", response_model=DecisionResponse)
def decide(review_id: int, req: DecisionRequest):
    """Approve or reject a draft. THIS is the human-in-the-loop gate."""
    con = db()
    row = con.execute("SELECT * FROM reviews WHERE review_id = ?",
                      (review_id,)).fetchone()
    if row is None:
        con.close()
        raise HTTPException(status_code=404, detail="no such review")

    # 409 = the request is valid, but conflicts with the current state.
    # A decision already made must not be silently overwritten.
    if row["status"] != "PENDING":
        con.close()
        raise HTTPException(
            status_code=409,
            detail=f"review {review_id} was already {row['status']}")

    now = datetime.now().isoformat(timespec="seconds")
    con.execute("UPDATE reviews SET status = ? WHERE review_id = ?",
                (req.decision, review_id))
    # The audit row is APPEND-ONLY. The review's status can change; this cannot.
    con.execute(
        "INSERT INTO decisions (review_id, decision, reviewer, notes, decided_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (review_id, req.decision, req.reviewer, req.notes, now))
    con.commit()
    con.close()

    return DecisionResponse(review_id=review_id, claim_id=row["claim_id"],
                            status=req.decision, reviewer=req.reviewer,
                            decided_at=now)


@app.get("/reviews/{review_id}/history")
def history(review_id: int):
    """Every decision ever recorded for this review - the audit trail."""
    con = db()
    rows = con.execute(
        "SELECT decision, reviewer, notes, decided_at FROM decisions "
        "WHERE review_id = ? ORDER BY decision_id", (review_id,)).fetchall()
    con.close()
    return [dict(r) for r in rows]
