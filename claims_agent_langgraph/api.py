"""
LangGraph version - APPROVAL API.

approve.py is a CLI. This exposes the same thing over HTTP so a UI can drive it.

    GET  /health                        liveness + how many are parked
    GET  /pending                       every graph waiting on a human
    GET  /pending/{claim_id}            what this one is asking you to approve
    POST /pending/{claim_id}/decision   resume it - approve or reject

The shape deliberately MIRRORS claims_agent_api.py, so one reviewer UI can talk
to both queues with only the base URL changing.

But note the difference underneath:
    claims_agent_api  a decision UPDATES A ROW. The work was already finished.
    this api          a decision RESUMES A PAUSED PROGRAM. The work happens now.

START (from this folder):
    ../.venv/bin/uvicorn api:app --port 8200
    http://127.0.0.1:8200/docs
"""
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langgraph.types import Command

# This file lives inside the package folder, so make sibling modules importable
# when uvicorn is started from elsewhere.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import audit_store                                         # noqa: E402
from graph import build_graph, make_checkpointer          # noqa: E402
# Import the FUNCTION, not the dict. Binding the dict here would freeze the
# claim list at startup and hide every claim added afterwards.
from tools import load_claims                              # noqa: E402
# Registry-aware: a corrected claim gets a NEW thread (v2, v3...), so hardcoding
# "claim-<id>" here would leave the reviewer approving a superseded version.
from run_batch import thread_id_for                        # noqa: E402

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Building the graph loads the model - do it ONCE at startup.
    print("Building the graph and opening the checkpointer...")
    state["graph"] = build_graph(make_checkpointer())
    audit_store.init()
    print(f"Ready. {len(load_claims())} claims in claims_lg.csv, "
          f"{audit_store.decision_count()} decisions in the audit log.")
    yield
    state.clear()


app = FastAPI(
    title="Claims Agent (LangGraph) - Approval API",
    description="Approve or reject prior-authorization submissions that are "
                "paused mid-execution, waiting for a human.",
    version="1.0.0",
    lifespan=lifespan,
)


class PendingItem(BaseModel):
    claim_id: str
    thread_id: str
    action: str
    drug: Optional[str] = None
    prompt: Optional[str] = None


class PendingDetail(PendingItem):
    justification: Optional[str] = None


class DecisionRequest(BaseModel):
    decision: str = Field(..., pattern="^(APPROVED|REJECTED)$")
    reviewer: str = Field(..., min_length=2)
    notes: Optional[str] = None


class DecisionResponse(BaseModel):
    claim_id: str
    decision: str
    reviewer: str
    result: str            # what the tool actually returned
    thread_complete: bool


def _interrupt_payload(claim_id):
    """Return the interrupt payload for a parked thread, or None."""
    config = {"configurable": {"thread_id": thread_id_for(claim_id)}}
    snap = state["graph"].get_state(config)
    if not snap.values or not snap.next:
        return None
    for task in snap.tasks:
        for itr in getattr(task, "interrupts", []):
            if isinstance(itr.value, dict):
                return itr.value
    return {}          # parked, but no payload we can read


@app.get("/health")
def health():
    claims = load_claims()          # fresh read, every request
    parked = sum(1 for c in claims if _interrupt_payload(c) is not None)
    return {"status": "ok", "claims_loaded": len(claims), "pending_reviews": parked}


@app.get("/pending", response_model=list[PendingItem])
def pending():
    """Every graph currently paused, waiting for a human."""
    out = []
    for claim_id in load_claims():   # fresh read, so new claims appear at once
        payload = _interrupt_payload(claim_id)
        if payload is None:
            continue
        out.append(PendingItem(
            claim_id=claim_id, thread_id=thread_id_for(claim_id),
            action=payload.get("action", "unknown"),
            drug=payload.get("drug"), prompt=payload.get("prompt")))
    return out


@app.get("/pending/{claim_id}", response_model=PendingDetail)
def pending_one(claim_id: str):
    payload = _interrupt_payload(claim_id)
    if payload is None:
        raise HTTPException(status_code=404,
                            detail=f"{claim_id} is not waiting for approval")
    return PendingDetail(
        claim_id=claim_id, thread_id=thread_id_for(claim_id),
        action=payload.get("action", "unknown"), drug=payload.get("drug"),
        prompt=payload.get("prompt"), justification=payload.get("justification"))


@app.get("/decisions")
def decisions(claim_id: Optional[str] = None):
    """The audit trail: every decision a human ever made here.

    Unlike /pending (which is derived live from the checkpointer), this is a
    permanent append-only record - including REJECTIONS, which leave no other
    durable trace.
    """
    return audit_store.history(claim_id)


@app.post("/pending/{claim_id}/decision", response_model=DecisionResponse)
def decide(claim_id: str, req: DecisionRequest):
    """Resume the paused graph with the human's decision.

    This does NOT just record something - it restarts execution inside
    submit_prior_auth(), and the submission either happens or does not.
    """
    # Read the payload BEFORE resuming - once the graph moves on, the interrupt
    # is gone and we can no longer see what was being authorised.
    payload = _interrupt_payload(claim_id)
    if payload is None:
        raise HTTPException(status_code=409,
                            detail=f"{claim_id} is not waiting for approval")

    config = {"configurable": {"thread_id": thread_id_for(claim_id)}}
    approved = req.decision == "APPROVED"

    tool_result = ""
    for event in state["graph"].stream(
        Command(resume={"approved": approved,
                        "reviewer": req.reviewer,
                        "reason": req.notes or ""}),
        config, stream_mode="updates"):
        for node, update in event.items():
            if node != "tools":
                continue
            for msg in update.get("messages", []):
                text = getattr(msg, "content", "")
                if isinstance(text, list):
                    text = " ".join(b.get("text", "") for b in text
                                    if isinstance(b, dict))
                if text:
                    tool_result = str(text).strip()

    # Append to the audit log AFTER the resume, so `result` is what the tool
    # really did - the PA reference on approval, or the BLOCKED message on a
    # rejection. A rejection used to leave no durable trace at all.
    audit_store.record(
        claim_id=claim_id, thread_id=thread_id_for(claim_id),
        action=payload.get("action", "unknown"), drug=payload.get("drug"),
        decision=req.decision, reviewer=req.reviewer, notes=req.notes,
        result=tool_result, source="api")

    snap = state["graph"].get_state(config)
    return DecisionResponse(claim_id=claim_id, decision=req.decision,
                            reviewer=req.reviewer,
                            result=tool_result or "(no tool output captured)",
                            thread_complete=not snap.next)
