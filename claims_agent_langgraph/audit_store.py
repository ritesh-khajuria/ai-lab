"""
LangGraph version - THE AUDIT TRAIL.

THE GAP THIS CLOSES
-------------------
This pipeline could ACT but could not ACCOUNT for its actions.

    approve  ->  a row appeared in pa_submissions.csv
    reject   ->  nothing durable at all

"SUBMISSION BLOCKED by r.khajuria" existed only inside a serialized checkpoint
blob in graph_checkpoints.db. So the question a pharmacy compliance team
actually asks - "who denied this PA, when, and why?" - had no queryable answer.
And denials are the ones that get audited.

WHY ONE TABLE HERE, WHEN claims_agent HAS TWO
---------------------------------------------
claims_agent/review_store.py has `reviews` (mutable status) + `decisions`
(append-only). It needs `reviews` because the draft and its PENDING status live
nowhere else.

Here, PENDING is not ours to store. A paused thread IS the pending state, and
the checkpointer owns it - that is what /pending already reads. If we also kept
a `reviews` table saying "PENDING", we would have two sources of truth for the
same fact, and they would drift the first time a graph was resumed by any other
route.

    Rule: do not duplicate state you can derive.
          Do persist events you cannot reconstruct.

A decision is an event. It is not recoverable from anywhere else in a form you
can query, so it goes here - append-only, never updated, never deleted.

FEDERATION
----------
The columns are deliberately compatible with claims_agent's `decisions`, so one
query can answer "who decided what" across BOTH pipelines:

    sqlite3 ../claims_agent/claims_reviews.db
    ATTACH '../claims_agent_langgraph/lg_audit.db' AS lg;

    SELECT 'claims_agent' AS pipeline, decision, reviewer, notes, decided_at
      FROM decisions
    UNION ALL
    SELECT 'langgraph', decision, reviewer, notes, decided_at
      FROM lg.decisions
    ORDER BY decided_at;

Identical schemas are what make that possible later. That is the whole reason
to keep the shape the same even though the storage is separate.
"""
import os
import sqlite3
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(HERE, "lg_audit.db")

# NOTE: no UPDATE and no DELETE anywhere in this file. That is the point.
SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id    TEXT NOT NULL,
    thread_id   TEXT NOT NULL,
    action      TEXT NOT NULL,      -- the tool that was gated, e.g. submit_prior_auth
    drug        TEXT,
    decision    TEXT NOT NULL,      -- APPROVED / REJECTED
    reviewer    TEXT NOT NULL,
    notes       TEXT,               -- the human's justification
    result      TEXT,               -- what the tool ACTUALLY returned afterwards
    source      TEXT NOT NULL,      -- 'api' (reviewer UI) or 'cli' (approve.py)
    decided_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decisions_claim ON decisions(claim_id);
"""


def connect():
    con = sqlite3.connect(DB_FILE, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init():
    con = connect()
    con.executescript(SCHEMA)
    con.commit()
    con.close()


def record(claim_id, thread_id, action, decision, reviewer,
           drug=None, notes=None, result=None, source="api"):
    """Append one decision. Returns the new decision_id.

    `result` is captured AFTER the graph resumes, so the log holds what the tool
    really did - not what we intended it to do. On a rejection that is the
    "SUBMISSION BLOCKED" message; on an approval it is the PA reference.
    """
    init()
    con = connect()
    cur = con.execute(
        "INSERT INTO decisions (claim_id, thread_id, action, drug, decision, "
        "reviewer, notes, result, source, decided_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (claim_id, thread_id, action, drug, decision, reviewer, notes, result,
         source, datetime.now().isoformat(timespec="seconds")))
    con.commit()
    decision_id = cur.lastrowid
    con.close()
    return decision_id


def history(claim_id=None):
    """Every decision, or every decision for one claim. Oldest first."""
    init()
    con = connect()
    if claim_id:
        rows = con.execute(
            "SELECT * FROM decisions WHERE claim_id = ? ORDER BY decision_id",
            (claim_id,)).fetchall()
    else:
        rows = con.execute("SELECT * FROM decisions ORDER BY decision_id").fetchall()
    con.close()
    return [dict(r) for r in rows]


def decision_count():
    init()
    con = connect()
    n = con.execute("SELECT count(*) FROM decisions").fetchone()[0]
    con.close()
    return n


if __name__ == "__main__":
    # Quick look at the log:  ../.venv/bin/python audit_store.py
    rows = history()
    if not rows:
        print(f"no decisions recorded yet in {DB_FILE}")
    for r in rows:
        print(f"{r['decided_at']}  {r['claim_id']:<8} {r['decision']:<8} "
              f"by {r['reviewer']:<12} ({r['source']})")
        print(f"    drug   : {r['drug'] or '-'}")
        print(f"    reason : {r['notes'] or '(none given)'}")
        print(f"    result : {(r['result'] or '')[:120]}")
