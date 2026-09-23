"""
CLAIMS AGENT - the review queue, in one place.

GAP 1 FIX.

Before this file, there were two separate output paths:

    claims_agent_batch.py  ->  drafts/*.md          (files, no review record)
    claims_agent_api.py    ->  claims_reviews.db    (the review queue)

...so a draft produced by the overnight batch NEVER reached a reviewer. A UI
reading the review queue would show an empty list.

Now BOTH write here. One queue, two producers.

Two tables, on purpose:
    reviews    current state of each draft   (status can change)
    decisions  every decision ever made      (append-only - history cannot be rewritten)
"""
import os
import sqlite3
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(HERE, "claims_reviews.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    review_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id    TEXT NOT NULL,
    draft       TEXT NOT NULL,
    status      TEXT NOT NULL,          -- PENDING / APPROVED / REJECTED
    source      TEXT NOT NULL DEFAULT 'api',   -- which producer made it
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id   INTEGER NOT NULL,
    decision    TEXT NOT NULL,
    reviewer    TEXT NOT NULL,
    notes       TEXT,
    decided_at  TEXT NOT NULL,
    FOREIGN KEY (review_id) REFERENCES reviews(review_id)
);
"""


def connect():
    con = sqlite3.connect(DB_FILE, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init():
    con = connect()
    con.executescript(SCHEMA)
    # Older databases were created without `source`; add it if missing.
    cols = [r["name"] for r in con.execute("PRAGMA table_info(reviews)")]
    if "source" not in cols:
        con.execute("ALTER TABLE reviews ADD COLUMN source TEXT NOT NULL DEFAULT 'api'")
    con.commit()
    con.close()


def add_review(claim_id, draft, source="api"):
    """Queue a draft for review.

    IDEMPOTENT: if this claim already has a PENDING review, return that one
    instead of creating a duplicate. Returns (review_id, created_new).
    """
    con = connect()
    existing = con.execute(
        "SELECT review_id FROM reviews WHERE claim_id = ? AND status = 'PENDING' "
        "ORDER BY review_id DESC LIMIT 1", (claim_id,)).fetchone()
    if existing:
        con.close()
        return existing["review_id"], False

    cur = con.execute(
        "INSERT INTO reviews (claim_id, draft, status, source, created_at) "
        "VALUES (?, ?, 'PENDING', ?, ?)",
        (claim_id, draft, source, datetime.now().isoformat(timespec="seconds")))
    con.commit()
    review_id = cur.lastrowid
    con.close()
    return review_id, True


def list_reviews(status=None):
    con = connect()
    if status:
        rows = con.execute(
            "SELECT review_id, claim_id, status, source, created_at FROM reviews "
            "WHERE status = ? ORDER BY review_id", (status.upper(),)).fetchall()
    else:
        rows = con.execute(
            "SELECT review_id, claim_id, status, source, created_at FROM reviews "
            "ORDER BY review_id").fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_review(review_id):
    con = connect()
    row = con.execute("SELECT * FROM reviews WHERE review_id = ?",
                      (review_id,)).fetchone()
    con.close()
    return dict(row) if row else None


def decide(review_id, decision, reviewer, notes=None):
    """Record a human decision. Returns (ok, message_or_row)."""
    con = connect()
    row = con.execute("SELECT * FROM reviews WHERE review_id = ?",
                      (review_id,)).fetchone()
    if row is None:
        con.close()
        return False, "no such review"
    if row["status"] != "PENDING":
        con.close()
        return False, f"review {review_id} was already {row['status']}"

    now = datetime.now().isoformat(timespec="seconds")
    con.execute("UPDATE reviews SET status = ? WHERE review_id = ?",
                (decision, review_id))
    con.execute(
        "INSERT INTO decisions (review_id, decision, reviewer, notes, decided_at) "
        "VALUES (?, ?, ?, ?, ?)", (review_id, decision, reviewer, notes, now))
    con.commit()
    con.close()
    return True, {"review_id": review_id, "claim_id": row["claim_id"],
                  "status": decision, "reviewer": reviewer, "decided_at": now}


def history(review_id):
    con = connect()
    rows = con.execute(
        "SELECT decision, reviewer, notes, decided_at FROM decisions "
        "WHERE review_id = ? ORDER BY decision_id", (review_id,)).fetchall()
    con.close()
    return [dict(r) for r in rows]


def pending_count():
    con = connect()
    n = con.execute("SELECT count(*) FROM reviews WHERE status='PENDING'").fetchone()[0]
    con.close()
    return n
