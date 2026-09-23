"""
CLAIMS AGENT - chase the humans.

Airflow cannot WAIT for a reviewer. But it can NUDGE them, and that is a
perfectly good Airflow task: it is scheduled, it finishes, it never blocks.

Finds reviews still PENDING beyond the SLA and sends a reminder.

    ESCALATE_AFTER_HOURS   how old before we nudge (default 24)
                           set to 0 to nudge everything - useful for a demo
    ESCALATE_TO            who to email (default riteshkhajuria238@gmail.com)

EMAIL: real sending needs SMTP credentials. If SMTP_USER and SMTP_PASSWORD are
set in the environment, this sends a real email. If they are NOT set, it writes
the notice to escalations.log and prints it - so the task still works and you
can see exactly what would have been sent.

To enable real email with Gmail you would add to .env:
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=your.address@gmail.com
    SMTP_PASSWORD=<a Google APP PASSWORD, not your account password>
Never put a real account password in a file.

Run from this folder:
    ../.venv/bin/python escalate_stale.py
"""
import os
import smtplib
import sqlite3
from datetime import datetime, timedelta
from email.message import EmailMessage

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(os.path.dirname(HERE), ".env"))

DB_FILE = os.path.join(HERE, "claims_reviews.db")
LOG_FILE = os.path.join(HERE, "escalations.log")

HOURS = float(os.environ.get("ESCALATE_AFTER_HOURS", "24"))
TO_ADDR = os.environ.get("ESCALATE_TO", "riteshkhajuria238@gmail.com")


def find_stale():
    """Reviews still PENDING past the SLA."""
    if not os.path.exists(DB_FILE):
        return []
    cutoff = (datetime.now() - timedelta(hours=HOURS)).isoformat(timespec="seconds")
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT review_id, claim_id, created_at FROM reviews "
        "WHERE status = 'PENDING' AND created_at <= ? ORDER BY created_at",
        (cutoff,)).fetchall()
    con.close()
    return [dict(r) for r in rows]


def build_message(stale):
    lines = [
        f"{len(stale)} claim review(s) are still awaiting a decision.",
        "",
        f"SLA threshold: {HOURS} hour(s).",
        "",
        f"{'review':<8}{'claim':<12}{'raised at':<22}age",
        "-" * 60,
    ]
    now = datetime.now()
    for r in stale:
        age = now - datetime.fromisoformat(r["created_at"])
        hrs = age.total_seconds() / 3600
        lines.append(f"{r['review_id']:<8}{r['claim_id']:<12}"
                     f"{r['created_at']:<22}{hrs:.1f}h")
    lines += ["", "Review them at: http://127.0.0.1:8100/reviews?status=PENDING",
              "", "-- automated nudge from the claims agent pipeline"]
    return "\n".join(lines)


def send(subject, body):
    """Send by SMTP if configured; otherwise log it so nothing is silently lost."""
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")

    if user and password:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = TO_ADDR
        msg.set_content(body)
        host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        port = int(os.environ.get("SMTP_PORT", "587"))
        with smtplib.SMTP(host, port) as s:
            s.starttls()
            s.login(user, password)
            s.send_message(msg)
        return f"emailed {TO_ADDR}"

    # Not configured - write it down rather than pretend it went out.
    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'=' * 70}\n{datetime.now().isoformat(timespec='seconds')}\n"
                f"TO: {TO_ADDR}\nSUBJECT: {subject}\n\n{body}\n")
    return f"SMTP not configured - written to {os.path.basename(LOG_FILE)}"


if __name__ == "__main__":
    stale = find_stale()
    print(f"[escalate] SLA = {HOURS}h, notify = {TO_ADDR}")

    if not stale:
        # Nothing overdue is a HEALTHY result, not a failure.
        print("[escalate] nothing overdue")
        raise SystemExit(0)

    subject = f"[ACTION NEEDED] {len(stale)} claim review(s) overdue"
    body = build_message(stale)
    result = send(subject, body)

    print(f"[escalate] {len(stale)} overdue: "
          f"{', '.join(r['claim_id'] for r in stale)}")
    print(f"[escalate] {result}")
    print("\n" + body)
