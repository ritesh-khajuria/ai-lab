"""
CLAIMS AGENT - build the work queue.

This is the UPSTREAM step. It looks at the claims system and decides which
claims need the agent's attention.

    claims.csv  --(filter status = REJECTED)-->  reject_queue.csv

Why this matters: before, reject_queue.csv was hand-written by me, so it could
say anything - including CLM9999, which does not exist. Now the queue is
DERIVED from the source data, so it can only contain real claims.

In production this is a SQL query:
    SELECT claim_id FROM claims WHERE status = 'REJECTED' AND adjudicated_on = :run_date

Run from the ai-lab folder:
    .venv/bin/python claims_agent_build_queue.py
"""
import os
import csv

# Anchor every path to THIS file's folder, so the script works
# no matter which directory it is run from (Airflow runs it from elsewhere).
HERE = os.path.dirname(os.path.abspath(__file__))
CLAIMS_FILE = os.path.join(HERE, "claims.csv")
QUEUE_FILE = os.path.join(HERE, "reject_queue.csv")
TARGET_STATUS = "REJECTED"


def build_queue():
    """Write the claim_ids needing attention. Returns how many."""
    rejected = []
    total = 0

    with open(CLAIMS_FILE, "r") as f:
        for row in csv.DictReader(f):
            total += 1
            if row["status"].strip().upper() == TARGET_STATUS:
                rejected.append(row["claim_id"].strip())

    with open(QUEUE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["claim_id"])          # header
        for claim_id in rejected:
            writer.writerow([claim_id])

    print(f"[queue] read {total} claims from {CLAIMS_FILE}")
    print(f"[queue] {len(rejected)} have status={TARGET_STATUS}")
    print(f"[queue] wrote {QUEUE_FILE}: {', '.join(rejected) if rejected else '(empty)'}")
    return len(rejected)


if __name__ == "__main__":
    build_queue()
