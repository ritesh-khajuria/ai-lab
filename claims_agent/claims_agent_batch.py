"""
CLAIMS AGENT - batch runner.

Reads a queue of rejected claims, runs the agent on each one, writes a draft
per claim, and reports run statistics. This is the WORK that an Airflow task
will call - Airflow only schedules and monitors it.

Every piece of this you already built in day01_practice.py:
    try/except per record   ->  one bad claim must not kill the run
    continue + reject file  ->  failures are retryable
    checkpoint + done set   ->  never reprocess a completed claim
    counters                ->  run statistics for the job log

THE ONE DESIGN DECISION - FAIL_FAST:
    False (default) : a bad claim is logged and skipped. The run SUCCEEDS.
    True            : a bad claim raises. The run FAILS.
    Set with:  CLAIMS_FAIL_FAST=1

Run from the ai-lab folder:
    .venv/bin/python claims_agent_batch.py
"""
import os
import csv
from datetime import datetime

from claims_agent import resolve_claim, CLAIMS
import review_store          # GAP 1 FIX - the batch now writes to the review queue

# Anchor every path to THIS file's folder, so the script works
# no matter which directory it is run from (Airflow runs it from elsewhere).
HERE = os.path.dirname(os.path.abspath(__file__))
QUEUE_FILE = os.path.join(HERE, "reject_queue.csv")
CHECKPOINT_FILE = os.path.join(HERE, "claims_checkpoint.csv")
FAILED_FILE = os.path.join(HERE, "claims_failed.csv")
DRAFTS_DIR = os.path.join(HERE, "drafts")

# Read the switch from the environment so Airflow can set it per-DAG.
FAIL_FAST = os.environ.get("CLAIMS_FAIL_FAST", "0") == "1"


# ---------------------------------------------------------------------------
def read_queue():
    """Return the list of claim_ids waiting to be processed."""
    claim_ids = []
    try:
        with open(QUEUE_FILE, "r") as f:
            for row in csv.DictReader(f):
                claim_ids.append(row["claim_id"].strip())
    except FileNotFoundError:
        pass                      # empty queue is a normal, healthy state
    return claim_ids


def load_checkpoint():
    """Which claims have we already completed? Survives restarts."""
    done = set()
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            for line in f:
                done.add(line.split("|")[0].strip())
    except FileNotFoundError:
        pass                      # first run
    return done


def record_done(claim_id):
    """Append to the checkpoint AFTER the work succeeded."""
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(f"{claim_id}|{datetime.now().isoformat(timespec='seconds')}\n")


def record_failure(claim_id, error):
    """Append to the reject file so it can be fixed and retried."""
    with open(FAILED_FILE, "a") as f:
        f.write(f"{claim_id}|{datetime.now().isoformat(timespec='seconds')}|{error}\n")


# ---------------------------------------------------------------------------
def process_one(claim_id):
    """Run the agent on one claim. Raises if anything is wrong."""
    # Validate the queue entry BEFORE spending money on an API call.
    if claim_id not in CLAIMS:
        raise ValueError(f"{claim_id} is not present in the claims system")

    draft = resolve_claim(
        f"Claim {claim_id} was rejected. Explain why and how to fix it."
    )

    if not draft:
        raise RuntimeError(f"agent returned no draft for {claim_id}")

    os.makedirs(DRAFTS_DIR, exist_ok=True)
    path = os.path.join(DRAFTS_DIR, f"{claim_id}.md")
    with open(path, "w") as f:
        f.write(draft)

    # GAP 1 FIX: the draft file is for humans to read; the REVIEW RECORD is what
    # puts it in front of a reviewer. Without this line the overnight batch
    # produced drafts that no UI would ever show.
    review_id, created = review_store.add_review(claim_id, draft, source="batch")
    return path, review_id, created


# ---------------------------------------------------------------------------
def run_batch():
    """Process the whole queue. Returns a stats dict."""
    review_store.init()               # make sure the review queue exists
    queue = read_queue()
    done = load_checkpoint()
    stats = {"queued": len(queue), "processed": 0,
             "already_done": 0, "failed": 0}

    print(f"[batch] queue={len(queue)}  already completed={len(done)}  "
          f"FAIL_FAST={FAIL_FAST}")

    # A queue of zero is NOT an error. It proves the pipeline ran and found
    # nothing to do - which is very different from the pipeline not running.
    if not queue:
        print("[batch] nothing to do")
        return stats

    for claim_id in queue:
        if claim_id in done:
            print(f"[batch] {claim_id}: already done, skipping")
            stats["already_done"] += 1
            continue

        try:
            path, review_id, created = process_one(claim_id)
            record_done(claim_id)
            stats["processed"] += 1
            queued = "queued for review" if created else "review already pending"
            print(f"[batch] {claim_id}: OK -> {os.path.basename(path)} "
                  f"(review {review_id}, {queued})")

        except Exception as e:
            stats["failed"] += 1
            record_failure(claim_id, e)
            print(f"[batch] {claim_id}: FAILED - {e}")
            if FAIL_FAST:
                # Re-raise: the Airflow task turns red, downstream tasks skip.
                raise
            # otherwise: logged for retry, carry on with the next claim

    print(f"[batch] queued={stats['queued']} processed={stats['processed']} "
          f"already_done={stats['already_done']} failed={stats['failed']}")
    return stats


if __name__ == "__main__":
    run_batch()
