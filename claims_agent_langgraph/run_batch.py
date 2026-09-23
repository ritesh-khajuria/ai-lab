"""
LangGraph version - BATCH STARTER.

THE KEY IDEA, and the whole reason this differs from ../claims_agent/:

    An Airflow task MUST FINISH.
    A graph awaiting human approval MAY WAIT FOR DAYS.

So this task does NOT wait. For each rejected claim it starts a graph run and
lets it proceed to its own natural stopping point, which is one of two things:

    COMPLETED            the agent finished - no action was needed
    AWAITING_APPROVAL    the agent stopped at interrupt(), and the state is
                         saved on disk until a human decides

Either way THIS TASK EXITS. The waiting happens outside Airflow.

Run from this folder:
    ../.venv/bin/python run_batch.py
"""
import csv
import hashlib
import os
from datetime import datetime

from langchain_core.messages import HumanMessage

from graph import build_graph, make_checkpointer
from tools import load_claims

HERE = os.path.dirname(os.path.abspath(__file__))
STATUS_FILE = os.path.join(HERE, "graph_runs.csv")
THREADS_FILE = os.path.join(HERE, "graph_threads.csv")


# ---------------------------------------------------------------------------
# CHANGE DATA CAPTURE
#
# The old skip rule was "does a finished thread exist for this claim?" - which
# answers the WRONG question. It means a claim whose data was CORRECTED is
# skipped forever: fix a miskeyed reject code and the pipeline says
# ALREADY_DONE and quietly ignores the correction. In a claims system that is
# the bug nobody finds until an audit.
#
# The right question is "has this claim changed since we last processed it?"
# That is a content signature - the same idea as a hash diff in a warehouse
# load. Same signature -> skip (cheap, idempotent). Different signature ->
# reprocess, because it is genuinely different work.
# ---------------------------------------------------------------------------
SIGNATURE_FIELDS = ("member_id", "drug", "ndc", "quantity", "days_supply",
                    "prescriber", "plan", "status", "reject_code")


def signature_of(row):
    """A short hash of the fields that would change the answer.

    claim_id is deliberately EXCLUDED - it identifies the claim, it is not part
    of what makes the work different. Everything that could change the agent's
    reasoning is included.
    """
    raw = "|".join(str(row.get(f, "")).strip() for f in SIGNATURE_FIELDS)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def load_threads():
    """claim_id -> the NEWEST {thread_id, signature, version}.

    The file is APPEND-ONLY, keyed on (claim_id, version): every version a claim
    has ever had stays in it. Overwriting the row would have thrown away the
    version history the version column exists to record - the same mistake as
    keeping only a mutable status and no decision log.

    We fold to the highest version per claim to answer "where is this claim now".
    """
    reg = {}
    try:
        with open(THREADS_FILE, "r") as f:
            for r in csv.DictReader(f):
                version = int(r["version"])
                current = reg.get(r["claim_id"])
                if current is None or version > current["version"]:
                    reg[r["claim_id"]] = {"thread_id": r["thread_id"],
                                          "signature": r["signature"],
                                          "version": version}
    except FileNotFoundError:
        pass                       # first run - no registry yet
    return reg


def record_thread(reg, claim_id, thread_id, signature, version):
    """APPEND one (claim_id, version) row, and update the in-memory view."""
    is_new_file = not os.path.exists(THREADS_FILE)
    with open(THREADS_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if is_new_file:
            w.writerow(["claim_id", "thread_id", "signature", "version", "updated_at"])
        w.writerow([claim_id, thread_id, signature, version,
                    datetime.now().isoformat(timespec="seconds")])
    reg[claim_id] = {"thread_id": thread_id, "signature": signature,
                     "version": version}


def thread_id_for(claim_id):
    """The CURRENT thread for a claim.

    Not always "claim-<id>" any more. When a claim is corrected we start a NEW
    thread (v2, v3...) instead of overwriting the old one, so the earlier
    reasoning and its human decision stay intact - you can still answer "what
    did we decide on the previous version of this claim?".

    The reviewer API and approve.py both call this, so they follow a claim to
    its latest thread automatically.
    """
    entry = load_threads().get(claim_id)
    return entry["thread_id"] if entry else f"claim-{claim_id}"


def rejected_claims(claims):
    return [c for c, row in claims.items()
            if row["status"].strip().upper() == "REJECTED"]


def run_one(graph, claim_id, claims, reg):
    """Start (or continue) the graph for one claim. Never blocks on a human.

    Mutates `reg` (the thread registry) - the caller writes it out once.
    """
    row = claims[claim_id]
    sig = signature_of(row)
    entry = reg.get(claim_id)

    if entry is None:
        # No registry entry. There may still be a thread from before signatures
        # existed - ADOPT it rather than redoing work we already paid for.
        legacy_id = f"claim-{claim_id}"
        legacy = graph.get_state({"configurable": {"thread_id": legacy_id}})
        if legacy.values:
            record_thread(reg, claim_id, legacy_id, sig, 1)
            if legacy.next:
                return "AWAITING_APPROVAL", "already paused, waiting on a human"
            return "ALREADY_DONE", "adopted existing thread, signature recorded"
        thread_id, version = legacy_id, 1

    elif entry["signature"] == sig:
        # Unchanged since we last processed it - the cheap, common path.
        existing = graph.get_state({"configurable": {"thread_id": entry["thread_id"]}})
        if existing.values:
            if existing.next:
                return "AWAITING_APPROVAL", "already paused, waiting on a human"
            return "ALREADY_DONE", "unchanged since the last run"
        thread_id, version = entry["thread_id"], entry["version"]

    else:
        # THE CLAIM WAS CORRECTED. New thread, so the old one stays readable.
        version = entry["version"] + 1
        thread_id = f"claim-{claim_id}-v{version}"

    config = {"configurable": {"thread_id": thread_id}}
    record_thread(reg, claim_id, thread_id, sig, version)
    changed = version > 1

    # ROUTE IN CODE, NOT IN THE PROMPT.
    #
    # The first version of this said "...if and only if the reject code is 75,
    # submit the prior authorization" and left the model to evaluate that
    # condition. It declined on every claim, including the one that qualified.
    #
    # We already KNOW the reject code - it is right here in the data. A
    # deterministic condition belongs in an `if`, not in a sentence the model
    # has to interpret. Ask the model to reason; make the code decide routing.
    needs_pa = row["reject_code"].strip() == "75"
    if needs_pa:
        task = ("Investigate why using the tools and the plan policy, then "
                "submit the required prior authorization.")
    else:
        task = ("Investigate why using the tools and the plan policy, then "
                "explain the fix. Do NOT submit a prior authorization.")

    for _ in graph.stream(
        {"messages": [HumanMessage(content=f"Claim {claim_id} was rejected. {task}")]},
        config, stream_mode="updates"):
        pass                                   # we do not need the events here

    why = f"claim was corrected - reprocessed as v{version}" if changed else ""
    state = graph.get_state(config)
    if state.next:                             # graph is parked at an interrupt
        return "AWAITING_APPROVAL", why or "stopped before acting - needs approval"
    return "COMPLETED", why or "finished, no action required"


def list_pending(graph, claims):
    """Which threads are parked waiting for a human?"""
    pending = []
    for claim_id in rejected_claims(claims):
        config = {"configurable": {"thread_id": thread_id_for(claim_id)}}
        state = graph.get_state(config)
        if state.values and state.next:
            what = ""
            for task in state.tasks:
                for itr in getattr(task, "interrupts", []):
                    what = itr.value.get("action", "")
            pending.append((claim_id, thread_id_for(claim_id), what))
    return pending


if __name__ == "__main__":
    graph = build_graph(make_checkpointer())

    all_claims = load_claims()          # fresh read every run
    claims = rejected_claims(all_claims)
    reg = load_threads()
    print(f"[graph-batch] {len(claims)} rejected claims: {', '.join(claims)}")

    counts = {}
    rows = []
    for claim_id in claims:
        outcome, note = run_one(graph, claim_id, all_claims, reg)
        counts[outcome] = counts.get(outcome, 0) + 1
        rows.append((claim_id, outcome, note))
        print(f"[graph-batch] {claim_id}: {outcome} - {note}")

    # Write the registry ONCE, after the loop - so a crash mid-batch cannot
    # leave a claim marked processed when its thread never finished.
    # APPEND, never overwrite. This used to open with "w", so every run erased
    # the one before it and run-over-run history was impossible - the same
    # mistake as overwriting the thread registry.
    #
    # The old file has a 3-column header; appending 4-column rows under it would
    # misparse, so rotate it once instead of silently corrupting it.
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            if (f.readline() or "").count(",") == 2:
                os.rename(STATUS_FILE, os.path.join(HERE, "graph_runs_v1.csv"))
                print("[graph-batch] rotated old graph_runs.csv -> graph_runs_v1.csv")

    run_at = datetime.now().isoformat(timespec="seconds")
    is_new_file = not os.path.exists(STATUS_FILE)
    with open(STATUS_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if is_new_file:
            w.writerow(["run_at", "claim_id", "outcome", "note"])
        for claim_id, outcome, note in rows:
            w.writerow([run_at, claim_id, outcome, note])

    print("[graph-batch] " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    pending = list_pending(graph, all_claims)
    print(f"\n[graph-batch] {len(pending)} thread(s) awaiting human approval:")
    for claim_id, thread, action in pending:
        print(f"[graph-batch]    {claim_id}  thread={thread}  action={action}")
    if pending:
        print("[graph-batch] approve with:  ../.venv/bin/python approve.py <claim_id> "
              "approve|reject <reviewer> \"<reason>\"")
