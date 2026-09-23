"""
LangGraph version - THE HUMAN GATE, as a command line tool.

This stands in for the reviewer's UI. It is deliberately NOT an Airflow task:
the decision happens on a human's schedule, not the scheduler's.

    ../.venv/bin/python approve.py CLM1004 approve r.khajuria "Meets criteria"
    ../.venv/bin/python approve.py CLM1004 reject  s.mishra   "No HbA1c on file"

Resuming restarts the graph INSIDE submit_prior_auth, exactly where it stopped -
possibly days later, in this completely separate process.
"""
import sys

from langgraph.types import Command

import audit_store
from graph import build_graph, make_checkpointer
from run_batch import thread_id_for


def interrupt_payload(state):
    """What is this paused thread asking a human to authorise?

    Must be read BEFORE resuming - afterwards the interrupt is gone.
    """
    for task in state.tasks:
        for itr in getattr(task, "interrupts", []):
            if isinstance(itr.value, dict):
                return itr.value
    return {}


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    claim_id, action, reviewer = sys.argv[1], sys.argv[2].lower(), sys.argv[3]
    reason = sys.argv[4] if len(sys.argv) > 4 else ""

    if action not in ("approve", "reject"):
        print("action must be 'approve' or 'reject'")
        sys.exit(1)

    graph = build_graph(make_checkpointer())
    config = {"configurable": {"thread_id": thread_id_for(claim_id)}}

    state = graph.get_state(config)
    if not state.values:
        print(f"no graph run exists for {claim_id} - run run_batch.py first")
        sys.exit(1)
    if not state.next:
        print(f"{claim_id} is not waiting for approval (already finished)")
        sys.exit(1)

    payload = interrupt_payload(state)

    print(f"resuming {claim_id} with decision={action} by {reviewer}\n")

    tool_result = ""
    # Command(resume=...) sends this value back into the interrupt() call.
    for event in graph.stream(
        Command(resume={"approved": action == "approve",
                        "reviewer": reviewer,
                        "reason": reason}),
        config, stream_mode="updates"):
        for node, update in event.items():
            for msg in update.get("messages", []):
                text = getattr(msg, "content", "")
                if isinstance(text, list):
                    text = " ".join(b.get("text", "") for b in text
                                    if isinstance(b, dict))
                text = str(text).strip()
                if text:
                    print(f"[{node}] {text[:500]}")
                    if node == "tools":
                        tool_result = text

    # Same audit log the API writes to - the record must not depend on WHICH
    # door the human walked through.
    audit_store.record(
        claim_id=claim_id, thread_id=thread_id_for(claim_id),
        action=payload.get("action", "unknown"), drug=payload.get("drug"),
        decision="APPROVED" if action == "approve" else "REJECTED",
        reviewer=reviewer, notes=reason, result=tool_result, source="cli")

    final = graph.get_state(config)
    print(f"\nthread now {'still paused' if final.next else 'COMPLETE'}")
    print(f"decision written to the audit log ({audit_store.DB_FILE})")


if __name__ == "__main__":
    main()
