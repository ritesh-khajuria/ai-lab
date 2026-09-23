"""
LangGraph version - WATCH IT PAUSE.

Runs the agent on CLM1004 (Wegovy, reject code 75 = prior authorization
required). The agent will gather facts, check policy, and then try to SUBMIT a
prior authorization - at which point the graph STOPS and waits for you.

Then we resume it twice, on two separate conversations:
    thread "approve-demo"  -> a human approves  -> the PA is submitted
    thread "reject-demo"   -> a human refuses   -> nothing is submitted

Run from this folder:
    ../.venv/bin/python run_local.py
"""
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph import build_graph, make_checkpointer

line = "=" * 76


def show_new_messages(events):
    """Print anything the agent says or does as the graph runs."""
    interrupted = None
    for event in events:
        if "__interrupt__" in event:
            interrupted = event["__interrupt__"]
            continue
        for node, update in event.items():
            for msg in update.get("messages", []):
                calls = getattr(msg, "tool_calls", None)
                if calls:
                    for c in calls:
                        print(f"   [{node}] -> calling {c['name']}({c['args']})")
                elif getattr(msg, "content", None):
                    text = msg.content
                    if isinstance(text, list):          # content blocks
                        text = " ".join(b.get("text", "") for b in text
                                        if isinstance(b, dict))
                    text = str(text).strip()
                    if text:
                        print(f"   [{node}] {text[:400]}")
    return interrupted


def run_until_pause(graph, thread_id, claim_id):
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n>>> starting thread {thread_id!r} on {claim_id}\n")
    events = graph.stream(
        # NOTE the wording. An earlier version said "...and if a prior
        # authorization is required, submit one." On one run the agent chose to
        # summarise instead of submitting - same prompt, different behaviour.
        # If an action must happen, INSTRUCT it; do not leave it to inference.
        {"messages": [HumanMessage(
            content=f"Claim {claim_id} was rejected. Investigate why using the "
                    f"tools and the plan policy, then submit the required "
                    f"prior authorization.")]},
        config, stream_mode="updates")
    return show_new_messages(events), config


if __name__ == "__main__":
    checkpointer = make_checkpointer()
    graph = build_graph(checkpointer)

    # ===================================================================
    print("\n" + line)
    print("PART 1 - run the graph. It should STOP before acting.")
    print(line)
    interrupt_info, config = run_until_pause(graph, "approve-demo", "CLM1004")

    state = graph.get_state(config)
    print("\n" + "-" * 76)
    print("GRAPH IS PAUSED")
    print("-" * 76)
    print(f"  next node waiting to run : {state.next}")
    print(f"  messages saved so far    : {len(state.values['messages'])}")
    if interrupt_info:
        payload = interrupt_info[0].value
        print("\n  The human is being asked to approve:")
        for k, v in payload.items():
            print(f"     {k:<14}: {str(v)[:150]}")
    print("\n  NOTHING HAS BEEN SUBMITTED. The state is on disk in")
    print("  graph_checkpoints.db - this could sit here for a week.")

    # ===================================================================
    print("\n" + line)
    print("PART 2 - a human APPROVES. Execution resumes mid-tool.")
    print(line)
    events = graph.stream(
        Command(resume={"approved": True,
                        "reviewer": "r.khajuria",
                        "reason": "Meets PLAN-A GLP-1 criteria"}),
        config, stream_mode="updates")
    show_new_messages(events)

    # ===================================================================
    print("\n" + line)
    print("PART 3 - same claim, different thread, human REFUSES.")
    print(line)
    _, config2 = run_until_pause(graph, "reject-demo", "CLM1004")
    events = graph.stream(
        Command(resume={"approved": False,
                        "reviewer": "s.mishra",
                        "reason": "No HbA1c on file - get labs first"}),
        config2, stream_mode="updates")
    show_new_messages(events)

    print("\n" + line)
    print("Same agent, same claim. The human decision changed what happened.")
    print("Check pa_submissions.csv - it should contain ONE submission, not two.")
    print(line + "\n")
