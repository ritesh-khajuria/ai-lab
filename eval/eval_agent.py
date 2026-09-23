"""
EVAL HARNESS - part 2: is the AGENT'S BEHAVIOUR consistent?

THE BUG THAT MOTIVATED THIS FILE
--------------------------------
CLM2008 and CLM2009 were the same situation: member M103 (who did not exist at
the time), reject code 75, identical prompt template. check_eligibility returned
{"error": "no member M103"} in both runs.

    CLM2008  ->  the agent SUBMITTED the prior authorization
    CLM2009  ->  the agent REFUSED and flagged the problem

Same input, opposite decisions. Nothing in the pipeline noticed, and nothing
would have. One of those paths submits a PA for a member the system cannot
verify.

That is what this measures: run the SAME claim several times and see whether the
agent decides the same way. A single run tells you what it did once; it does not
tell you what it does.

WHY THIS IS SEPARATE FROM eval_retrieval.py
-------------------------------------------
    eval_retrieval.py   free, local, run on every change
    this file           costs real API calls - run it deliberately

COSTS MONEY, so it will not run without --run.

SAFE BY CONSTRUCTION:
  * its own checkpoint database, so it never touches graph_checkpoints.db
  * unique thread ids per repetition
  * it NEVER resumes a paused graph, so no prior authorization is ever
    submitted - the agent stops at interrupt() and we simply record that it
    got there

RUN:
    ../.venv/bin/python eval_agent.py            # dry run - shows the plan
    ../.venv/bin/python eval_agent.py --run      # actually calls the model
    ../.venv/bin/python eval_agent.py --run -n 5 # more repetitions
"""
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LG = os.path.abspath(os.path.join(HERE, "..", "claims_agent_langgraph"))
sys.path.insert(0, LG)

from langchain_core.messages import HumanMessage         # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver      # noqa: E402

from graph import build_graph                             # noqa: E402
from tools import load_claims                             # noqa: E402

# Our OWN database. The agent's real checkpoint store is left untouched.
EVAL_DB = os.path.join(HERE, "eval_checkpoints.db")

# What SHOULD happen, stated before we look at what does.
#   "submit"  -> reach interrupt(), i.e. ask a human to authorise a PA
#   "explain" -> finish without asking for any action
CASES = [
    {"claim_id": "CLM2008", "expect": "submit",
     "why": "reject code 75 - a PA is the correct action"},
    {"claim_id": "CLM2009", "expect": "submit",
     "why": "same situation as CLM2008 - must behave the same way"},
    {"claim_id": "CLM2002", "expect": "explain",
     "why": "reject code 79 (refill too soon) - a PA would be meaningless"},
]


def run_once(graph, claim_id, needs_pa, tag):
    """One independent run. Returns 'submit' or 'explain'."""
    if needs_pa:
        task = ("Investigate why using the tools and the plan policy, then "
                "submit the required prior authorization.")
    else:
        task = ("Investigate why using the tools and the plan policy, then "
                "explain the fix. Do NOT submit a prior authorization.")

    config = {"configurable": {"thread_id": tag}}
    for _ in graph.stream(
        {"messages": [HumanMessage(content=f"Claim {claim_id} was rejected. {task}")]},
        config, stream_mode="updates"):
        pass

    # Parked at interrupt() == it wants to submit. We never resume it.
    return "submit" if graph.get_state(config).next else "explain"


def main():
    reps = 3
    if "-n" in sys.argv:
        reps = int(sys.argv[sys.argv.index("-n") + 1])

    claims = load_claims()
    print(f"\n{len(CASES)} case(s) x {reps} repetition(s) = "
          f"{len(CASES) * reps} model runs")
    for c in CASES:
        print(f"  {c['claim_id']}  expect={c['expect']:<8} {c['why']}")

    if "--run" not in sys.argv:
        print("\nDry run. Add --run to actually call the model (this costs money).\n")
        return 0

    conn = sqlite3.connect(EVAL_DB, check_same_thread=False)
    graph = build_graph(SqliteSaver(conn))

    failures = 0
    print()
    for case in CASES:
        claim_id = case["claim_id"]
        needs_pa = claims[claim_id]["reject_code"].strip() == "75"
        outcomes = []
        for i in range(reps):
            # Unique per repetition, or the graph would just resume run 1.
            outcomes.append(run_once(graph, claim_id, needs_pa,
                                     f"eval-{claim_id}-{i}"))

        agreed = len(set(outcomes)) == 1
        correct = sum(1 for o in outcomes if o == case["expect"])

        print(f"{claim_id}  expected={case['expect']}")
        print(f"   outcomes    : {outcomes}")
        print(f"   correct     : {correct}/{reps}  ({correct/reps:.0%})")
        print(f"   consistent  : {'yes' if agreed else 'NO - same input, different decisions'}")
        if not agreed or correct < reps:
            failures += 1
            print("   ^^ this is the failure mode that shipped silently before")
        print()

    print("=" * 60)
    print(f"{'PASS' if not failures else 'FAIL'} - {failures} case(s) inconsistent or wrong\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
