"""
EVAL HARNESS - part 3: did the AGENT actually find the right policy?

THE BLIND SPOT THIS CLOSES
--------------------------
eval_retrieval.py scores retrieval using questions *I* wrote. It proves the
index can find the right document when asked well. It never sees the question
the AGENT writes - and the agent writes its own.

So the two can disagree completely:

    eval_retrieval.py :  "Is Ozempic covered by the plan?"   -> hit@3 100%  PASS
    the real agent    :  "Trulicity prior authorization..."  -> may miss entirely

A component eval that passes while the system fails is worse than no eval,
because it produces confidence instead of information.

This one runs the REAL agent, records every search it chose to make, and asks a
different question: of everything the agent actually saw, was the document that
holds the answer in there at all?

    eval_retrieval.py   is the index good?      free, run constantly
    this file           did the agent find it?  costs API calls, run deliberately

HOW IT WORKS
------------
It swaps the agent's search_policy tool for a recording wrapper. Same function
underneath, so behaviour is unchanged - we simply keep a note of every query the
model composed and every document that came back.

COSTS MONEY, so it will not run without --run.

RUN:
    ../.venv/bin/python eval_end_to_end.py            # dry run - shows the plan
    ../.venv/bin/python eval_end_to_end.py --run      # actually calls the model
"""
import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.abspath(os.path.join(HERE, "..", "claims_agent"))
sys.path.insert(0, AGENT_DIR)

# What the answer SHOULD have come from, decided before we look at any run.
CASES = [
    {"claim_id": "CLM1004", "expect": "plan_a_prior_authorization.md",
     "why": "Wegovy, code 75 - a GLP-1 needing prior authorization"},
    {"claim_id": "CLM1006", "expect": "plan_a_refill_policy.md",
     "why": "oxycodone, code 79 - Schedule II refill timing"},
    {"claim_id": "CLM1001", "expect": "plan_a_dur_policy.md",
     "why": "Ozempic, code 88 - a drug utilization review reject"},
]


def main():
    print(f"\n{len(CASES)} claim(s), one real agent run each:")
    for c in CASES:
        print(f"  {c['claim_id']}  expect {c['expect']:<32} {c['why']}")

    if "--run" not in sys.argv:
        print("\nDry run. Add --run to actually call the model (this costs money).\n")
        return 0

    import claims_agent as agent
    from claims_agent_policy_search import search_policy as real_search

    calls = []

    def recording_search(question, top_k=3):
        """Same search, but we keep the receipt.

        This is the whole trick: the agent cannot tell the difference, so we
        measure real behaviour rather than a simulation of it.
        """
        hits = real_search(question, top_k=top_k)
        calls.append({"question": question,
                      "sources": [h["source"] for h in hits]})
        return hits

    agent.TOOLS_BY_NAME["search_policy"] = recording_search

    found = 0
    print()
    for case in CASES:
        calls.clear()
        # The agent prints its whole draft as it works. Useful when you are
        # running it for real, noise when you are scoring it - so swallow it
        # and report only what this eval is actually measuring.
        with contextlib.redirect_stdout(io.StringIO()):
            agent.resolve_claim(
                f"Claim {case['claim_id']} was rejected. Investigate why using "
                f"the tools and the plan policy, then explain the fix.")

        seen = {src for call in calls for src in call["sources"]}
        hit = case["expect"] in seen
        found += hit

        print(f"{case['claim_id']}  expected {case['expect']}")
        if not calls:
            print("   the agent never searched policy at all")
        for call in calls:
            print(f"   asked : {call['question']}")
            print(f"   got   : {', '.join(call['sources'])}")
        print(f"   -> {'FOUND' if hit else 'NEVER SAW IT'}\n")

    n = len(CASES)
    print("=" * 60)
    print(f"end-to-end retrieval: {found}/{n} ({found/n:.0%})")
    print("(eval_retrieval.py measures the index; this measures the agent using it)\n")
    return 0 if found == n else 1


if __name__ == "__main__":
    sys.exit(main())
