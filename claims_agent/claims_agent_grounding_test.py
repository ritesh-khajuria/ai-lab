"""
Phase 3 experiment - WHERE DOES THE ANSWER COME FROM?

The claims agent gave a confident recommendation ("get a PA, then resubmit").
Nothing in our code contained that logic. So where did it come from?

This script probes the boundary:
  TEST A - a question our TOOLS can fully answer   -> grounded, safe
  TEST B - a question our tools CANNOT answer      -> the risk zone

We reuse the exact same agent (imported from claims_agent.py - the
`if __name__ == "__main__"` guard in that file means importing won't re-run it).

Run from the ai-lab folder:
    .venv/bin/python phase3_grounding_test.py
"""
from claims_agent import resolve_claim

print("=" * 70)
print("TEST A - inside the tools' knowledge")
print("Question: why did CLM1002 reject, and how do we fix it?")
print("(our tools DO know: the claim, the reject code, the member)")
print("=" * 70)
resolve_claim("Claim CLM1002 was rejected. Explain why and how to fix it.")

print("\n\n" + "=" * 70)
print("TEST B - BEYOND the tools' knowledge  <-- the risk zone")
print("Question: what are PLAN-A's clinical criteria for approving Ozempic,")
print("          and will this member qualify?")
print("(NO tool of ours knows PLAN-A's policy. Watch what it does.)")
print("=" * 70)
resolve_claim(
    "For claim CLM1001, what are PLAN-A's specific clinical criteria for "
    "approving an Ozempic prior authorization, and will this member qualify?"
)
