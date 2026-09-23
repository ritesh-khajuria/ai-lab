"""
TEST for detect.py - the half of this agent that can be tested exactly.

WHY THIS FILE CAN EXIST AT ALL
------------------------------
Every anomaly in claims_history.csv was PLANTED, at a known day, with a known
size. So "did the detector find it?" has a right answer, and a test can assert
it. No model is involved, nothing costs money, and the result is identical on
every run.

That is the whole reason detection lives in code. Compare with explain.py:
"is this a good explanation?" has no single right answer, cannot be asserted,
and needs an eval that scores tendencies instead of a test that proves facts.

    detect.py    a TEST     - exact, free, deterministic
    explain.py   an EVAL    - scored, costs money, run deliberately

Knowing which of those two a component needs is most of what evaluating an AI
system is.

MEASURES BOTH DIRECTIONS
------------------------
  RECALL    did we find everything planted?     -> a miss is a silent failure
  PRECISION did we flag things we should not?   -> noise trains people to ignore
                                                   the alerts entirely

A detector that flags everything has perfect recall and is worthless. Both
numbers or neither.

    ../.venv/bin/python test_detect.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from detect import detect_all, MIN_Z, MIN_RATE_JUMP, MIN_VOLUME_MULTIPLE  # noqa: E402
from make_history import PLANTED, GLP1                                    # noqa: E402

# A detector tuned to never cry wolf will also miss real problems. One marginal
# false positive out of ~11 drugs is a deliberate trade, not an accident - so
# state the budget out loud and fail if it grows.
FALSE_POSITIVE_BUDGET = 1


def subjects(findings, kind):
    return {f["subject"] for f in findings if f["kind"] == kind}


def test_glp1_spike(findings):
    """The policy change: all three GLP-1 drugs should spike together."""
    flagged = subjects(findings, "reject_rate_spike")
    missing = GLP1 - flagged
    assert not missing, f"planted GLP-1 spike not detected for: {sorted(missing)}"


def test_code_mix_identifies_75(findings):
    """A rate jump says something broke; the code says WHAT broke."""
    flagged = subjects(findings, "reject_code_mix_shift")
    expected = f"reject code {PLANTED['glp1_pa_spike']['code']}"
    assert expected in flagged, \
        f"expected {expected!r} in the code-mix findings, got {sorted(flagged)}"


def test_prescriber_surge(findings):
    who = PLANTED["prescriber_volume"]["prescriber"]
    flagged = subjects(findings, "prescriber_volume_shift")
    assert who in flagged, f"planted volume surge for {who} not detected"

    hit = next(f for f in findings
               if f["kind"] == "prescriber_volume_shift" and f["subject"] == who)
    planted = PLANTED["prescriber_volume"]["multiplier"]
    # Generous band: the surge is real but sampling wobbles around the target.
    assert planted * 0.7 <= hit["multiple"] <= planted * 1.3, \
        (f"detected {hit['multiple']}x but planted {planted}x - the fixture and "
         f"the detector disagree about how big this is")


def test_new_drug(findings):
    drug = PLANTED["new_drug"]["drug"]
    flagged = subjects(findings, "new_drug")
    assert drug in flagged, f"{drug} appeared for the first time and was not flagged"


def test_quiet_drugs_stay_quiet(findings):
    """PRECISION. Drugs with no planted change should mostly not appear.

    This is the test that stops someone 'fixing' a miss by dropping the
    thresholds to zero.
    """
    planted_subjects = GLP1 | {PLANTED["new_drug"]["drug"]}
    noise = [f for f in findings
             if f["kind"] == "reject_rate_spike" and f["subject"] not in planted_subjects]
    assert len(noise) <= FALSE_POSITIVE_BUDGET, (
        f"{len(noise)} false positives, budget is {FALSE_POSITIVE_BUDGET}: "
        f"{[f['subject'] for f in noise]}")


TESTS = [test_glp1_spike, test_code_mix_identifies_75, test_prescriber_surge,
         test_new_drug, test_quiet_drugs_stay_quiet]


def main():
    if not os.path.exists(os.path.join(HERE, "claims_history.csv")):
        print("\n  claims_history.csv missing - run make_history.py first\n")
        return 1

    findings, span = detect_all()
    print(f"\nthresholds: z>={MIN_Z}  jump>={MIN_RATE_JUMP:.0%}  "
          f"volume>={MIN_VOLUME_MULTIPLE}x")
    print(f"recent {span['recent_from']}..{span['recent_to']}  "
          f"vs baseline {span['baseline_from']}..{span['baseline_to']}")
    print(f"{len(findings)} finding(s)\n")

    passed = failed = 0
    for test in TESTS:
        name = test.__name__.replace("test_", "").replace("_", " ")
        try:
            test(findings)
        except AssertionError as e:
            print(f"  FAIL  {name}\n          {e}")
            failed += 1
        else:
            print(f"  pass  {name}")
            passed += 1

    planted_total = len(GLP1) + 3          # 3 drugs + code mix + prescriber + new drug
    print(f"\n{passed} passed, {failed} failed")
    print(f"recall: {planted_total}/{planted_total} planted anomalies detected"
          if not failed else "")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
