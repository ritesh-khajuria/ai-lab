"""
CLAIMS ANOMALY AGENT - DETECTION.  No LLM in this file. On purpose.

THE DESIGN DECISION THIS FILE IS
--------------------------------
An LLM handed a table of daily reject rates will confidently describe patterns
that are not there, and miss ones that are. Spotting that a rate moved from 17%
to 57% is ARITHMETIC. It is deterministic, it costs nothing, it gives the same
answer every time, and you can unit-test it against a known answer.

So detection happens here, in code. The model's job starts afterwards, in
explain.py, where the question stops being "did something change?" and becomes
"what does this mean and what should someone check?" - which is judgement, and
judgement is what the model is actually for.

    Statistics find it.  The model explains it.

Same rule as routing on reject code 75 in the other pipeline: a deterministic
question belongs in code, not in a sentence a model has to interpret.

HOW IT DECIDES
--------------
Compare a RECENT window against a BASELINE window and ask whether the
difference is bigger than normal wobble. Two guards stop it crying wolf:

  * minimum volume - 3 rejections out of 5 claims is not a trend
  * a two-proportion z-score - the standard "is this difference real, or is it
    the size of the sample?" check. z >= 3 is roughly a 1-in-370 fluke.

Both thresholds are constants below, visible and arguable. A detector whose
thresholds are buried is a detector nobody trusts.

    ../.venv/bin/python detect.py
"""
import collections
import csv
import math
import os
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.path.join(HERE, "claims_history.csv")

RECENT_DAYS = 7        # the window under suspicion
BASELINE_DAYS = 28     # what "normal" is measured over
GAP_DAYS = 1           # ignore the day between, so a spike does not pollute
                       # its own baseline

MIN_RECENT_CLAIMS = 20   # below this, ignore - small numbers are noisy
MIN_RATE_JUMP = 0.15     # 15 percentage points
MIN_Z = 3.0              # ~1-in-370 chance of being a fluke
MIN_VOLUME_MULTIPLE = 2.5  # prescriber volume vs their own baseline


def load_history(path=HISTORY):
    rows = []
    with open(path, "r") as f:
        for r in csv.DictReader(f):
            r["rejected"] = r["status"].strip().upper() == "REJECTED"
            rows.append(r)
    return rows


def two_proportion_z(x1, n1, x2, n2):
    """How surprising is the difference between two rates?

    x1/n1 = recent, x2/n2 = baseline. Returns 0.0 when it cannot be computed
    (no data, or both rates identical) rather than raising - a detector should
    stay quiet on thin data, not crash the run.
    """
    if n1 == 0 or n2 == 0:
        return 0.0
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    if pooled in (0.0, 1.0):
        return 0.0
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return 0.0 if se == 0 else (p1 - p2) / se


def windows(rows):
    """Split history into (recent, baseline) by date."""
    last = max(date.fromisoformat(r["fill_date"]) for r in rows)
    recent_start = last - timedelta(days=RECENT_DAYS - 1)
    base_end = recent_start - timedelta(days=GAP_DAYS + 1)
    base_start = base_end - timedelta(days=BASELINE_DAYS - 1)

    recent, baseline = [], []
    for r in rows:
        d = date.fromisoformat(r["fill_date"])
        if d >= recent_start:
            recent.append(r)
        elif base_start <= d <= base_end:
            baseline.append(r)
    return recent, baseline, {"recent_from": recent_start.isoformat(),
                              "recent_to": last.isoformat(),
                              "baseline_from": base_start.isoformat(),
                              "baseline_to": base_end.isoformat()}


def _rate(rows, key):
    """{subject: (rejections, claims)} grouped by key(row)."""
    out = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        k = key(r)
        out[k][1] += 1
        out[k][0] += r["rejected"]
    return out


# ---------------------------------------------------------------------------
# DETECTOR 1 - a drug's rejection rate jumped
# ---------------------------------------------------------------------------
def detect_reject_rate(recent, baseline):
    findings = []
    now = _rate(recent, lambda r: r["drug"])
    then = _rate(baseline, lambda r: r["drug"])

    for drug, (x1, n1) in sorted(now.items()):
        if n1 < MIN_RECENT_CLAIMS:
            continue
        x2, n2 = then.get(drug, [0, 0])
        if n2 == 0:
            continue                     # brand new - detector 4 handles it
        p1, p2 = x1 / n1, x2 / n2
        z = two_proportion_z(x1, n1, x2, n2)
        if (p1 - p2) >= MIN_RATE_JUMP and z >= MIN_Z:
            # WHICH codes, not just how many. The first version omitted this,
            # and the explainer's top recommended action came back as "pull the
            # reject codes and tally them" - asking a human to go and find data
            # we were already holding.
            #
            # If you can answer a question deterministically, answer it before
            # you ask the model anything. Same rule as pre-fetching policy.
            codes = collections.Counter(
                r["reject_code"] for r in recent
                if r["drug"] == drug and r["rejected"])

            # CONCENTRATION - the share held by the single most common code.
            #
            # This is the difference between signal and noise, and it is
            # arithmetic, so it belongs here. A policy change rejects for ONE
            # reason: 34 of 34 on code 75. Random variation scatters: 4 on 79,
            # 2 on 75, 2 on 76, 1 on 88.
            #
            # Added because the explainer rated a known false positive
            # "urgency: high" and invented a separate plausible cause for each
            # stray code. Told that a finding is real and asked for causes, a
            # model will always supply them - so give it a number it can
            # reason AGAINST.
            top_share = (codes.most_common(1)[0][1] / sum(codes.values())
                         if codes else 0.0)
            findings.append({
                "kind": "reject_rate_spike",
                "subject": drug,
                "recent_rate": round(p1, 3), "baseline_rate": round(p2, 3),
                "recent_claims": n1, "baseline_claims": n2,
                "z": round(z, 1),
                "recent_reject_codes": dict(codes.most_common()),
                "top_code_share": round(top_share, 2),
                "headline": (f"{drug}: rejections went from {p2:.0%} to {p1:.0%} "
                             f"({x1} of {n1} claims)"),
            })
    return findings


# ---------------------------------------------------------------------------
# DETECTOR 2 - WHICH reject code changed. A rate jump says something broke;
# the code mix says what kind of thing broke.
# ---------------------------------------------------------------------------
def detect_code_mix(recent, baseline):
    findings = []
    now = collections.Counter(r["reject_code"] for r in recent if r["rejected"])
    then = collections.Counter(r["reject_code"] for r in baseline if r["rejected"])
    n1, n2 = sum(now.values()), sum(then.values())
    if n1 < MIN_RECENT_CLAIMS or n2 == 0:
        return findings

    for code, x1 in sorted(now.items()):
        x2 = then.get(code, 0)
        p1, p2 = x1 / n1, x2 / n2
        z = two_proportion_z(x1, n1, x2, n2)
        if (p1 - p2) >= MIN_RATE_JUMP and z >= MIN_Z:
            findings.append({
                "kind": "reject_code_mix_shift",
                "subject": f"reject code {code}",
                "recent_rate": round(p1, 3), "baseline_rate": round(p2, 3),
                "recent_claims": n1, "baseline_claims": n2,
                "z": round(z, 1),
                "headline": (f"Reject code {code} grew from {p2:.0%} to {p1:.0%} "
                             f"of all rejections"),
            })
    return findings


# ---------------------------------------------------------------------------
# DETECTOR 3 - a prescriber's VOLUME moved, measured against their own history
# ---------------------------------------------------------------------------
def detect_prescriber_volume(recent, baseline):
    findings = []
    now = collections.Counter(r["prescriber"] for r in recent)
    then = collections.Counter(r["prescriber"] for r in baseline)

    for who, n1 in sorted(now.items()):
        per_day_now = n1 / RECENT_DAYS
        per_day_then = then.get(who, 0) / BASELINE_DAYS
        if per_day_then == 0 or n1 < MIN_RECENT_CLAIMS:
            continue
        multiple = per_day_now / per_day_then
        if multiple >= MIN_VOLUME_MULTIPLE:
            findings.append({
                "kind": "prescriber_volume_shift",
                "subject": who,
                "recent_per_day": round(per_day_now, 1),
                "baseline_per_day": round(per_day_then, 1),
                "multiple": round(multiple, 1),
                "recent_claims": n1,
                "headline": (f"{who}: {per_day_now:.1f} claims/day vs "
                             f"{per_day_then:.1f} normally ({multiple:.1f}x)"),
            })
    return findings


# ---------------------------------------------------------------------------
# DETECTOR 4 - something appeared that has never been seen before
# ---------------------------------------------------------------------------
def detect_new_entities(recent, baseline):
    findings = []
    seen = {r["drug"] for r in baseline}
    now = _rate(recent, lambda r: r["drug"])

    for drug, (x1, n1) in sorted(now.items()):
        if drug in seen or n1 < 5:
            continue
        findings.append({
            "kind": "new_drug",
            "subject": drug,
            "recent_claims": n1, "recent_rate": round(x1 / n1, 3),
            "headline": (f"{drug}: {n1} claims, never dispensed in the "
                         f"baseline period ({x1} rejected)"),
        })
    return findings


DETECTORS = [detect_reject_rate, detect_code_mix,
             detect_prescriber_volume, detect_new_entities]


def detect_all(rows=None):
    """Every finding, plus the windows they were measured over."""
    rows = rows if rows is not None else load_history()
    recent, baseline, span = windows(rows)
    findings = []
    for fn in DETECTORS:
        findings.extend(fn(recent, baseline))
    return findings, span


if __name__ == "__main__":
    findings, span = detect_all()
    print(f"\nrecent   {span['recent_from']} .. {span['recent_to']}")
    print(f"baseline {span['baseline_from']} .. {span['baseline_to']}\n")
    if not findings:
        print("  nothing above threshold - a quiet week is a valid result\n")
    for f in findings:
        print(f"  [{f['kind']}]")
        print(f"    {f['headline']}")
        if "z" in f:
            print(f"    z = {f['z']}  (threshold {MIN_Z})")
        print()
