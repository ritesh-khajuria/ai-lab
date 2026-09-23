"""
CLAIMS ANOMALY AGENT - build the synthetic claims history.

This agent works on a POPULATION, not one claim. So it needs history: enough
days of normal behaviour that abnormal behaviour is visible against it.

WHY GENERATE RATHER THAN HAND-WRITE
-----------------------------------
Because we need to know the right answer. Every anomaly below is PLANTED at a
known day, with known size. That is what makes the detectors testable: a test
can assert "day 52 must be flagged" instead of squinting at output.

Hand-written fixtures cannot do that at 60 days x ~60 claims.

SEEDED, so it is reproducible. Same seed -> byte-identical file -> a detector
change is the only thing that can move the numbers.

    ../.venv/bin/python make_history.py

SYNTHETIC. No real members, prescribers, or claims.
"""
import csv
import os
import random
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "claims_history.csv")

SEED = 20260913
DAYS = 60
END = date(2026, 9, 12)              # last day in the file
START = END - timedelta(days=DAYS - 1)

# --- the normal world ------------------------------------------------------
# Each drug carries its BASELINE reject rate. A claim either pays or rejects
# with one of the codes below, at roughly these odds.
DRUGS = [
    # drug,                        tier, baseline reject rate
    ("Atorvastatin 20 mg tab",      1, 0.04),
    ("Metformin 500 mg tab",        1, 0.04),
    ("Lisinopril 10 mg tab",        1, 0.03),
    ("Amoxicillin 500 mg cap",      1, 0.05),
    ("Warfarin 5 mg tab",           2, 0.08),
    ("Insulin glargine 100 units/mL", 3, 0.10),
    ("Oxycodone 5 mg tab",          2, 0.12),
    ("Ozempic 1 mg pen",            5, 0.15),
    ("Wegovy 0.5 mg pen",           5, 0.15),
    ("Mounjaro 5 mg pen",           5, 0.15),
]
GLP1 = {"Ozempic 1 mg pen", "Wegovy 0.5 mg pen", "Mounjaro 5 mg pen"}

PRESCRIBERS = ["Dr. A. Rao", "Dr. B. Shah", "Dr. C. Iyer",
               "Dr. D. Nair", "Dr. E. Menon"]

# How a rejection splits across codes, normally.
REJECT_MIX = [("79", 0.40), ("76", 0.25), ("88", 0.20), ("75", 0.15)]

# --- what we are planting --------------------------------------------------
# Written down here, in one place, so the tests can import it and assert
# against the SAME numbers the generator used. A fixture whose expected values
# live in two places drifts.
PLANTED = {
    # A plan policy change on day 52: GLP-1 drugs start demanding prior auth,
    # so reject code 75 jumps for exactly those drugs. This is the realistic
    # one - nobody tells the pharmacy team, they just see rejections.
    "glp1_pa_spike": {"from_day": 52, "drugs": GLP1, "reject_rate": 0.62,
                      "code": "75"},

    # A prescriber whose volume triples - could be a locum, could
    # be a data problem, could be fraud. The agent should not guess which.
    "prescriber_volume": {"from_day": 53, "prescriber": "Dr. E. Menon",
                          "multiplier": 3},

    # A drug that has never appeared before shows up on day 57.
    "new_drug": {"from_day": 57, "drug": "Zepbound 2.5 mg pen", "tier": 5,
                 "reject_rate": 0.55, "code": "75", "per_day": 4},
}


def reject_code(rng):
    roll = rng.random()
    running = 0.0
    for code, share in REJECT_MIX:
        running += share
        if roll <= running:
            return code
    return "79"


def build():
    rng = random.Random(SEED)
    rows = []
    claim_no = 50000

    for day_index in range(DAYS):
        day = START + timedelta(days=day_index)
        # Weekends are quieter. Without this, a weekday/weekend wobble would
        # look like an anomaly and the detectors would cry wolf every Saturday.
        weekday = day.weekday()
        base_volume = 34 if weekday >= 5 else 62
        volume = base_volume + rng.randint(-6, 6)

        for _ in range(volume):
            drug, tier, base_rate = rng.choice(DRUGS)
            prescriber = rng.choice(PRESCRIBERS)

            # planted: prescriber volume surge.
            #
            # DERIVE the reassignment probability from the multiplier, do not
            # hand-pick it. The first version used a hard-coded 0.18 while the
            # answer key above said "multiplier: 3" - they disagreed, the real
            # surge came out at 1.47x, and the detector correctly ignored it.
            # The fixture was wrong, not the detector.
            #
            # With P prescribers each holding share s = 1/P, reassigning a
            # fraction p of the rest gives share = s + p(1 - s). Setting that
            # equal to s*M and solving:   p = s(M - 1) / (1 - s)
            surge = PLANTED["prescriber_volume"]
            if day_index >= surge["from_day"]:
                s = 1 / len(PRESCRIBERS)
                p = s * (surge["multiplier"] - 1) / (1 - s)
                if rng.random() < p:
                    prescriber = surge["prescriber"]

            # planted: GLP-1 prior-auth policy change
            spike = PLANTED["glp1_pa_spike"]
            if day_index >= spike["from_day"] and drug in spike["drugs"]:
                rate, forced = spike["reject_rate"], spike["code"]
            else:
                rate, forced = base_rate, None

            if rng.random() < rate:
                status = "REJECTED"
                code = forced or reject_code(rng)
            else:
                status, code = "PAID", ""

            claim_no += 1
            rows.append([f"CLH{claim_no}", day.isoformat(), drug, tier,
                         prescriber, status, code])

        # planted: a drug nobody has dispensed before
        new = PLANTED["new_drug"]
        if day_index >= new["from_day"]:
            for _ in range(new["per_day"]):
                claim_no += 1
                rejected = rng.random() < new["reject_rate"]
                rows.append([f"CLH{claim_no}", day.isoformat(), new["drug"],
                             new["tier"], rng.choice(PRESCRIBERS),
                             "REJECTED" if rejected else "PAID",
                             new["code"] if rejected else ""])

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "fill_date", "drug", "tier",
                    "prescriber", "status", "reject_code"])
        w.writerows(rows)

    return rows


if __name__ == "__main__":
    rows = build()
    rejected = sum(1 for r in rows if r[5] == "REJECTED")
    print(f"wrote {OUT}")
    print(f"  {len(rows)} claims over {DAYS} days "
          f"({START} .. {END})")
    print(f"  {rejected} rejected ({rejected/len(rows):.1%} overall)")
    print("\n  planted anomalies (the answer key):")
    for name, spec in PLANTED.items():
        day = (START + timedelta(days=spec["from_day"])).isoformat()
        print(f"    {name:<18} from {day} (day {spec['from_day']})")
