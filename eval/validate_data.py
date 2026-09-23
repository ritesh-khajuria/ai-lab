"""
PRE-FLIGHT DATA VALIDATION.  Ritesh's idea, and it is the right one.

This is NOT the eval harness. They do different jobs, and the difference is
worth being clear about:

    validate_data.py   Is the INPUT sane?      deterministic, free, runs first
    eval_retrieval.py  Is the AGENT any good?  statistical, needs scoring

A foreign-key check would have PREVENTED the mess we hit tonight: CLM2007-2009
were written with members M102/M103 that did not exist in the member table.
check_eligibility returned {"error": "no member M103"} and the agent then
handled that error INCONSISTENTLY - it submitted a prior authorization for
CLM2008 and refused for CLM2009. Same input, opposite decisions.

The eval harness would have DETECTED that. This file stops it happening.

Prevention is cheaper than detection. Run this BEFORE the agent, and a bad row
never reaches an LLM call you have to pay for and then argue about.

RUN:
    ../.venv/bin/python validate_data.py            # langgraph dataset
    ../.venv/bin/python validate_data.py --batch    # claims_agent dataset

Exit code 0 = clean, 1 = problems found. That is what makes it usable as an
Airflow task: the DAG stops instead of feeding garbage to the agent.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LG_DIR = os.path.join(HERE, "..", "claims_agent_langgraph")
BATCH_DIR = os.path.join(HERE, "..", "claims_agent")

# reject_code is deliberately NOT here: a PAID claim has none, and that is
# correct. It is required only when status is REJECTED - see check 5 below.
# (First version of this file got that wrong and failed every PAID claim.)
REQUIRED = ["claim_id", "member_id", "drug", "ndc", "quantity",
            "days_supply", "prescriber", "plan", "status"]

# Codes the agent knows how to explain. An unknown code is not fatal, but it
# means the agent will be reasoning without a definition - worth a warning.
KNOWN_CODES = {"70", "75", "76", "79", "88", ""}


def load_members(folder):
    """The member table the tools actually consult.

    It lives in tools.py as a dict rather than a CSV, so we import it - the
    validator must check against the SAME source of truth the agent uses, not a
    copy that can drift.
    """
    sys.path.insert(0, os.path.abspath(folder))
    for mod in ("tools", "claims_agent"):
        try:
            m = __import__(mod)
            if hasattr(m, "MEMBERS"):
                return set(m.MEMBERS)
        except Exception:
            continue
    return None


def check(claims_file, members, label):
    problems, warnings = [], []
    seen = set()

    with open(claims_file, "r") as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows, start=2):        # line 1 is the header
        cid = (row.get("claim_id") or "").strip()

        # 1. Structural: is every column present and non-empty?
        for col in REQUIRED:
            if not (row.get(col) or "").strip():
                problems.append(f"line {i} ({cid or '?'}): missing '{col}'")

        # 2. Primary key: claim_id must be unique.
        if cid in seen:
            problems.append(f"line {i}: duplicate claim_id {cid}")
        seen.add(cid)

        # 3. FOREIGN KEY - the check that would have caught tonight's bug.
        mid = (row.get("member_id") or "").strip()
        if members is not None and mid and mid not in members:
            problems.append(
                f"line {i} ({cid}): member_id {mid} does not exist "
                f"-> check_eligibility will error and the agent's behaviour "
                f"becomes unpredictable")

        # 4. Types: these are used in arithmetic downstream.
        for col in ("quantity", "days_supply"):
            val = (row.get(col) or "").strip()
            if val and not val.isdigit():
                problems.append(f"line {i} ({cid}): {col}={val!r} is not a number")

        # 5. Domain: a REJECTED claim with no reject code cannot be resolved.
        status = (row.get("status") or "").strip().upper()
        code = (row.get("reject_code") or "").strip()
        if status == "REJECTED" and not code:
            problems.append(f"line {i} ({cid}): REJECTED but no reject_code")
        if code and code not in KNOWN_CODES:
            warnings.append(f"line {i} ({cid}): reject_code {code} is not one "
                            f"the agent has a definition for")

    print(f"\n=== {label} ===")
    print(f"file    : {os.path.relpath(claims_file, HERE)}")
    print(f"rows    : {len(rows)}")
    print(f"members : {'not found - FK check skipped' if members is None else sorted(members)}")

    for w in warnings:
        print(f"  WARN  {w}")
    for p in problems:
        print(f"  FAIL  {p}")

    if not problems:
        print(f"  OK    {len(rows)} rows passed all checks")
    return len(problems)


if __name__ == "__main__":
    use_batch = "--batch" in sys.argv
    folder = BATCH_DIR if use_batch else LG_DIR
    claims_file = os.path.join(folder, "claims.csv" if use_batch else "claims_lg.csv")
    label = "claims_agent (batch)" if use_batch else "claims_agent_langgraph"

    failures = check(claims_file, load_members(folder), label)

    # Non-zero exit = an Airflow task turns red and the pipeline stops here,
    # instead of paying for LLM calls on data we already know is broken.
    print(f"\n{'FAILED' if failures else 'PASSED'} - {failures} problem(s)\n")
    sys.exit(1 if failures else 0)
