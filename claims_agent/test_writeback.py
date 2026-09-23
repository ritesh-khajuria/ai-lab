"""
TEST for claims_writeback.py  -  YOU write the module, this checks it.

    cd ai-lab/claims_agent
    ../.venv/bin/python test_writeback.py

Spec: ../../SPEC_LEVEL1_WRITEBACK.md

This is your first real test file, so read how it works - it is not magic:

    assert <something that must be true>, "message if it isn't"

`assert` raises AssertionError when the condition is False. That's the whole
mechanism. pytest adds discovery and nicer output on top, but the check itself
is this one keyword.

Each test builds its OWN throwaway claims file in a temp directory, so this can
never touch your real claims.csv. That matters: a test that mutates production
data is worse than no test.
"""
import csv
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

FIXTURE = """\
claim_id,member_id,drug,ndc,quantity,days_supply,prescriber,plan,status,reject_code
CLM9001,M100,Drug A 10 mg tab,00000-0000-01,30,30,Dr. X,PLAN-A,REJECTED,76
CLM9002,M101,Drug B 5 mg cap,00000-0000-02,60,30,Dr. Y,PLAN-A,PAID,
CLM9003,M100,Drug C 20 mg tab,00000-0000-03,30,30,Dr. Z,PLAN-A,REJECTED,79
"""


def fresh_file(tmpdir):
    """A clean copy of the fixture for one test."""
    path = os.path.join(tmpdir, "claims.csv")
    with open(path, "w") as f:
        f.write(FIXTURE)
    return path


def rows_of(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def row_for(path, claim_id):
    for r in rows_of(path):
        if r["claim_id"] == claim_id:
            return r
    return None


# ---------------------------------------------------------------------------
# the tests
# ---------------------------------------------------------------------------
def test_happy_path(wb, tmp):
    path = fresh_file(tmp)
    ok, msg = wb.write_back("CLM9001", "r.khajuria", "2026-08-21T10:00:00",
                            claims_file=path)
    assert ok is True, f"expected ok=True, got {ok!r} ({msg})"

    row = row_for(path, "CLM9001")
    assert row["status"] == "RESOLUTION_APPROVED", \
        f"status should be RESOLUTION_APPROVED, got {row['status']!r}"
    assert row.get("resolved_by") == "r.khajuria", \
        f"resolved_by should be r.khajuria, got {row.get('resolved_by')!r}"
    assert row.get("resolved_at") == "2026-08-21T10:00:00", \
        f"resolved_at not recorded, got {row.get('resolved_at')!r}"


def test_other_rows_untouched(wb, tmp):
    path = fresh_file(tmp)
    before = rows_of(path)
    wb.write_back("CLM9001", "r.khajuria", "2026-08-21T10:00:00", claims_file=path)
    after = {r["claim_id"]: r for r in rows_of(path)}

    for original in before:
        cid = original["claim_id"]
        if cid == "CLM9001":
            continue
        for field, value in original.items():
            assert after[cid][field] == value, \
                f"{cid}.{field} changed from {value!r} to {after[cid][field]!r}"


def test_row_count_survives(wb, tmp):
    path = fresh_file(tmp)
    wb.write_back("CLM9001", "r.khajuria", "2026-08-21T10:00:00", claims_file=path)
    n = len(rows_of(path))
    assert n == 3, f"started with 3 rows, now {n} - the file was damaged"


def test_missing_claim(wb, tmp):
    path = fresh_file(tmp)
    ok, msg = wb.write_back("CLM9999", "r.khajuria", "2026-08-21T10:00:00",
                            claims_file=path)
    assert ok is False, "a claim that does not exist must not succeed"
    assert "not found" in msg.lower(), f"say what went wrong; got {msg!r}"
    assert len(rows_of(path)) == 3, "must not add a row for an unknown claim"


def test_paid_claim_is_protected(wb, tmp):
    path = fresh_file(tmp)
    ok, msg = wb.write_back("CLM9002", "r.khajuria", "2026-08-21T10:00:00",
                            claims_file=path)
    assert ok is False, "a PAID claim must never be written back"
    assert "not rejected" in msg.lower(), f"explain why; got {msg!r}"
    assert row_for(path, "CLM9002")["status"] == "PAID", "PAID was overwritten"


def test_idempotent(wb, tmp):
    path = fresh_file(tmp)
    wb.write_back("CLM9001", "r.khajuria", "2026-08-21T10:00:00", claims_file=path)
    ok, msg = wb.write_back("CLM9001", "someone.else", "2026-08-22T09:00:00",
                            claims_file=path)
    assert ok is False, "second write-back must be refused"
    assert "already" in msg.lower(), f"say it was already done; got {msg!r}"

    row = row_for(path, "CLM9001")
    assert row["resolved_by"] == "r.khajuria", \
        "the FIRST reviewer must stand - history is not overwritten"


def test_untouched_rejected_claim(wb, tmp):
    path = fresh_file(tmp)
    wb.write_back("CLM9001", "r.khajuria", "2026-08-21T10:00:00", claims_file=path)
    row = row_for(path, "CLM9003")
    assert row["status"] == "REJECTED", "CLM9003 was not decided - leave it alone"
    assert not (row.get("resolved_by") or ""), \
        "CLM9003 has no reviewer; resolved_by must stay empty"


TESTS = [test_happy_path, test_other_rows_untouched, test_row_count_survives,
         test_missing_claim, test_paid_claim_is_protected, test_idempotent,
         test_untouched_rejected_claim]


def main():
    try:
        import claims_writeback as wb
    except ImportError:
        print("\n  claims_writeback.py does not exist yet.")
        print("  Create it in this folder - see SPEC_LEVEL1_WRITEBACK.md\n")
        return 1

    if not hasattr(wb, "write_back"):
        print("\n  claims_writeback.py exists but has no write_back() function.\n")
        return 1

    passed = failed = 0
    tmp = tempfile.mkdtemp(prefix="wb_test_")
    try:
        for test in TESTS:
            name = test.__name__.replace("test_", "").replace("_", " ")
            try:
                test(wb, tmp)
            except AssertionError as e:
                print(f"  FAIL  {name}\n          {e}")
                failed += 1
            except Exception as e:
                print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
                failed += 1
            else:
                print(f"  pass  {name}")
                passed += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{passed} passed, {failed} failed\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
