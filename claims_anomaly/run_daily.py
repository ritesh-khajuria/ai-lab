"""
CLAIMS ANOMALY AGENT - the daily run.

    detect (free, deterministic)  ->  explain (costs money)  ->  report + log

WHAT A HUMAN GETS
-----------------
A dated markdown report, ranked by urgency, that a pharmacy operations lead can
read in two minutes. Not a queue to approve - this agent has no authority to do
anything, so there is nothing to authorise. It reports; people decide.

That is a deliberate difference from the other two pipelines:

    claims_agent            drafts a resolution -> human approves the OUTPUT
    claims_agent_langgraph  submits a PA        -> human approves the ACTION
    this agent              explains a pattern  -> nobody approves anything

The review pattern follows the authority, and this agent has none. Adding an
approval step here would be ceremony.

INTEGRITY GATE
--------------
Checks the history file before spending anything on a model. A malformed date
or an unknown status silently distorts every window calculation downstream, and
a wrong number with a confident explanation attached is worse than no report.

    ../.venv/bin/python run_daily.py            # detect + report, NO model calls
    ../.venv/bin/python run_daily.py --explain  # adds the model layer
"""
import csv
import os
import sys
from datetime import date, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from detect import detect_all, load_history, HISTORY      # noqa: E402

REPORT_DIR = os.path.join(HERE, "reports")
FINDINGS_LOG = os.path.join(HERE, "findings_log.csv")
VALID_STATUS = {"PAID", "REJECTED"}

URGENCY_ORDER = {"high": 0, "medium": 1, "low": 2, None: 3}


def check_history(rows):
    """Fail loudly before the expensive part. Returns a list of problems."""
    problems = []
    if not rows:
        return ["claims_history.csv is empty"]

    for i, r in enumerate(rows, start=2):
        try:
            date.fromisoformat(r["fill_date"])
        except (ValueError, KeyError):
            problems.append(f"line {i}: unreadable fill_date {r.get('fill_date')!r}")
        status = (r.get("status") or "").strip().upper()
        if status not in VALID_STATUS:
            problems.append(f"line {i}: status {status!r} is not PAID or REJECTED")
        if status == "REJECTED" and not (r.get("reject_code") or "").strip():
            problems.append(f"line {i}: REJECTED with no reject_code")
        if len(problems) > 10:
            problems.append("... more suppressed")
            break
    return problems


def log_findings(findings, run_at):
    """APPEND-ONLY. Yesterday's findings are how you spot a pattern that has
    been building for a week, so they are never overwritten."""
    is_new = not os.path.exists(FINDINGS_LOG)
    with open(FINDINGS_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["run_at", "kind", "subject", "headline", "top_code_share"])
        for x in findings:
            w.writerow([run_at, x["kind"], x["subject"], x["headline"],
                        x.get("top_code_share", "")])


def write_report(findings, span, explanations, run_at):
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, f"{span['recent_to']}.md")

    def urgency_of(f):
        e = explanations.get(f["subject"], {})
        return URGENCY_ORDER.get(e.get("urgency"), 3)

    lines = [
        f"# Claims anomaly report — {span['recent_to']}",
        "",
        f"Recent window **{span['recent_from']} .. {span['recent_to']}** "
        f"compared against **{span['baseline_from']} .. {span['baseline_to']}**.",
        "",
        f"{len(findings)} finding(s). Generated {run_at}.",
        "",
        "> Detected statistically; explanations are hypotheses for a human to "
        "check, not conclusions.",
        "",
    ]

    for f in sorted(findings, key=urgency_of):
        e = explanations.get(f["subject"], {})
        urgency = e.get("urgency", "—")
        lines += [f"## {f['subject']}  ·  `{urgency}`", "",
                  f"**{f['headline']}**", ""]
        if "z" in f:
            lines.append(f"- z = {f['z']}, {f['recent_claims']} claims in window")
        if f.get("recent_reject_codes"):
            codes = ", ".join(f"`{c}` × {n}"
                              for c, n in f["recent_reject_codes"].items())
            lines.append(f"- reject codes: {codes} "
                         f"(top code holds {f.get('top_code_share', 0):.0%})")
        lines.append("")

        if e.get("summary"):
            lines += [e["summary"], ""]
        for c in e.get("likely_causes", [])[:4]:
            lines.append(f"- **{c['cause']}** _({c['likelihood']})_ — {c['why']}")
        if e.get("checks"):
            lines += ["", "**Check first:**"]
            lines += [f"{i}. {c}" for i, c in enumerate(e["checks"][:5], 1)]
        if e.get("policy_note"):
            lines += ["", f"> {e['policy_note']}"]
        lines += ["", "---", ""]

    lines.append("_Synthetic claims data. No PHI._")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


def main():
    rows = load_history()
    problems = check_history(rows)
    if problems:
        print(f"\n  {len(problems)} problem(s) in {HISTORY} — stopping before "
              f"any model call:")
        for p in problems:
            print(f"    {p}")
        return 1

    findings, span = detect_all(rows)
    run_at = datetime.now().isoformat(timespec="seconds")
    print(f"\n{len(findings)} finding(s)  "
          f"{span['recent_from']}..{span['recent_to']}")
    for f in findings:
        print(f"  [{f['kind']}] {f['headline']}")

    explanations = {}
    if "--explain" in sys.argv and findings:
        import anthropic
        from dotenv import load_dotenv
        import explain as explainer

        load_dotenv(os.path.join(os.path.dirname(HERE), ".env"))
        load_dotenv()
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        print(f"\nexplaining ({len(findings)} model calls)...")
        for f in findings:
            out = explainer.explain_one(client, f)
            if "error" in out:
                print(f"  {f['subject']}: {out['error']}")
            explanations[f["subject"]] = out
    elif not explanations:
        print("\n(no --explain: report will carry the numbers without "
              "hypotheses)")

    log_findings(findings, run_at)
    path = write_report(findings, span, explanations, run_at)
    print(f"\nreport  {path}")
    print(f"log     {FINDINGS_LOG}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
