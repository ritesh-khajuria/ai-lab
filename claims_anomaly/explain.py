"""
CLAIMS ANOMALY AGENT - EXPLANATION.  This is where the model earns its place.

WHAT IT IS AND IS NOT FOR
-------------------------
detect.py already established, by arithmetic, that something changed. The model
is never asked whether the change is real - it is told, and its job starts
after that:

    what could plausibly cause this, and what should a person check first?

That is judgement over context, which is the thing a model is genuinely good
at. Detecting a rate move is not.

THREE GUARDRAILS, AND WHY
-------------------------
1. IT NEVER RE-DECIDES THE NUMBERS. The metrics are given as facts. If the
   model could disagree with the arithmetic, you would have two sources of
   truth and no way to tell which was right.

2. IT MUST NOT ACCUSE ANYONE. A prescriber whose volume tripled might be
   covering for a colleague on leave, or a data-feed duplicate, or a clinic
   merger - or fraud. Only one of those is an accusation, and a system that
   reaches for it first is a system that will eventually be very wrong about a
   real person. The prompt requires the benign explanations first and forbids
   a conclusion.

3. IT ONLY CITES POLICY IT WAS GIVEN. Policy context is pre-fetched
   deterministically (see policy_search.py) and passed in. The model is told to
   cite only from that, and to say so when nothing was supplied.

STRUCTURED OUTPUT
-----------------
JSON, not prose, so a downstream report or queue can consume it - and so a
malformed answer is a caught error rather than a silently odd paragraph.

    ../.venv/bin/python explain.py           # dry run - shows the findings only
    ../.venv/bin/python explain.py --run     # calls the model (costs money)
"""
import json
import os
import sys

import anthropic
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from detect import detect_all                    # noqa: E402
from policy_search import context_for            # noqa: E402

load_dotenv(os.path.join(os.path.dirname(HERE), ".env"))
load_dotenv()
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a pharmacy claims data-quality analyst.

You are given an anomaly that has ALREADY been detected and verified
statistically. Do not re-evaluate whether it is real - it is. Your job is to
explain what could cause it and what a human should check.

RULES, which matter more than being helpful:

1. Never accuse anyone of anything. For a change in a prescriber's volume, the
   ordinary explanations - a colleague on leave, a clinic merger, a duplicated
   data feed, a seasonal clinic - come FIRST and are usually correct. You may
   note that misuse cannot be ruled out without review, but you must not
   conclude it, imply it, or rank it first.

2. Cite plan policy ONLY from the policy excerpts you are given. If none were
   supplied, say that no policy was retrieved rather than recalling anything
   from memory.

3. Distinguish what the DATA SHOWS from what it MIGHT MEAN. Never present a
   hypothesis as established.

4. Prefer the boring explanation. Most anomalies in claims data are policy
   changes, feed problems, or calendar effects.

5. ORDINARY VARIATION IS A VALID ANSWER, and often the right one. Look at
   "top_code_share" when it is given - the share of rejections held by a single
   reject code:

     high (0.8-1.0)  every rejection is for the SAME reason. That is a
                     systematic cause: a policy change, an edit, a feed.
     low  (below ~0.5)  rejections are scattered across unrelated reasons.
                     Nothing systematic rejects claims for four different
                     reasons at once. This is usually a small sample moving
                     around, and it crossed the threshold by luck.

   When the pattern is scattered, SAY SO as your first cause, set urgency to
   "low", and do not invent a separate mechanism for each stray code. A list of
   plausible-sounding causes for random noise is worse than saying "this looks
   like normal variation" - it sends people to investigate nothing.

Reply with ONLY a JSON object, no markdown fence:

{
  "summary": "one sentence a pharmacy ops lead can read in five seconds",
  "likely_causes": [
    {"cause": "...", "why": "what in the data points here", "likelihood": "high|medium|low"}
  ],
  "checks": ["the first thing a human should verify", "..."],
  "policy_note": "what the supplied policy says, with (source, version) - or null",
  "urgency": "high|medium|low"
}"""


def build_prompt(finding, policy):
    facts = {k: v for k, v in finding.items() if k != "headline"}
    lines = [
        f"ANOMALY: {finding['headline']}",
        "",
        "Verified metrics (these are facts, do not dispute them):",
        json.dumps(facts, indent=2),
        "",
    ]
    if policy:
        lines.append("Plan policy excerpts retrieved for this finding:")
        for p in policy:
            lines.append(f"\n--- {p['source']} (v{str(p['version']).split('|')[0].strip()}) ---")
            lines.append(p["text"][:1200])
    else:
        lines.append("No plan policy was retrieved for this finding "
                     "(it is not a policy question).")
    return "\n".join(lines)


def explain_one(client, finding):
    policy = context_for(finding)
    msg = client.messages.create(
        # 2000, not 900. The first version truncated 5 of 7 answers mid-object:
        # the JSON was being written correctly and simply ran out of room, so
        # every one failed to parse. max_tokens caps the OUTPUT - structured
        # output needs headroom, and the failure looks like a model problem
        # when it is a budget problem.
        model=MODEL, max_tokens=2000, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(finding, policy)}],
    )
    raw = "".join(b.text for b in msg.content if b.type == "text").strip()

    # Distinguish the two failures, because the fixes are completely different:
    #   ran out of room   -> raise max_tokens
    #   returned prose    -> fix the prompt
    # Reporting both as "invalid JSON" sent me looking in the wrong place once
    # already.
    if msg.stop_reason == "max_tokens":
        return {"error": "answer was cut off - max_tokens too low",
                "raw": raw[-300:]}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "model returned something that is not JSON",
                "raw": raw[:600]}

    missing = [k for k in ("summary", "likely_causes", "checks", "urgency")
               if k not in parsed]
    if missing:
        return {"error": f"missing keys: {missing}", "raw": raw[:600]}

    parsed["policy_sources"] = [p["source"] for p in policy]
    return parsed


def main():
    findings, span = detect_all()
    print(f"\n{len(findings)} finding(s), "
          f"{span['recent_from']}..{span['recent_to']}")
    for f in findings:
        print(f"  [{f['kind']}] {f['headline']}")

    if "--run" not in sys.argv:
        print(f"\nDry run. Add --run to explain these "
              f"({len(findings)} model calls, costs money).\n")
        return 0

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    print()
    for f in findings:
        out = explain_one(client, f)
        print("=" * 74)
        print(f"{f['kind']}  ·  {f['subject']}")
        print("=" * 74)
        if "error" in out:
            print(f"  MALFORMED: {out['error']}\n  {out['raw'][:200]}\n")
            continue
        print(f"  urgency : {out['urgency']}")
        print(f"  summary : {out['summary']}\n")
        for c in out["likely_causes"]:
            print(f"   ({c['likelihood']:<6}) {c['cause']}")
            print(f"            {c['why']}")
        print("\n  check:")
        for c in out["checks"]:
            print(f"    - {c}")
        if out.get("policy_note"):
            print(f"\n  policy: {out['policy_note']}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
