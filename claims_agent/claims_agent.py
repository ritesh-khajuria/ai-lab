"""
Phase 3 - Your first AGENT: a Pharmacy Claims Rejection Resolver.

WHAT AN AGENT IS (in one loop):
    ask the model -> it may ask to call a tool -> we run the tool ->
    hand back the result -> repeat -> until it has enough to answer.

Everything here uses MOCK data (fake but realistic). No real PHI, no real
systems - safe to run, safe to demo to anyone.

Run from the ai-lab folder:
    .venv/bin/python phase3_claims_agent.py
"""
import os
import csv
import json
from dotenv import load_dotenv
import anthropic

# EDIT 1 of 4 - bring in the policy search we built and tested separately.
# NOTE: importing this loads the embedding model, so the first start is a few
# seconds slower. That happens ONCE, not per question.
from claims_agent_policy_search import search_policy

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "claude-sonnet-4-6"

# ============================================================================
# 1) MOCK DATA - stands in for Cigna-style systems. 100% fake.
# ============================================================================
REJECT_CODES = {
    "70": "Product/Service not covered by the plan.",
    "75": "Prior Authorization (PA) required before the drug is covered.",
    "76": "Plan limitation exceeded (e.g., quantity or days-supply limit).",
    "79": "Refill too soon - filled earlier than the plan allows.",
    "88": "DUR reject - drug utilization review flagged a safety concern.",
}

# Anchor every path to THIS file's folder, so the script works
# no matter which directory it is run from (Airflow runs it from elsewhere).
HERE = os.path.dirname(os.path.abspath(__file__))
CLAIMS_FILE = os.path.join(HERE, "claims.csv")


def load_claims():
    """Read the claims 'system' from claims.csv, keyed by claim_id.

    In production this is a query against the claims database. The SHAPE of
    what comes back is identical - which is why nothing downstream changes.
    """
    claims = {}
    with open(CLAIMS_FILE, "r") as f:
        for row in csv.DictReader(f):
            # CSV hands back only text. Convert numbers at the boundary.
            row["quantity"] = int(row["quantity"])
            row["days_supply"] = int(row["days_supply"])
            claims[row["claim_id"]] = row
    return claims


CLAIMS = load_claims()          # loaded once, when this module is imported

MEMBERS = {
    "M100": {"member_id": "M100", "name": "Jordan Lee", "plan": "PLAN-A",
             "eligibility": "ACTIVE", "coverage_start": "202-21-21"},
    "M101": {"member_id": "M101", "name": "Priya Nair", "plan": "PLAN-A",
             "eligibility": "ACTIVE", "coverage_start": "2025-06-01"},
}

# ============================================================================
# 2) THE TOOLS - your first REAL `def` functions. Each returns fake-but-real
#    data. (Remember the "your first function" preview in CODE_WALKTHROUGH.md?
#    Here they are - three of them.)
# ============================================================================
def get_claim(claim_id):
    """Look up a pharmacy claim by its ID."""
    return CLAIMS.get(claim_id, {"error": f"no claim {claim_id}"})

def lookup_reject_code(code):
    """Translate an NCPDP reject code into plain English."""
    return {"code": code, "meaning": REJECT_CODES.get(code, "unknown code")}

def check_eligibility(member_id):
    """Check whether a member is currently active on their plan."""
    return MEMBERS.get(member_id, {"error": f"no member {member_id}"})

# Map each tool NAME (what Claude will ask for) to the REAL function.
TOOLS_BY_NAME = {
    "get_claim": get_claim,
    "lookup_reject_code": lookup_reject_code,
    "check_eligibility": check_eligibility,
    "search_policy": search_policy,        # EDIT 2 of 4 - name -> function
}

def run_tool(name, args):
    """Execute one tool Claude asked for, and return its result as text."""
    result = TOOLS_BY_NAME[name](**args)   # **args unpacks e.g. {"claim_id": "CLM1001"}
    return json.dumps(result)              # tool results are sent back as text

# ============================================================================
# 3) TOOL SCHEMAS - how we DESCRIBE the tools to Claude so it knows they exist
#    and what inputs each one needs. Claude reads these to decide what to call.
# ============================================================================
TOOLS = [
    {
        "name": "get_claim",
        "description": "Get the full details of a pharmacy claim by its claim_id.",
        "input_schema": {
            "type": "object",
            "properties": {"claim_id": {"type": "string"}},
            "required": ["claim_id"],
        },
    },
    {
        "name": "lookup_reject_code",
        "description": "Explain what an NCPDP claim reject code means.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
    {
        "name": "check_eligibility",
        "description": "Check if a member is active/eligible on their plan.",
        "input_schema": {
            "type": "object",
            "properties": {"member_id": {"type": "string"}},
            "required": ["member_id"],
        },
    },
    # EDIT 3 of 4 - describe the new tool so Claude knows it exists.
    # The `description` is the ONLY thing telling Claude when to reach for it.
    # The last sentence is what makes it choose between claim data and policy.
    {
        "name": "search_policy",
        "description": (
            "Search PLAN-A policy documents for prior authorization criteria, "
            "refill and refill-too-soon rules, DUR review procedures, formulary "
            "tiers and exclusions, and the appeals process. "
            "Use this whenever the answer depends on what the PLAN RULES say, "
            "rather than on the facts of a specific claim."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "A natural-language question about plan policy.",
                }
            },
            "required": ["question"],
        },
    },
]

SYSTEM_PROMPT = """You are a pharmacy claims resolution assistant for a health plan.
Given a rejected claim, use the tools to gather facts (never guess), then write a
short, structured draft for a human claims specialist to review:

What happened: (one line)
Why it rejected: (plain English, based on the reject code)
Recommended fix: (the concrete next step)

Only use information the tools return.

EDIT 4 of 4 - policy guidance:
Use search_policy whenever the answer depends on plan rules rather than the
facts of the claim. When you use retrieved policy text, cite the source
document and version, e.g. (plan_a_refill_policy.md, v2026.02). If the policy
search returns nothing relevant, say so - do not fill the gap from memory."""

# ============================================================================
# 4) THE AGENT LOOP - the heart of it. Ask -> maybe call tools -> repeat.
#    This is what makes it an "agent" instead of a one-shot chatbot.
# ============================================================================
def resolve_claim(user_request, max_steps=6):
    # The running conversation. We keep adding to it each step.
    messages = [{"role": "user", "content": user_request}]

    final_text = ""                      # remember the last thing it said

    for step in range(max_steps):        # max_steps is a safety cap (no runaway loops)
        response = client.messages.create(
            model=MODEL, max_tokens=800,
            system=SYSTEM_PROMPT,
            tools=TOOLS,                 # <- telling Claude which tools exist
            messages=messages,
        )

        # Print anything the agent 'says' as it reasons toward the answer.
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[agent] {block.text.strip()}")
                final_text = block.text.strip()

        # If it stopped naturally, it's done - the text above was the final answer.
        # RETURN it (don't just print) so a pipeline can store it.
        if response.stop_reason == "end_turn":
            return final_text

        # Otherwise it asked to call one or more tools. Run each one and reply.
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":     # the model's request to call a tool
                print(f"   -> calling tool: {block.name}({block.input})")
                output = run_tool(block.name, block.input)   # we run the real function
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,   # ties the result back to the request
                    "content": output,
                })
        messages.append({"role": "user", "content": tool_results})
        # ...loop again: now Claude sees the tool results and continues.

    print("\n[stopped: hit the max-steps safety cap]")

# ============================================================================
# 5) RUN IT - watch the agent investigate a rejected claim on its own.
#    Try changing CLM1001 to CLM1002 (a different reject reason) and re-run.
# ============================================================================
if __name__ == "__main__":
    #for claim_id in CLAIMS:
        print("===== Claims Rejection Resolver (mock data) =====")
        resolve_claim("Claim CLM1002 was rejected. Explain why and how to fix it.")
