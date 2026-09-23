"""
LangGraph version - THE TOOLS.

Four READ-ONLY tools (same as the hand-written agent), plus one that is
different in kind:

    submit_prior_auth()  ACTUALLY DOES SOMETHING.

That one tool is the whole reason this version exists. A tool that only reads
can be reviewed afterwards. A tool that ACTS must be approved BEFORE it runs -
you cannot un-submit a prior authorization.

LangChain's @tool decorator turns a normal Python function into something the
model can call. The DOCSTRING becomes the description Claude reads to decide
when to use it - so the docstring is not a comment here, it is functional.
"""
import csv
import os
from datetime import datetime

from langchain_core.tools import tool
from langgraph.types import interrupt

HERE = os.path.dirname(os.path.abspath(__file__))

# CLAIMS: this pipeline has its OWN claims file (CLM2xxx) so the two demos
#         never touch each other's data.
# POLICY: still shared. One ingestion pipeline builds the index; many
#         applications read it. You would never have each app maintain its own
#         copy of the same policy corpus - they would drift.
SHARED = os.path.join(os.path.dirname(HERE), "claims_agent")
CLAIMS_FILE = os.path.join(HERE, "claims_lg.csv")   # OWN claims - no overlap with claims_agent/
POLICY_INDEX = os.path.join(SHARED, "policy_index")
SUBMISSIONS_LOG = os.path.join(HERE, "pa_submissions.csv")

REJECT_CODES = {
    "70": "Product/Service not covered by the plan.",
    "75": "Prior Authorization (PA) required before the drug is covered.",
    "76": "Plan limitation exceeded (e.g., quantity or days-supply limit).",
    "79": "Refill too soon - filled earlier than the plan allows.",
    "88": "DUR reject - drug utilization review flagged a safety concern.",
}

MEMBERS = {
    "M100": {"member_id": "M100", "name": "Jordan Lee", "plan": "PLAN-A",
             "eligibility": "ACTIVE", "coverage_start": "2026-01-01"},
    "M101": {"member_id": "M101", "name": "Priya Nair", "plan": "PLAN-A",
             "eligibility": "ACTIVE", "coverage_start": "2025-06-01"},
    # Added after CLM2007-2009 were written against members that did not exist
    # here. check_eligibility returned {"error": "no member M103"} and the agent
    # handled it INCONSISTENTLY - see the note in README.md.
    "M102": {"member_id": "M102", "name": "Sam Ortiz", "plan": "PLAN-A",
             "eligibility": "ACTIVE", "coverage_start": "2026-02-01"},
    "M103": {"member_id": "M103", "name": "Dana Whitfield", "plan": "PLAN-A",
             "eligibility": "ACTIVE", "coverage_start": "2025-11-01"},
}


def load_claims():
    """Read claims_lg.csv FRESH. Call this per request - do not cache the result.

    A long-running server that snapshots this at import time goes stale the
    moment a claim is added, and every new claim 404s. That exact bug was found
    on the other API (CLM1007 was invisible until restart), and it was sitting
    here too: /pending iterated a dict loaded once at startup, so CLM2008 did
    not appear even though its graph really was paused and waiting.
    """
    claims = {}
    with open(CLAIMS_FILE, "r") as f:
        for row in csv.DictReader(f):
            row["quantity"] = int(row["quantity"])
            row["days_supply"] = int(row["days_supply"])
            claims[row["claim_id"]] = row
    return claims


# Snapshot for the TOOLS, which run inside a short-lived batch process where a
# stale read is not possible. Servers must call load_claims() instead.
CLAIMS = load_claims()


# ===========================================================================
# READ-ONLY TOOLS - safe to run without asking anyone.
# ===========================================================================
@tool
def get_claim(claim_id: str) -> dict:
    """Get the full details of a pharmacy claim by its claim_id."""
    return CLAIMS.get(claim_id, {"error": f"no claim {claim_id}"})


@tool
def lookup_reject_code(code: str) -> dict:
    """Explain what an NCPDP claim reject code means."""
    return {"code": code, "meaning": REJECT_CODES.get(code, "unknown code")}


@tool
def check_eligibility(member_id: str) -> dict:
    """Check if a member is active and eligible on their plan."""
    return MEMBERS.get(member_id, {"error": f"no member {member_id}"})


@tool
def search_policy(question: str) -> list:
    """Search PLAN-A policy documents for prior authorization criteria, refill
    rules, DUR review procedures, formulary tiers, and the appeals process.
    Use this whenever the answer depends on what the PLAN RULES say rather than
    on the facts of a specific claim."""
    # Imported lazily so the module loads fast when policy search is not needed.
    from sentence_transformers import SentenceTransformer
    import chromadb

    from query_expansion import expand

    # EXPAND FIRST - the model writes this question and is unreliable about drug
    # names. "Is Trulicity covered?" misses the right document entirely: the
    # corpus never says "Trulicity", it says GLP-1. We know the drug, so the
    # class gets looked up in code rather than recalled by the model.
    question, added = expand(question)
    if added:
        print(f"[policy-search] expanded query with: {', '.join(added)}")

    global _embedder, _collection
    try:
        _embedder
    except NameError:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        _collection = chromadb.PersistentClient(
            path=POLICY_INDEX).get_collection("plan_a_policies")

    hits = _collection.query(
        query_embeddings=[_embedder.encode(question).tolist()],
        n_results=3, include=["documents", "metadatas", "distances"])

    return [{"text": hits["documents"][0][i],
             "source": hits["metadatas"][0][i]["source"],
             "version": hits["metadatas"][0][i]["version"],
             "relevance": round(1 - hits["distances"][0][i], 3)}
            for i in range(len(hits["ids"][0]))]


# ===========================================================================
# THE ACTION TOOL - this one changes the world, so it STOPS AND ASKS.
# ===========================================================================
@tool
def submit_prior_auth(claim_id: str, drug: str, justification: str) -> str:
    """Submit a prior authorization request to the plan for a rejected claim.

    THIS PERFORMS A REAL SUBMISSION and cannot be undone. Only call it once you
    have confirmed, using the policy documents, that the member plausibly meets
    the approval criteria. Provide a clear clinical justification."""

    # ---------------------------------------------------------------------
    # interrupt() PAUSES THE ENTIRE GRAPH right here.
    #
    # LangGraph saves the whole conversation state to the checkpointer, and the
    # call returns to the caller. Nothing below this line runs yet.
    #
    # When a human later resumes with Command(resume={...}), execution starts
    # again AT THIS LINE and `decision` receives whatever they sent.
    # ---------------------------------------------------------------------
    decision = interrupt({
        "action": "submit_prior_auth",
        "claim_id": claim_id,
        "drug": drug,
        "justification": justification,
        "prompt": "A prior authorization is about to be SUBMITTED. Approve?",
    })

    reviewer = decision.get("reviewer", "unknown")

    if not decision.get("approved"):
        reason = decision.get("reason", "no reason given")
        return (f"SUBMISSION BLOCKED by {reviewer}. Reason: {reason}. "
                f"No prior authorization was submitted for {claim_id}.")

    # Only now does the side effect actually happen.
    reference = f"PA-{claim_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    with open(SUBMISSIONS_LOG, "a") as f:
        f.write(f"{reference}|{claim_id}|{drug}|{reviewer}|"
                f"{datetime.now().isoformat(timespec='seconds')}\n")

    return (f"Prior authorization SUBMITTED for {claim_id} ({drug}). "
            f"Reference {reference}. Approved by {reviewer}.")


READ_ONLY_TOOLS = [get_claim, lookup_reject_code, check_eligibility, search_policy]
ACTION_TOOLS = [submit_prior_auth]
ALL_TOOLS = READ_ONLY_TOOLS + ACTION_TOOLS
