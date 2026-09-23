"""
CLAIMS ANOMALY AGENT - policy lookup.

OWN COPY, on purpose - claims_agent/ has its own. Code is duplicated so each
pipeline reads standalone; the INDEX is shared, because there is one ingestion
pipeline and many readers. You would never have three applications maintain
three copies of the same policy corpus - those drift, and a drifted policy
index is far worse than a duplicated function.

    code    duplicated  (independence)
    index   shared      (one producer, many consumers)

WHAT IS DIFFERENT HERE - PRE-FETCH, NOT AGENTIC SEARCH
------------------------------------------------------
In claims_agent/ the model writes its own search query. That turned out to be
unreliable: given two identical claims it added the generic drug name to one
query and not the other, and a bare "Is Ozempic covered?" missed the right
document entirely.

Here we do not ask. A finding already names the drug and the reject code, so
the query is BUILT FROM THE FINDING in code. There is no coin flip, no wasted
first search, and the same finding always retrieves the same policy.

    claims_agent    the model decides what to look up   (flexible, variable)
    here            the finding determines it           (fixed, repeatable)

Flexibility is worth paying for when you cannot predict the question. For a
fixed set of finding types, you can - so don't pay.
"""
import os

from sentence_transformers import SentenceTransformer
import chromadb

from query_expansion import expand

HERE = os.path.dirname(os.path.abspath(__file__))
# The index built by claims_agent/claims_agent_policy_ingest.py.
INDEX_DIR = os.path.join(os.path.dirname(HERE), "claims_agent", "policy_index")
COLLECTION = "plan_a_policies"
EMBED_MODEL = "all-MiniLM-L6-v2"        # MUST match the model used to ingest

_embedder = None
_collection = None


def _load():
    """Load the model and index once, on first use.

    Deferred rather than done at import, because detect.py and the tests import
    nothing from here and should not pay a multi-second model load to run.
    """
    global _embedder, _collection
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
        _collection = chromadb.PersistentClient(path=INDEX_DIR).get_collection(COLLECTION)
    return _embedder, _collection


# A reject code is an IDENTIFIER, and embeddings are as weak on "75" as they
# are on "Ozempic" - the corpus is written in words, not code numbers. The
# first version of this file asked "What does reject code 75 mean?" and
# retrieved the APPEALS and FORMULARY documents; the prior-authorization
# document, which is the entire answer, never came back.
#
# Same fix as brand names: translate the identifier into the vocabulary the
# documents actually use, in code, before searching.
REJECT_MEANING = {
    "70": "product or service not covered by the plan, formulary exclusion",
    "75": "prior authorization required before the drug is covered",
    "76": "plan limitation exceeded, quantity or days supply limit",
    "79": "refill too soon, filled earlier than the plan allows",
    "88": "drug utilization review safety concern, DUR reject",
}


# What each kind of finding should look up. Fixed, because the finding types
# are fixed - this is a routing table, not a prompt.
QUERY_FOR = {
    "reject_rate_spike":
        "Why would claims for {subject} be rejected? Coverage and prior "
        "authorization criteria for {subject}.",
    "reject_code_mix_shift":
        "{meaning}. What causes this rejection and how is it resolved?",
    "new_drug":
        "Is {subject} covered? Formulary tier and prior authorization "
        "requirements for {subject}.",
    "prescriber_volume_shift":
        None,          # a volume change is not a policy question - do not fake one
}


def context_for(finding, top_k=2):
    """Retrieve the policy that bears on THIS finding. May return []."""
    template = QUERY_FOR.get(finding["kind"])
    if not template:
        return []

    # Translate a bare reject code into what it MEANS before searching.
    subject = finding["subject"]
    code = subject.replace("reject code", "").strip()
    meaning = REJECT_MEANING.get(code, subject)

    question, _ = expand(template.format(subject=subject, meaning=meaning))
    embedder, collection = _load()
    hits = collection.query(
        query_embeddings=[embedder.encode(question).tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"])

    return [{
        "text": hits["documents"][0][i],
        "source": hits["metadatas"][0][i]["source"],
        "version": hits["metadatas"][0][i]["version"],
        "relevance": round(1 - hits["distances"][0][i], 3),
    } for i in range(len(hits["ids"][0]))]


if __name__ == "__main__":
    demo = {"kind": "reject_code_mix_shift", "subject": "reject code 75"}
    for hit in context_for(demo):
        print(f"\n  {hit['source']}  relevance {hit['relevance']}")
        print(f"  {hit['text'][:240].replace(chr(10), ' ')}...")
