"""
CLAIMS AGENT - JOB 2: look something up in the filing cabinet.

This file does ONE thing: given a question, find the most relevant pieces of
policy. It does NOT call Claude. It is just the search step, on its own, so you
can see exactly what retrieval returns before we wire it into the agent.

Requires: run claims_agent_policy_ingest.py first (it builds policy_index/).

Run from the ai-lab folder:
    .venv/bin/python claims_agent_policy_search.py
"""
import os
import re

from sentence_transformers import SentenceTransformer
import chromadb

from query_expansion import expand      # brand -> generic + drug class, in code

# Anchor every path to THIS file's folder, so the script works
# no matter which directory it is run from (Airflow runs it from elsewhere).
HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(HERE, "policy_index")
COLLECTION = "plan_a_policies"
EMBED_MODEL = "all-MiniLM-L6-v2"      # MUST be the same model used to ingest

# Load these ONCE when the file is imported or run - not on every search.
# Loading the model takes a few seconds; doing it per question would be painful.
_embedder = SentenceTransformer(EMBED_MODEL)
_collection = chromadb.PersistentClient(path=INDEX_DIR).get_collection(COLLECTION)


# Tokens worth searching for LITERALLY, not by meaning. Embeddings compress a
# passage into 384 numbers describing what it is ABOUT - which is exactly wrong
# for an identifier, where the only thing that matters is the exact characters.
EXACT_TOKEN = re.compile(
    r"\b("
    r"\d{4,5}-\d{3,4}-\d{1,2}"    # NDC, e.g. 00169-4132-12
    r"|J\d{4}"                     # HCPCS J-code, e.g. J3490
    r"|[A-Z][A-Za-z]{3,}\d*"       # Capitalised word - brand names like Ozempic
    r")\b"
)

# Words that match the capitalised-word rule but are policy vocabulary, not
# identifiers. Searching for these literally would match nearly every chunk.
NOT_IDENTIFIERS = {
    "Prior", "Authorization", "Plan", "Member", "What", "When", "Where", "Which",
    "How", "Does", "Schedule", "Tier", "Claim", "Drug", "Policy", "Refill",
    "Coverage", "Covered", "Criteria", "Requirements", "Rules", "Review",
}


def _lexical_hits(question, limit=5):
    """Find chunks that literally CONTAIN an identifier from the question.

    This is the half that embeddings cannot do. A brand name or an NDC either
    appears in a passage or it does not - that is a string question, not a
    similarity question, and full-text search answers it exactly.
    """
    tokens = [t for t in EXACT_TOKEN.findall(question) if t not in NOT_IDENTIFIERS]
    found = {}
    for token in dict.fromkeys(tokens):          # de-duplicate, keep order
        try:
            res = _collection.get(where_document={"$contains": token},
                                  include=["documents", "metadatas"], limit=limit)
        except Exception:
            continue                              # full-text unsupported - skip
        for i, cid in enumerate(res["ids"]):
            found.setdefault(cid, {
                "text": res["documents"][i],
                "source": res["metadatas"][i]["source"],
                "version": res["metadatas"][i]["version"],
                "matched": token,
            })
    return found


def search_policy(question, top_k=3, hybrid=True):
    """Find the policy passages most relevant to `question`.

    HYBRID: semantic search finds passages that MEAN the same thing; keyword
    search finds passages that literally CONTAIN the identifier. Brand names,
    NDC and J-codes need the second kind - "Is Ozempic covered?" scored 0.254
    and missed the right document entirely, because the corpus talks about
    GLP-1 receptor agonists, never about Ozempic by name.

    Query expansion (below) fixes the drugs we KNOW about. Keyword search
    catches the ones we do not: a new brand, or an NDC in no lookup table.
    The two are complementary, not alternatives.

    Returns a plain list of dicts - plain types only, so it can later be
    converted to text with json.dumps() when we hand it to Claude.
    """
    # 0. EXPAND FIRST. The agent writes this question itself and is unreliable
    #    about drug names - "Is Ozempic covered?" misses the right document
    #    entirely, because the corpus talks about GLP-1 receptor agonists, not
    #    brand names. We know the drug, so we look the class up in code rather
    #    than hoping the model remembers it. See query_expansion.py.
    question, added = expand(question)
    if added:
        print(f"[policy-search] expanded query with: {', '.join(added)}")

    # 1. Turn the question into numbers, the SAME way the documents were turned
    #    into numbers during ingestion. This only works because it's the same model.
    question_vector = _embedder.encode(question).tolist()

    # 2. Ask the index for the closest pieces. Pull MORE than we need, so the
    #    merge below has candidates to re-rank rather than a fixed top_k.
    hits = _collection.query(
        query_embeddings=[question_vector],
        n_results=min(top_k * 3, 20),
        include=["documents", "metadatas", "distances"],
    )

    # 3. Repackage. Chroma nests everything one level deep (hits["documents"][0])
    #    because you can search several questions at once. We only sent one.
    scored = {}
    for i in range(len(hits["ids"][0])):
        meta = hits["metadatas"][0][i]
        scored[hits["ids"][0][i]] = {
            "text": hits["documents"][0][i],
            "source": meta["source"],       # which file it came from
            "version": meta["version"],     # which version of that policy
            # distance: 0 = perfect match, higher = less related.
            # We flip it so a bigger number means "more relevant".
            "relevance": round(1 - hits["distances"][0][i], 3),
            "matched": None,
        }

    # 4. THE KEYWORD HALF. Anything containing the literal identifier is
    #    promoted: an exact match on a brand name or NDC is strong evidence,
    #    and it is evidence the vector search structurally cannot produce.
    if hybrid:
        for cid, hit in _lexical_hits(question).items():
            if cid in scored:
                # Found by BOTH - the strongest signal there is.
                scored[cid]["relevance"] = round(scored[cid]["relevance"] + 0.25, 3)
                scored[cid]["matched"] = hit["matched"]
            else:
                # Found ONLY by keyword - the case semantic search missed.
                # Seeded at a floor score so it can enter the results at all.
                hit["relevance"] = 0.50
                scored[cid] = hit

    ranked = sorted(scored.values(), key=lambda r: r["relevance"], reverse=True)
    return ranked[:top_k]


# ===========================================================================
# Try it out.
# ===========================================================================
if __name__ == "__main__":
    questions = [
        "What are the prior authorization criteria for Ozempic?",
        "How do I resolve a DUR reject?",
        "When can a member get an early refill?",
    ]

    for q in questions:
        print("\n" + "=" * 72)
        print(f"QUESTION: {q}")
        print("=" * 72)
        for r in search_policy(q):
            print(f"\n  relevance {r['relevance']}   from {r['source']}  (v{r['version'].split('|')[0].strip()})")
            snippet = r["text"].replace("\n", " ")
            print(f"  {snippet[:260]}...")
    print()
