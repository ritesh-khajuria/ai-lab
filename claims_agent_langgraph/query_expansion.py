"""
LANGGRAPH PIPELINE - deterministic query expansion for policy search.

NOTE: this is a DELIBERATE DUPLICATE of claims_agent/query_expansion.py. The two
pipelines stay independently readable, so neither can break the other. The cost
is real and worth saying out loud: a drug added here must be added there too.

THE PROBLEM THIS SOLVES
-----------------------
The agent writes its own policy search query. Watch what it actually asked on
two identical Trulicity claims:

    CLM2008:  "Trulicity dulaglutide prior authorization criteria..."
    CLM2009:  "Trulicity prior authorization criteria..."

It added the generic name once out of two attempts. Nobody told it to; it was
guessing. And a bare drug question misses outright:

    "Is Ozempic covered?"                      -> right document NOT in top 3
    "Is Ozempic semaglutide GLP-1 covered?"    -> rank 1

WHY IT MISSES
-------------
Embeddings are weak on exact tokens - brand names, NDC and J-codes. Worse, the
policy corpus does not contain the word "Trulicity" ANYWHERE, and does not
contain "dulaglutide" either. The only route to that rule is the drug CLASS:
"GLP-1". So expanding brand -> generic is not enough on its own; the class term
is what actually connects a brand name to the rule that governs it.

THE PRINCIPLE
-------------
We already know the drug - it is right there in the claim. Whether Trulicity is
a GLP-1 is a fact, not a judgement, and a fact belongs in a lookup table, not in
a sentence a model has to remember mid-reasoning.

    Ask the model to reason. Look facts up in code.

Same lesson as routing on reject code 75: the first version asked the model to
evaluate a condition we already knew the answer to, and it got it wrong.

IN PRODUCTION this table is not hand-written - it comes from the drug reference
file you already license (First Databank, Medi-Span, RxNorm), keyed on the NDC
that is already on the claim.
"""

# brand (lowercase) -> terms the POLICY CORPUS actually uses.
# Chosen by checking the corpus, not by guessing: "GLP-1" appears in
# plan_a_prior_authorization.md, which is why it is on every GLP-1 row.
DRUG_TERMS = {
    # GLP-1 receptor agonists - the class the PA rule is written about
    "ozempic":     "semaglutide GLP-1 receptor agonist",
    "wegovy":      "semaglutide GLP-1 receptor agonist",
    "rybelsus":    "semaglutide GLP-1 receptor agonist",
    "mounjaro":    "tirzepatide GLP-1 receptor agonist",
    "zepbound":    "tirzepatide GLP-1 receptor agonist",
    "trulicity":   "dulaglutide GLP-1 receptor agonist",
    "victoza":     "liraglutide GLP-1 receptor agonist",
    "saxenda":     "liraglutide GLP-1 receptor agonist",

    # Common maintenance drugs, so formulary questions land too
    "lipitor":     "atorvastatin statin",
    "zocor":       "simvastatin statin",
    "glucophage":  "metformin",
    "zestril":     "lisinopril ACE inhibitor",
    "prinivil":    "lisinopril ACE inhibitor",
    "coumadin":    "warfarin anticoagulant",
    "lantus":      "insulin glargine",
    "basaglar":    "insulin glargine",

    # Controlled substances - the schedule is what the refill rule keys on
    "oxycontin":   "oxycodone Schedule II controlled substance",
    "oxycodone":   "Schedule II controlled substance",
    "percocet":    "oxycodone acetaminophen Schedule II controlled substance",
}


def expand(question):
    """Add generic and class terms for any drug named in the question.

    Returns (expanded_question, added_terms). The caller gets the added terms
    back so it can log them - a silent rewrite of someone's query is the kind
    of thing that is impossible to debug later.

    Deliberately additive: the original wording is never removed, so expansion
    can only give the index MORE to match on, never less.
    """
    if not question:
        return question, []

    lowered = question.lower()
    added = []
    for brand, terms in DRUG_TERMS.items():
        if brand not in lowered:
            continue
        # Only add terms that are not already in the question, so repeated
        # words do not dilute the embedding.
        for term in terms.split():
            if term.lower() not in lowered and term not in added:
                added.append(term)

    if not added:
        return question, []
    return f"{question} {' '.join(added)}", added


if __name__ == "__main__":
    # See it work:  ../.venv/bin/python query_expansion.py
    for q in ["Is Ozempic covered?",
              "Trulicity prior authorization criteria",
              "refill rules for oxycodone",
              "How do I appeal a denial?"]:
        expanded, added = expand(q)
        print(f"  in  : {q}")
        print(f"  out : {expanded}")
        print(f"  added: {added or '(nothing - no drug recognised)'}\n")
