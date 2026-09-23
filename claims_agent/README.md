# Claims Agent — Pharmacy Claims Rejection Resolver

A tool-calling agent that investigates a rejected pharmacy claim and drafts a structured
resolution (*what happened / why it rejected / recommended fix*) for a human specialist to
review. Synthetic data only — no PHI.

> **How to read this file:** it's an architecture-decisions record, not a tutorial. It states
> the decisions the system makes, what I reviewed, what failed first, and what I'd do before
> production. Implementation was AI-assisted (Claude Code); the decisions and the review are mine.

## Problem
A rejected claim needs a human-readable explanation and a concrete next step. The facts live
in several places — the claim record, the NCPDP reject-code meaning, member eligibility, and
the *plan policy* — and which ones matter depends on the claim. A one-shot prompt can't know
in advance which lookups it needs.

## Why an agent (tool-calling loop), not a one-shot prompt
The agent loop — *ask the model → it requests a tool → run it → feed the result back → repeat
until it stops* — lets the model gather exactly the facts a given claim requires. Four tools:
`get_claim`, `lookup_reject_code`, `check_eligibility` (structured lookups) and `search_policy`
(RAG over plan documents). A `max_steps` cap bounds runaway loops.

## Why hybrid retrieval (the key RAG decision)
Pure semantic search **failed** on the exact thing this domain needs. *"Is Ozempic covered?"*
scored 0.254 and missed the right document entirely — because the corpus talks about *GLP-1
receptor agonists*, never "Ozempic" by name. Embeddings compress a passage into ~384 numbers
about what it *means*, which is precisely wrong for an **identifier** (a brand name, an NDC like
`00169-4132-12`, a J-code) where only the exact characters matter.

So `search_policy` is **hybrid**: semantic search (MiniLM + Chroma) for meaning, plus a lexical
pass that promotes any chunk literally containing an identifier. Found by both = strongest
signal. Found by keyword only = the case semantic search structurally cannot catch.

## Why query expansion lives in code, not the prompt
The agent writes its own search queries and is unreliable about drug names. Rather than hope the
model remembers that Ozempic is semaglutide is a GLP-1, `query_expansion.py` expands
brand → generic → drug class **in code** before retrieval. Deterministic facts we already have
shouldn't be left to inference — the same principle repeats across this repo.

## Grounding — where does the answer actually come from?
The system prompt forces the model to use only tool output, **cite source + version**
(e.g. `plan_a_refill_policy.md, v2026.02`), and say so if retrieval returns nothing rather than
filling the gap from memory. `claims_agent_grounding_test.py` probes the boundary deliberately:
a question the tools *can* answer (grounded, safe) vs one they *cannot* (the hallucination risk
zone) — so I could see, not assume, where answers originate.

## What I reviewed / decided vs. what was generated
- **Decided:** hybrid retrieval was necessary (proved by the Ozempic miss); expansion belongs in
  code; grounding must be enforced and *tested*, not trusted; the tool boundary defines the risk.
- **Reviewed:** tool schemas (the description is the only thing telling the model when to reach
  for policy vs claim data), the citation discipline, the max-steps guard.
- **Generated (Claude Code):** the implementation of the loop, tools, and retrieval.

## Production concerns
- **PHI:** synthetic here; real deployment needs access control, masking, and audit on every lookup.
- **Retrieval quality:** gated by `../eval/eval_retrieval.py` (hit@3 threshold) so a re-chunk or
  model swap can't quietly degrade answers.
- **Auditability:** citations make each recommendation traceable to a policy version.
- **Cost / latency:** the embedding model loads once; model calls are the spend to watch.

## What I'd do next
Wire the tools to a real claims DB and shared vector store (see the LangGraph version, which
already treats them as one producer / many consumers), expand the retrieval gold set, add
tool-call observability, and secure the write-back API.

## Files
`claims_agent.py` (the loop + tools) · `claims_agent_policy_search.py` (hybrid retrieval) ·
`query_expansion.py` (brand→generic→class) · `claims_agent_policy_ingest.py` (build the index) ·
`claims_agent_grounding_test.py` (grounding probe) · `claims_agent_api.py` + `review_store.py`
(draft + human review) · `mock_policies/` (synthetic plan documents).
