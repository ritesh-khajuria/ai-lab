# Evaluation Harness

Three layers of evaluation for the claims agent + its RAG. The premise: an LLM system fails
*silently* — it reasons fluently over the wrong passage and no error is ever raised — so quality
has to be measured as a number, not eyeballed.

> Architecture-decisions record. Implementation was AI-assisted (Claude Code); the evaluation
> strategy and the failures it surfaced are the point.

## The core decision: separate the cheap constant checks from the expensive deliberate ones
| | measures | cost | when |
|---|---|---|---|
| `eval_retrieval.py` | is the **index** good? | free, local | every change; **gates releases** |
| `eval_agent.py` | is the agent's **decision consistent**? | API calls | deliberately, before shipping |
| `eval_end_to_end.py` | did the agent **find** the right policy? | API calls | deliberately, before shipping |

## Layer 1 — retrieval (`eval_retrieval.py`)
A gold set of questions with known-correct source documents, scored as **hit@1 / hit@3 / MRR**.
Retrieval runs locally, so this is free and runs on every change — and it's the layer most likely
to break silently (re-chunk, swap the embedding model, add docs → answers quietly get worse). The
gold set deliberately includes the same question in **brand and generic form** ("Ozempic" vs
"semaglutide") — the weakness found by hand, turned into a number. Exits non-zero below the hit@3
threshold, so it can **gate a release** in CI.

## Layer 2 — agent consistency (`eval_agent.py`)
**The bug that motivated it:** two claims in the identical situation (same reject code, same
missing member, same prompt) produced **opposite decisions** — one *submitted* a prior
authorization, one *refused*. Same input, different action; nothing noticed. So this runs the
same claim N times and checks the agent decides the same way each time. A single run tells you
what it did *once*, not what it *does*. Safe by construction: its own checkpoint DB, unique thread
ids, and it never resumes a paused graph, so **no PA is ever actually submitted** — it only
records that the agent reached the interrupt.

## Layer 3 — end-to-end retrieval (`eval_end_to_end.py`)
**The blind spot it closes:** Layer 1 scores retrieval on questions *I* wrote. But the agent
writes *its own* queries — so the index can pass at 100% while the real agent misses. This swaps
`search_policy` for a recording wrapper (identical behavior, keeps the receipt), runs the real
agent, and asks: of everything the agent actually retrieved, was the answer document in there?
> A component eval that passes while the system fails is worse than no eval — it manufactures
> confidence instead of information.

## What I reviewed / decided vs. generated
- **Decided:** the three layers and *why each exists*; that the free check gates while paid checks
  run deliberately; that consistency and end-to-end are distinct failure modes worth separate tests.
- **Generated (Claude Code):** the harness code, scoring, and CLI guards.

## Production concerns / next
Gate CI on Layer 1; run Layers 2–3 before any prompt/model change; grow the gold set as new
failure cases appear; track scores over time so regressions are visible, not discovered.
