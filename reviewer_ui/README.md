# Reviewer UI — human-in-the-loop review

One Chainlit screen for a pharmacist to review both agent queues, replacing two interfaces a
non-engineer should never see: the Swagger `/docs` page and a command-line `approve.py`.

> Architecture-decisions record. Implementation was AI-assisted (Claude Code); the safety model
> is the substance.

## The decision everything follows from: draft ≠ action
The two queues are **fundamentally different kinds of decision**, and the UI refuses to blur them:

| Queue | What you're deciding | Reversible? | Buttons say |
|---|---|---|---|
| `claims_agent` | sign off on a **draft** the agent already finished | yes — nothing runs | "Accept draft" / "Send back" |
| `langgraph` | **authorize a live action** — the agent is paused mid-run | **no** — submits a real PA | "🔴 SUBMIT prior auth" / "Block submission" |

Generic "Approve / Reject" buttons are meaningless — a reviewer must see *what the button does*.
For the action queue, the **consequence is named before the decision is taken** ("this goes to the
plan and cannot be recalled"), while the reviewer can still back out.

## Safety decisions (the part that matters more than the UI)
- **The reason is the audit record.** No justification (≥10 chars), no decision — it's recorded
  against the reviewer id and can't be edited later.
- **Consequence before reason.** For live actions, the irreversible effect is shown *before* the
  reviewer is asked to justify — not after.
- **Guards, learned from real bugs:** one decision at a time (clicking Reject then Approve used to
  open a second prompt and record "no reason given"); no re-deciding something already settled; the
  in-flight flag is always cleared even on timeout, so the session can't deadlock.

## A debugging finding worth keeping
Do **not** launch with `chainlit run` — its CLI calls `nest_asyncio.apply()`, which breaks static
file serving on Python 3.14 and yields a blank page. Launch through `serve.py` instead. (The kind
of environment-specific failure that only surfaces by actually running the thing.)

## What I reviewed / decided vs. generated
- **Decided:** the draft-vs-action distinction as the central safety model; reason-as-audit;
  consequence-before-decision; the guard set (each traceable to a concrete failure).
- **Generated (Claude Code):** the Chainlit app, the queue plumbing, the API calls.

## Production concerns / next
Real reviewer identity (SSO) instead of a typed id; a persistent, tamper-evident audit store;
role-based access to the live-action queue; and the two backing APIs (`:8100` draft, `:8200`
action) behind proper auth.
