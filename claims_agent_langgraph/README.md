# Claims Agent — LangGraph version

The same agent as `../claims_agent/`, rebuilt as a **graph** instead of a
`for` loop — plus one thing the original cannot do: a tool that **acts**, and
therefore **stops to ask a human first**.

---

## Files

| File | What it is |
|---|---|
| `tools.py` | 4 read-only tools + `submit_prior_auth` (the one that acts) |
| `graph.py` | The state graph, nodes, edges, and the SQLite checkpointer |
| `run_local.py` | Single-claim demo: shows the pause, then resumes — approved and refused |
| `run_batch.py` | Starts a graph per rejected claim. **Never waits.** Called by the DAG. |
| `approve.py` | The human gate as a CLI. Resumes a paused thread. **Not** an Airflow task. |
| `graph_checkpoints.db` | Saved graph state (generated) |
| `graph_runs.csv` | Outcome per claim (generated) |
| `pa_submissions.csv` | Proof of what was actually submitted (generated) |

Run it:

```bash
cd claims_agent_langgraph
../.venv/bin/python run_local.py          # single claim, see the pause
../.venv/bin/python run_batch.py          # all rejected claims
../.venv/bin/python approve.py CLM1004 approve r.khajuria "Meets criteria"
```

---

## The DAG is a different shape

`airflow_home/dags/claims_agent_langgraph_dag.py`

```
start_graph_runs  ->  report_pending
```

| | `claims_agent_dag.py` | this DAG |
|---|---|---|
| What a task does | Runs the agent to **completion** | **Starts** a graph, lets it reach its own stopping point |
| Possible outcomes | draft written | `COMPLETED` **or** `AWAITING_APPROVAL` |
| Where the human fits | Reviews the draft afterwards, over the API | Resumes the paused graph, via `approve.py` |
| Is there a "wait for approval" task? | n/a | ❌ **Deliberately not** |

**Why no waiting task:** an Airflow task must finish. A graph awaiting a
pharmacist may sit for days. A task that waits would hold a worker slot, time
out red, and leave the DAG run permanently incomplete.

> **Airflow starts work and reports on it. It never waits for a person.**

Example run:

```
[graph-batch] CLM1001: COMPLETED - finished, no action required
[graph-batch] CLM1002: COMPLETED - finished, no action required
[graph-batch] CLM1004: AWAITING_APPROVAL - stopped before acting
[graph-batch] CLM1006: COMPLETED - finished, no action required
[graph-batch] AWAITING_APPROVAL=1  COMPLETED=3
```

The task exits green with one claim parked. A human approves later, in a
separate process. Re-running the DAG reports `ALREADY_DONE` for every claim
and makes **zero** API calls.

Reference data (`claims.csv`, `policy_index/`) is read from `../claims_agent/`
— in production both applications would query the same claims database and the
same vector store, so sharing one copy is realistic rather than a shortcut.

---

## The same agent, expressed two ways

| Concept | Hand-written (`../claims_agent/claims_agent.py`) | LangGraph (`graph.py`) |
|---|---|---|
| The loop | `for step in range(max_steps):` | `StateGraph` with edges that cycle |
| Conversation | `messages = [...]`, `messages.append(...)` | `State["messages"]` with the `add_messages` reducer |
| "Are we done?" | `if response.stop_reason == "end_turn": return` | `should_continue()` returning `END` |
| Running a tool | `TOOLS_BY_NAME[name](**args)` — ~15 lines of dispatch | `ToolNode(ALL_TOOLS)` — one line |
| Describing tools | Hand-written JSON schema per tool | `@tool` decorator reads the function signature and docstring |
| Runaway protection | `max_steps=6` | Recursion limit on the graph |
| **State after a crash** | **Gone** | **Saved in SQLite after every node** |
| **Pause mid-run** | **Impossible** | `interrupt()` |

Everything above the bold rows is the same idea with different syntax. The bold
rows are what the framework actually buys you.

---

## The one thing that is genuinely different

The original agent only ever **drafts text**. Nothing it produces changes the
world, so reviewing its output afterwards is perfectly safe.

`submit_prior_auth()` is different — it writes a submission that cannot be
undone. For that, "approve afterwards" is useless. The gate has to fire
**before** the action.

```
READ-ONLY TOOL                    ACTION TOOL
get_claim(...)                    submit_prior_auth(...)
   runs immediately                  interrupt()  ← graph STOPS here
   returns data                      state saved to SQLite
                                     ...waits for a human (minutes or days)...
                                     Command(resume={...})
                                     execution restarts INSIDE the tool
                                     approved? -> submit. refused? -> don't.
```

### What that looks like when you run it

```
[agent] -> calling submit_prior_auth({'claim_id': 'CLM1004', ...})

GRAPH IS PAUSED
  next node waiting to run : ('tools',)
  messages saved so far    : 9
  The human is being asked to approve:
     action        : submit_prior_auth
     claim_id      : CLM1004
     prompt        : A prior authorization is about to be SUBMITTED. Approve?

  NOTHING HAS BEEN SUBMITTED.
```

Then, on resume:

| Human decision | Result |
|---|---|
| `{"approved": True, "reviewer": "r.khajuria"}` | `Prior authorization SUBMITTED … Reference PA-CLM1004-…` |
| `{"approved": False, "reason": "No HbA1c on file"}` | `SUBMISSION BLOCKED by s.mishra … No prior authorization was submitted.` |

`pa_submissions.csv` ends with **one** row — the refused thread wrote nothing.

---

## Which pattern to use

| If the agent… | Review pattern | Implementation |
|---|---|---|
| Only produces text | Approve the finished output | `../claims_agent/claims_agent_api.py` |
| **Performs actions** | **Pause before acting** | `interrupt()` + checkpointer (here) |

Both are correct — for different agents. **The review pattern must match what
the agent can do.** That is the question to ask first in any agent design
review: *can this thing act, or only suggest?*

---

## A finding worth keeping

On the first run, with the instruction *"…and if a prior authorization is
required, submit one"*, the agent **chose to summarise instead of submitting**.
Same prompt, same claim — different behaviour. Changing the wording to
*"…then submit the required prior authorization"* made it act reliably.

> **If an action must happen, instruct it. Do not leave it to inference.**

The interrupt protects you from the agent acting when it shouldn't. Nothing
protects you from it *failing to act* — except an eval set that checks it did.

---

## A second finding — route in code, not in the prompt

`run_batch.py` first said: *"if and only if the reject code is 75, submit the
prior authorization"* — leaving the model to evaluate that condition.

It declined on **all four claims**, including CLM1004, which qualified.

The fix was not a better sentence. It was moving the decision out of the prompt:

```python
needs_pa = CLAIMS[claim_id]["reject_code"].strip() == "75"
task = "...submit the required prior authorization." if needs_pa else \
       "...explain the fix. Do NOT submit a prior authorization."
```

> **A deterministic condition belongs in an `if`, not in a sentence the model
> has to interpret.** We already knew the reject code — it was sitting in the
> data. Ask the model to *reason*; let the code decide *routing*.

That is a general rule for reviewing agent code: every business condition
buried in a prompt is a condition that will sometimes be evaluated wrongly, and
silently.

---

## What this version does NOT have

- No review **API** — approval is a CLI (`approve.py`), standing in for a UI
- No ingestion pipeline — it *reads* `policy_index/` but cannot build one
  (correct: one producer, many consumers)

Deliberate: this folder exists to make the **interrupt/resume** difference
visible, not to duplicate the original.
