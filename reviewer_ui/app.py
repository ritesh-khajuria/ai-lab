"""
CLAIMS REVIEWER - one screen for both queues.

This replaces two clumsy interfaces:
    - the Swagger /docs page (for the batch/API pipeline)
    - approve.py on the command line (for the LangGraph pipeline)

A pharmacist should never see either of those.

    Settings (the gear icon)  ->  switch queue, set your reviewer name
    Type "refresh"            ->  reload the queue
    Click Approve / Reject    ->  the decision goes to the matching API

THE TWO QUEUES ARE NOT THE SAME THING - and the UI says so:

    claims_agent   the agent already finished. You are approving a DRAFT.
                   Rejecting costs nothing; nothing was going to happen.

    langgraph      the agent is PAUSED MID-EXECUTION. You are authorising an
                   ACTION. Approving submits a real prior authorization.

START:  do NOT use `chainlit run` - its CLI calls nest_asyncio.apply(), which
        breaks static file serving on Python 3.14 and gives you a blank page.
        Launch through serve.py instead (read its docstring for why):

    cd reviewer_ui
    .venv/bin/python serve.py          ->  http://127.0.0.1:8300/reviewer

The two APIs must be running:
    claims_agent   port 8100   (./start_services.sh)
    langgraph      port 8200   ../.venv/bin/uvicorn api:app --port 8200
"""
import httpx
import chainlit as cl
from chainlit.input_widget import Select, TextInput

# A decision with no justification is not an audit record. Short enough not to
# be a nuisance, long enough to stop "ok" and stray Enter presses.
MIN_REASON = 10

QUEUES = {
    "claims_agent": {
        "label": "Draft resolutions — review only",
        "base": "http://127.0.0.1:8100",
        "kind": "draft",
        "note": "📄 These are DRAFTS. The agent has already finished and nothing "
                "runs when you decide — you are signing off on wording. "
                "Reversible.",
        # The buttons SAY WHAT THEY DO. "Approve" is meaningless on its own.
        "yes": "✅ Accept draft",
        "no": "✏️ Send back",
        "confirm": None,          # nothing executes, so no consequence line
    },
    "langgraph": {
        "label": "🔴 Prior authorizations — LIVE, submits on approval",
        "base": "http://127.0.0.1:8200",
        "kind": "action",
        "note": "🔴 **These are NOT drafts.** Each agent is PAUSED mid-execution, "
                "waiting on you. Accepting **submits a real prior authorization "
                "to the plan**. It cannot be undone.",
        "yes": "🔴 SUBMIT prior auth",
        "no": "⛔ Block submission",
        # Shown BEFORE the decision is taken, naming the actual consequence.
        "confirm": ("Submitting a prior authorization for **{drug}** on claim "
                    "**{claim_id}**.\n\nThis goes to the plan and **cannot be "
                    "recalled.**"),
    },
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def cfg():
    return QUEUES[cl.user_session.get("queue", "claims_agent")]


async def fetch(path):
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(cfg()["base"] + path)
        r.raise_for_status()
        return r.json()


async def post(path, body):
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(cfg()["base"] + path, json=body)
        return r.status_code, (r.json() if r.content else {})


# ---------------------------------------------------------------------------
# show the queue
# ---------------------------------------------------------------------------
async def show_queue():
    c = cfg()
    try:
        if c["kind"] == "draft":
            items = await fetch("/reviews?status=PENDING")
        else:
            items = await fetch("/pending")
    except Exception as e:
        await cl.Message(
            content=f"**Cannot reach the {c['label']} API** at `{c['base']}`.\n\n"
                    f"`{type(e).__name__}: {e}`\n\nIs it running?").send()
        return

    header = f"### {c['label']}\n{c['note']}\n\n**{len(items)} awaiting your decision**"
    await cl.Message(content=header).send()

    if not items:
        await cl.Message(content="_Nothing in this queue. Type `refresh` to check again._").send()
        return

    for item in items:
        if c["kind"] == "draft":
            claim_id = item["claim_id"]
            ref = item["review_id"]
            body = (f"**{claim_id}** · review {ref} · raised {item['created_at']}\n\n"
                    f"_Type `show {claim_id}` to read the full draft._")
        else:
            claim_id = item["claim_id"]
            ref = claim_id
            body = (f"**{claim_id}** · {item.get('drug') or ''}\n\n"
                    f"{item.get('prompt') or 'Awaiting authorisation.'}\n\n"
                    f"_Type `show {claim_id}` to read the justification._")

        payload = {"claim_id": claim_id, "ref": ref,
                   "drug": item.get("drug") or "this medication"}
        await cl.Message(
            content=body,
            actions=[
                cl.Action(name="approve", label=c["yes"], payload=payload),
                cl.Action(name="reject", label=c["no"], payload=payload),
            ],
        ).send()


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------
@cl.on_chat_start
async def start():
    cl.user_session.set("queue", "claims_agent")
    cl.user_session.set("reviewer", "r.khajuria")

    await cl.ChatSettings([
        Select(id="queue", label="Queue",
               values=list(QUEUES.keys()), initial_index=0),
        TextInput(id="reviewer", label="Your reviewer id",
                  initial="r.khajuria"),
    ]).send()

    await cl.Message(
        content="## Claims review\n"
                "Use the **gear icon** to switch queue or set your reviewer id.\n"
                "Type `refresh` to reload, `show <CLAIM_ID>` to read the detail."
    ).send()
    await show_queue()


@cl.on_settings_update
async def settings_changed(settings):
    cl.user_session.set("queue", settings.get("queue", "claims_agent"))
    cl.user_session.set("reviewer", settings.get("reviewer") or "unknown")
    await cl.Message(content=f"Switched to **{cfg()['label']}** "
                             f"as `{cl.user_session.get('reviewer')}`").send()
    await show_queue()


@cl.on_message
async def on_message(message: cl.Message):
    text = message.content.strip()

    if text.lower() in ("refresh", "r"):
        await show_queue()
        return

    if text.lower().startswith("show "):
        claim_id = text.split(maxsplit=1)[1].strip().upper()
        c = cfg()
        try:
            if c["kind"] == "draft":
                reviews = await fetch("/reviews?status=PENDING")
                match = next((r for r in reviews if r["claim_id"] == claim_id), None)
                if not match:
                    await cl.Message(content=f"`{claim_id}` is not pending here.").send()
                    return
                detail = await fetch(f"/reviews/{match['review_id']}")
                await cl.Message(content=f"### {claim_id} — draft\n\n{detail['draft']}").send()
            else:
                detail = await fetch(f"/pending/{claim_id}")
                await cl.Message(
                    content=f"### {claim_id} — pending action\n\n"
                            f"**Action:** {detail['action']}\n\n"
                            f"**Drug:** {detail.get('drug') or '-'}\n\n"
                            f"**Justification the agent wrote:**\n\n"
                            f"{detail.get('justification') or '(none)'}").send()
        except Exception as e:
            await cl.Message(content=f"Could not load `{claim_id}`: {e}").send()
        return

    await cl.Message(
        content="Commands: `refresh` · `show <CLAIM_ID>` · or use the "
                "Approve / Reject buttons above."
    ).send()


# ---------------------------------------------------------------------------
# the decision
# ---------------------------------------------------------------------------
async def decide(action: cl.Action, decision: str):
    claim_id = action.payload["claim_id"]
    ref = action.payload["ref"]
    c = cfg()
    reviewer = cl.user_session.get("reviewer") or "unknown"

    # GUARD 1 - one decision at a time.
    # Without this the buttons stay live while the reason prompt is open, so
    # clicking Reject then Approve opened a SECOND prompt, orphaned the first,
    # and recorded the rejection with "no reason given".
    in_flight = cl.user_session.get("deciding")
    if in_flight:
        await cl.Message(
            content=f"⏳ Answer the open question for **{in_flight}** first — "
                    f"one decision at a time.").send()
        return

    # GUARD 2 - don't re-decide something settled in this session. The API
    # returns 409 anyway, but a reviewer shouldn't be invited to try.
    decided = cl.user_session.get("decided") or []
    if claim_id in decided:
        await cl.Message(content=f"**{claim_id}** was already decided. "
                                 f"Type `refresh` for the current queue.").send()
        return

    cl.user_session.set("deciding", claim_id)
    try:
        # Name the consequence BEFORE asking for a reason - the reviewer should
        # know what is about to happen while they are still able to back out.
        if c["confirm"] and decision == "APPROVED":
            await cl.Message(
                content="🔴 **About to act**\n\n" + c["confirm"].format(
                    drug=action.payload.get("drug", "this medication"),
                    claim_id=claim_id)).send()

        notes = await cl.AskUserMessage(
            content=f"**{decision.title()} {claim_id}** — why? "
                    f"(at least {MIN_REASON} characters; recorded against "
                    f"`{reviewer}` and cannot be edited later)",
            timeout=180).send()
        reason = ((notes or {}).get("output") or "").strip()

        # GUARD 3 - the reason IS the audit record. No reason, no decision.
        if len(reason) < MIN_REASON:
            await cl.Message(
                content=f"❌ **Nothing was recorded for {claim_id}.**\n\n"
                        f"{'You did not answer in time.' if not reason else 'That reason was too short.'} "
                        f"A decision needs a written justification of at least "
                        f"{MIN_REASON} characters — it is the audit record a "
                        f"reviewer is accountable for.\n\n"
                        f"The claim is still pending. Click Approve or Reject "
                        f"again when you're ready.").send()
            return
    finally:
        # Always clear the flag, even if the reviewer walked away and the
        # prompt timed out - otherwise the UI deadlocks for the whole session.
        cl.user_session.set("deciding", None)

    body = {"decision": decision, "reviewer": reviewer, "notes": reason}
    path = (f"/reviews/{ref}/decision" if c["kind"] == "draft"
            else f"/pending/{ref}/decision")

    async with cl.Step(name=f"{decision} {claim_id}") as step:
        status, data = await post(path, body)
        step.output = str(data)[:500]

    if status == 200:
        # Remember it here too, so the sibling button on the same card is
        # refused immediately instead of round-tripping to a 409.
        cl.user_session.set("decided", (cl.user_session.get("decided") or []) + [claim_id])
        if c["kind"] == "draft":
            msg = (f"✅ **{claim_id} {decision}** by `{reviewer}`\n\n"
                   f"Recorded at {data.get('decided_at')}. "
                   f"No action was executed — this was a draft.")
        else:
            msg = (f"✅ **{claim_id} {decision}** by `{reviewer}`\n\n"
                   f"**What the agent did:** {data.get('result')}\n\n"
                   f"Graph complete: `{data.get('thread_complete')}`")
    elif status == 409:
        msg = f"⚠️ Already decided: {data.get('detail')}"
    else:
        msg = f"❌ Failed (HTTP {status}): {data.get('detail') or data}"

    await cl.Message(content=msg).send()
    await action.remove()


@cl.action_callback("approve")
async def on_approve(action: cl.Action):
    await decide(action, "APPROVED")


@cl.action_callback("reject")
async def on_reject(action: cl.Action):
    await decide(action, "REJECTED")
