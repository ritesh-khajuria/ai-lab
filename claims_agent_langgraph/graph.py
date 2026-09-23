"""
LangGraph version - THE GRAPH.

The hand-written agent was a `for` loop. This is the same idea expressed as a
GRAPH - nodes joined by edges, with the conversation carried in a shared state
object that LangGraph saves after every step.

    START ──► agent ──(tool calls?)──► tools ──► agent ──► ... ──► END
                 │
                 └── no tool calls ──► END

What the graph gives you that the loop did not:

  1. CHECKPOINTING   state is saved after every node, into SQLite.
  2. INTERRUPT       a node can pause the whole graph and wait for a human.
  3. RESUME          execution continues from exactly where it stopped -
                     even in a different process, hours later.

Compare with ../claims_agent/claims_agent.py to see the same agent both ways.
"""
import os
import sqlite3
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import ALL_TOOLS

HERE = os.path.dirname(os.path.abspath(__file__))
# The .env with the API key lives with the original agent's project root.
load_dotenv(os.path.join(os.path.dirname(HERE), ".env"))

CHECKPOINT_DB = os.path.join(HERE, "graph_checkpoints.db")
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a pharmacy claims resolution assistant for a health plan.

Given a rejected claim: use the tools to gather facts (never guess), consult the
plan policy documents, and produce a short structured summary:

What happened: (one line)
Why it rejected: (plain English, based on the reject code)
Recommended fix: (the concrete next step)

Cite the policy document and version whenever you rely on retrieved policy text,
e.g. (plan_a_refill_policy.md, v2026.02).

If - and only if - the claim rejected because a prior authorization is required
(reject code 75) AND the policy criteria plausibly apply, call submit_prior_auth
to file it. That action requires human approval, so explain your justification
clearly. For any other reject reason, do NOT submit anything."""


# ===========================================================================
# 1. THE STATE - what flows between nodes.
#
# `add_messages` is a reducer: when a node returns messages, they are APPENDED
# to the list rather than replacing it. That is the graph equivalent of the
# `messages.append(...)` you wrote by hand in the loop version.
# ===========================================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ===========================================================================
# 2. THE NODES - each is just a function taking state and returning an update.
# ===========================================================================
llm = ChatAnthropic(model=MODEL, max_tokens=1200, temperature=0)
llm_with_tools = llm.bind_tools(ALL_TOOLS)      # tells Claude the tools exist


def agent_node(state: State) -> dict:
    """Ask Claude what to do next, given the conversation so far."""
    response = llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response]}


# ToolNode is LangGraph's prebuilt runner: it reads the tool calls off the last
# message, executes them, and appends the results. It replaces the ~15 lines of
# dispatch code (`TOOLS_BY_NAME[name](**args)`) from the hand-written version.
tool_node = ToolNode(ALL_TOOLS)


# ===========================================================================
# 3. THE EDGES - where to go next.
# ===========================================================================
def should_continue(state: State) -> str:
    """After the agent speaks: are there tools to run, or are we done?

    This is the graph's version of `if response.stop_reason == "end_turn"`.
    """
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


# ===========================================================================
# 4. BUILD IT
# ===========================================================================
def build_graph(checkpointer=None):
    builder = StateGraph(State)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue,
                                  {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")        # tool results go back to the agent

    # The checkpointer is what makes interrupt/resume possible. Without it,
    # a pause would simply lose everything.
    return builder.compile(checkpointer=checkpointer)


def make_checkpointer():
    """SQLite-backed state store. Survives process restarts - so a graph can be
    paused today and resumed tomorrow, by a different process."""
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    return SqliteSaver(conn)
