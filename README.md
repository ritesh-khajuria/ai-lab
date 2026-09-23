# AI Lab — Applied RAG, Agents & Evaluation

A hands-on portfolio of applied-AI projects I built to go deep on the systems behind
production LLM applications: retrieval-augmented generation (RAG), agentic workflows,
retrieval/agent evaluation, and anomaly detection. The theme is a synthetic
**healthcare pharmacy-claims** domain — chosen to mirror the regulated, data-heavy
environments I work in.

> **Background:** I'm a data engineering leader with 15+ years in enterprise ETL, cloud
> migrations (Oracle/DataStage → AWS Glue/Snowflake), and data quality. This lab is where I
> pair that data foundation with modern LLM tooling and the discipline to measure it.

> ⚠️ **All data here is synthetic.** Claims, members, drugs, and policies are fabricated for
> demonstration — no real, client, or personal data is included.

## Projects

### 1. Claims agent — `claims_agent/`
An agentic workflow over synthetic pharmacy claims: **policy retrieval (RAG)** over mock
plan documents, query expansion, grounded decisioning, a reject/appeal queue, and
human-in-the-loop write-back. Grounding is explicitly tested
(`claims_agent_grounding_test.py`) so answers stay tied to source policy — not the model's
imagination.

### 2. Claims agent, LangGraph edition — `claims_agent_langgraph/`
The same problem re-implemented as an explicit **LangGraph** state machine — tools, an
audit store, batch and local runners, and a prior-authorization submission path — to
compare orchestration approaches and make the agent's control flow inspectable.

### 3. Claims anomaly detection — `claims_anomaly/`
A daily job that scans synthetic claims history for anomalies (unusual reject patterns,
stale refills), explains them with **policy-search context**, and logs findings. Pairs
classic data engineering (scheduled batch over CSV history) with retrieval.

### 4. Evaluation harness — `eval/`
Evaluation for both retrieval and the end-to-end agent (`eval_retrieval.py`,
`eval_agent.py`, `eval_end_to_end.py`) plus a data validator. Evaluation is the discipline
that separates a demo from something you can trust in production.

### 5. Reviewer UI — `reviewer_ui/`
A **Chainlit** app for reviewing agent decisions — a lightweight human-in-the-loop
interface over the claims pipeline.

## Tech
Python · Anthropic API · ChromaDB (vector store) · LangGraph · Chainlit · pandas · SQLite

## Running locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your own ANTHROPIC_API_KEY
```
Each project folder runs on its own; see the scripts' headers for entry points.
Vector indexes and databases are generated locally and are not committed.
