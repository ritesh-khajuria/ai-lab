# Claims Anomaly Agent

Watches the claims feed at **population level**. The other two agents work one
claim at a time; this one asks what changed across thousands of them.

```
claims_history.csv  →  integrity gate  →  detect (arithmetic)  →  explain (model)  →  report
```

## The one design decision everything follows from

**Statistics find the anomaly. The model explains it.**

An LLM handed a table of daily reject rates will confidently describe patterns
that are not there, and miss ones that are. Noticing that a rate moved from 15%
to 63% is arithmetic: deterministic, free, and testable against a known answer.

So `detect.py` contains no model call at all. The model's job starts afterwards,
when the question stops being *"did something change?"* and becomes *"what could
cause this and what should someone check?"* — which is judgement, and judgement
is what a model is actually for.

Same rule as routing on reject code 75 in `claims_agent/`, one level up.

## What that buys: the two halves fail differently

| | `detect.py` | `explain.py` |
|---|---|---|
| cost | free | one model call per finding |
| repeatable | byte-identical every run | varies between runs |
| how you test it | **unit test against planted anomalies** | judgement; eval, not assert |
| failure mode | wrong threshold | plausible-sounding nonsense |

`test_detect.py` can assert *"the GLP-1 spike must be found"* because
`make_history.py` planted it at a known day with a known size. You cannot write
that assertion for an explanation — which is exactly why the expensive,
unpredictable half is kept as small as possible.

## Files

| file | what it does |
|---|---|
| `make_history.py` | generates 60 days of synthetic claims with **planted** anomalies. Seeded, so it is reproducible. `PLANTED` is the answer key the tests import. |
| `detect.py` | four detectors, no model. Two-proportion z-test plus minimum-volume guards so it does not cry wolf. |
| `test_detect.py` | asserts every planted anomaly is found (recall) and holds false positives to a stated budget (precision). |
| `policy_search.py` | **pre-fetched** policy lookup — the query is built from the finding, not written by the model. |
| `query_expansion.py` | third deliberate copy. Brand → generic → drug class. |
| `explain.py` | the model layer. Structured JSON out, three guardrails in. |
| `run_daily.py` | integrity gate → detect → explain → dated report + append-only log. |

## Three things that were wrong first, and what they taught

**The fixture lied.** The answer key said the prescriber surge was 3x; the
generator hard-coded a probability that produced 1.47x, and the detector
correctly ignored it. The fixture was wrong, not the detector. The multiplier is
now *derived* so the two cannot disagree.

**The model asked for data we already had.** Its top recommended action was
"pull the reject codes and tally them" — which `detect.py` had already computed.
If you can answer a question deterministically, answer it before you ask the
model anything.

**The model rationalised noise.** Told a finding was real and asked for causes,
it rated a known false positive `urgency: high` and invented a separate
mechanism for each stray reject code. The fix was not a sterner prompt — it was
giving it a number to reason against: `top_code_share`. A real policy change
rejects for one reason (34 of 34 on code 75); noise scatters across four. With
that number visible, the same model now calls it *"normal small-sample
variation"* and sets urgency low.

## No approval step, and that is not a compromise

This agent reads history and writes a report. It cannot change anything, so
there is nothing to authorise.

| pipeline | agent does | human approves |
|---|---|---|
| `claims_agent/` | drafts a resolution | the **output** |
| `claims_agent_langgraph/` | submits a PA | the **action** |
| **this one** | explains a pattern | **nothing** |

The review pattern follows the authority. Adding a gate here would be ceremony.

## Run it

```bash
../.venv/bin/python make_history.py      # regenerate the fixture
../.venv/bin/python detect.py            # findings only, free
../.venv/bin/python test_detect.py       # does it still catch the planted ones?
../.venv/bin/python run_daily.py         # gate + detect + report, no model calls
../.venv/bin/python run_daily.py --explain   # adds hypotheses (costs money)
```

Airflow: `claims_anomaly_daily` — detect → explain → publish. The cheap
deterministic half runs first so a model outage degrades the report to
numbers-only instead of failing it entirely.

---
Synthetic claims and invented policy documents. No PHI, no real prescribers.
