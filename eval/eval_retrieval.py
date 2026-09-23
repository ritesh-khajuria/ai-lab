"""
EVAL HARNESS - part 1: is RETRIEVAL any good?

WHAT AN EVAL HARNESS IS
-----------------------
A fixed set of questions whose CORRECT answer you already know, run
automatically, scored as a number. Regression tests - but for a component
allowed to be imperfect, so you measure a RATE rather than pass/fail.

WHY START WITH RETRIEVAL
------------------------
Because it is free and fast. Embeddings run locally, so this costs nothing and
takes seconds - you can run it on every change. And it is the layer most likely
to break silently: re-chunk the index, swap the embedding model, add documents,
and answers quietly get worse with no error anywhere.

If retrieval fails, the agent CANNOT be right. It will reason fluently over the
wrong passage. So this is the cheapest high-value thing to measure.

WHAT IT MEASURES
----------------
    hit@1   was the right document the TOP result?
    hit@3   was it anywhere in the top 3? (the agent sees 3)
    MRR     mean reciprocal rank - 1.0 if always first, 0.5 if usually second.
            Rewards being close, unlike a bare pass/fail.

The gold set deliberately includes the same question in BRAND and GENERIC form
("Ozempic" vs "semaglutide"), because that is the weakness we already found by
hand. An eval turns "I noticed this once" into a number you can defend.

RUN:
    ../.venv/bin/python eval_retrieval.py

Exit code 1 if hit@3 falls below THRESHOLD - so it can gate a release.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "claims_agent")))

from claims_agent_policy_search import search_policy      # noqa: E402

GOLD_FILE = os.path.join(HERE, "gold_retrieval.csv")
TOP_K = 3
THRESHOLD = 0.80          # hit@3 must stay at or above this


def evaluate():
    with open(GOLD_FILE, "r") as f:
        cases = list(csv.DictReader(f))

    hit1 = hit3 = 0
    reciprocal_total = 0.0
    misses = []
    rows = []

    for case in cases:
        question = case["question"]
        expected = case["expected_source"].strip()

        results = search_policy(question, top_k=TOP_K)
        sources = [r["source"] for r in results]

        # Where did the right document land? 1-based; 0 means "not found".
        rank = sources.index(expected) + 1 if expected in sources else 0

        if rank == 1:
            hit1 += 1
        if rank >= 1:
            hit3 += 1
            reciprocal_total += 1 / rank
        else:
            misses.append((question, expected, sources[0] if sources else "-"))

        rows.append((question, case["note"], rank,
                     results[0]["relevance"] if results else 0.0))

    n = len(cases)
    print(f"\n{'rank':<5} {'top-1 relevance':<16} {'note':<42} question")
    print("-" * 110)
    for question, note, rank, rel in rows:
        mark = {1: "  1  ", 2: "  2  ", 3: "  3  ", 0: " MISS"}[rank]
        print(f"{mark:<5} {rel:<16} {note[:40]:<42} {question[:44]}")

    print("\n" + "=" * 60)
    print(f"cases   : {n}")
    print(f"hit@1   : {hit1}/{n}  ({hit1/n:.0%})   right doc was the top result")
    print(f"hit@3   : {hit3}/{n}  ({hit3/n:.0%})   right doc was somewhere the agent could see")
    print(f"MRR     : {reciprocal_total/n:.3f}")

    if misses:
        print(f"\n{len(misses)} MISS(es) - the agent never saw the right document:")
        for question, expected, got in misses:
            print(f"  q: {question}")
            print(f"     expected {expected}, top hit was {got}")

    passed = (hit3 / n) >= THRESHOLD
    print(f"\n{'PASS' if passed else 'FAIL'} - hit@3 {hit3/n:.0%} vs threshold {THRESHOLD:.0%}\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(evaluate())
