"""
SEE IT: one chunk of text -> its embedding numbers.
And PROVE that text past the model's limit is silently thrown away.

    .venv/bin/python claims_agent_embedding_demo.py
"""
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
tok = model.tokenizer
LIMIT = model.max_seq_length          # 256 tokens for this model

line = "=" * 78


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5))


# ===========================================================================
print("\n" + line)
print("PART 1 - ONE CHUNK, AND THE NUMBERS IT BECOMES")
print(line)

chunk = ("Reject code 88 indicates a DUR conflict. A DUR reject requires "
         "pharmacist review before the claim can be resubmitted.")

print("\nTHE TEXT (what a human reads):\n")
print(f"  \"{chunk}\"")
print(f"\n  {len(chunk)} characters   |   {len(tok.encode(chunk))} tokens")

vec = model.encode(chunk).tolist()

print("\n            |")
print("            |  the embedding model reads it")
print("            v")
print("\nTHE NUMBERS (what the computer searches):\n")
for row_start in range(0, 48, 6):
    nums = "  ".join(f"{n:+.4f}" for n in vec[row_start:row_start + 6])
    print(f"  {nums}")
print(f"  ... continuing to position {len(vec)}")
print(f"\n  {len(vec)} numbers. Always {len(vec)}, for any input.")
print("  No individual number means anything on its own. Only the WHOLE")
print("  list has meaning - and only by comparison to another list.")


# ===========================================================================
print("\n" + line)
print("PART 2 - PROOF THAT LONG TEXT IS SILENTLY CUT OFF")
print(line)

# Build a body that is deliberately longer than the model can read.
body = ("PLAN-A requires prior authorization for selected drug classes. "
        "The prescriber must submit clinical documentation including chart "
        "notes, recent laboratory values, and a complete medication history. "
        "Standard determinations are issued within seventy two hours of "
        "receipt. Urgent requests are decided within twenty four hours. "
        "Approval is granted for an initial period of six months. Renewal "
        "requires documented clinical improvement or a written rationale for "
        "continuation of therapy. Claims submitted without an approved prior "
        "authorization on file are rejected at adjudication and returned to "
        "the dispensing pharmacy with the applicable reject code. The "
        "pharmacy is responsible for notifying the prescriber. ") * 2

# Two versions: IDENTICAL start, COMPLETELY DIFFERENT ending.
version_a = body + " FINAL NOTE: the approved drug is OZEMPIC."
version_b = body + " FINAL NOTE: the approved drug is ATORVASTATIN."

print(f"\n  version A: {len(tok.encode(version_a))} tokens  (limit is {LIMIT})")
print(f"  version B: {len(tok.encode(version_b))} tokens")
print("\n  The two texts are identical EXCEPT for the last few words:")
print(f"     A ends: \"...the approved drug is OZEMPIC.\"")
print(f"     B ends: \"...the approved drug is ATORVASTATIN.\"")

vec_a = model.encode(version_a).tolist()
vec_b = model.encode(version_b).tolist()
sim = cosine(vec_a, vec_b)

print(f"\n  similarity of their embeddings: {sim:.6f}")
if sim > 0.9999:
    print("\n  1.000000 = THE EMBEDDINGS ARE IDENTICAL.")
    print("  The model never saw the ending. Everything past token 256 was")
    print("  thrown away before it was read. No error. No warning.")

# Now the practical damage.
print("\n  Practical consequence - search for the word that was cut off:\n")
for query in ["Ozempic", "prior authorization documentation"]:
    q = model.encode(query).tolist()
    print(f"    query {query!r:<38} -> similarity to version A: {cosine(q, vec_a):.3f}")

print("\n  'Ozempic' is literally IN version A - and scores near zero,")
print("  because that part of the text was never embedded.")


# ===========================================================================
print("\n" + line)
print("THE RULE")
print(line)
print(f"""
  This model reads at most {LIMIT} tokens (roughly {LIMIT * 4} characters).
  Our chunker is set to 1200 characters (roughly 300 tokens).

  1200 chars > {LIMIT} tokens  ->  a long section WILL be silently truncated.

  We have not been bitten yet only because every policy section happened
  to be short. That is luck, not design.

  ALWAYS set chunk size against the model's token limit.
""")
print(line + "\n")
