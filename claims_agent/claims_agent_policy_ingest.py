"""
CLAIMS AGENT - JOB 1: fill the filing cabinet.

Reads the PLAN-A policy documents, cuts them into small pieces, turns each
piece into numbers, and saves everything to disk so we can search it later.

This file is SELF-CONTAINED. It does not import from any other file of ours.

    mock_policies/*.md  ->  chunks  ->  numbers  ->  policy_index/

Run from the ai-lab folder:
    .venv/bin/python claims_agent_policy_ingest.py

Run it once. Re-run it whenever the policy documents change.
"""
import os

# --- third-party packages (installed with pip, live in .venv/) --------------
from sentence_transformers import SentenceTransformer   # turns text into numbers
import chromadb                                          # stores + searches those numbers

# --- settings ---------------------------------------------------------------
# Anchor every path to THIS file's folder, so the script works
# no matter which directory it is run from (Airflow runs it from elsewhere).
HERE = os.path.dirname(os.path.abspath(__file__))
POLICY_DIR = os.path.join(HERE, "mock_policies")      # where the .md files are
INDEX_DIR = os.path.join(HERE, "policy_index")        # where we save the searchable index
COLLECTION = "plan_a_policies"      # a name for this set of documents
EMBED_MODEL = "all-MiniLM-L6-v2"    # small free model that runs on your Mac

CHUNK_CHARS = 1200                  # target size of each piece
CHUNK_OVERLAP = 200                 # repeat this much between pieces so we
                                    # don't cut a sentence in half and lose it


# ===========================================================================
# STEP 1 - READ the policy files off disk.
# ===========================================================================
def read_policy_files():
    """Return a list of {filename, title, version, text}, one per .md file."""
    docs = []
    for filename in sorted(os.listdir(POLICY_DIR)):
        if not filename.endswith(".md"):
            continue                                  # ignore anything else

        with open(os.path.join(POLICY_DIR, filename), "r") as f:
            text = f.read()                           # read the WHOLE file as one string

        lines = text.split("\n")
        title = lines[0].replace("#", "").strip()     # first line is "# Some Title"

        # Find the "Document version:" line. In healthcare, WHICH VERSION of a
        # policy was used is an audit question - so we keep it as metadata.
        version = "unknown"
        for line in lines[:5]:
            if line.startswith("Document version:"):
                version = line.replace("Document version:", "").strip()
                break

        docs.append({"filename": filename, "title": title,
                     "version": version, "text": text})
    return docs


# ===========================================================================
# STEP 2 - CHUNK: cut a long document into small retrievable pieces.
#
# WHY? We don't want to hand Claude a whole 1,500-word policy. We want the two
# or three paragraphs that actually answer the question.
#
# HOW? Split on the "## " headings first (they're natural section breaks).
# If a section is still too long, cut it further, with a little overlap.
# ===========================================================================
def chunk(text, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    """Split text into a list of pieces, each roughly `size` characters."""
    sections = [s.strip() for s in text.split("\n## ") if s.strip()]

    pieces = []
    for section in sections:
        if len(section) <= size:
            pieces.append(section)          # short enough already
            continue

        start = 0
        while start < len(section):
            piece = section[start:start + size]

            # try to end on a sentence boundary so pieces read cleanly
            dot = piece.rfind(". ")
            if dot > size * 0.5 and start + size < len(section):
                piece = piece[:dot + 1]

            pieces.append(piece.strip())

            if start + len(piece) >= len(section):
                break                        # reached the end of this section

            # Move forward. NOTE the max(): if we used (len(piece) - overlap)
            # alone, a short tail piece would advance the cursor by 1 character
            # and produce hundreds of near-duplicate chunks.
            start += max(len(piece) - overlap, size // 2)

    return [p for p in pieces if len(p) > 100]      # drop tiny fragments


# ===========================================================================
# MAIN - runs only when you execute this file directly.
# ===========================================================================
if __name__ == "__main__":

    print("\n=== STEP 1: READ the policy files ===")
    docs = read_policy_files()
    for d in docs:
        print(f"  {d['filename']:<34} {len(d['text']):>5} chars")

    print("\n=== STEP 2: CHUNK them into pieces ===")
    records = []                       # one entry per piece
    for d in docs:
        pieces = chunk(d["text"])
        print(f"  {d['filename']:<34} -> {len(pieces)} pieces")
        for i, piece in enumerate(pieces):
            records.append({
                "text": piece,                 # the actual words
                "title": d["title"],           # metadata: which document
                "source": d["filename"],       # metadata: the file name
                "version": d["version"],       # metadata: which version
                "chunk_index": i,              # metadata: which piece of it
            })
    print(f"  TOTAL: {len(records)} pieces")

    print("\n=== STEP 3: EMBED - turn each piece into numbers ===")
    print("  (runs locally on your Mac, costs nothing)")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [r["text"] for r in records]
    # GUARD - the quietest failure in the whole pipeline.
    #
    # all-MiniLM-L6-v2 reads at most 256 tokens (~1000 characters). Anything
    # past that is DROPPED SILENTLY: no error, no warning, just a vector built
    # from a fraction of the text. The chunk still looks complete when you print
    # it, so the index appears fine while retrieval quietly degrades.
    #
    # CHUNK_CHARS is 1200, which is larger than the model can read. Today the
    # mock policies are short enough that no chunk actually exceeds the limit -
    # so this has never bitten. Real policy PDFs would trip it immediately.
    #
    # A latent bug you cannot see is worse than one that shouts, so make it
    # shout.
    limit = model.max_seq_length
    oversized = [(i, len(model.tokenizer.encode(t))) for i, t in enumerate(texts)
                 if len(model.tokenizer.encode(t)) > limit]
    if oversized:
        worst = max(n for _, n in oversized)
        print(f"\n  WARNING: {len(oversized)} of {len(texts)} chunks exceed the "
              f"embedding model's {limit}-token limit (worst: {worst} tokens).")
        print(f"  Text past {limit} tokens is NOT embedded and cannot be retrieved.")
        print(f"  Fix: lower CHUNK_CHARS (line 31) until this warning disappears.\n")
    else:
        print(f"  all {len(texts)} chunks fit inside the model's {limit}-token limit")

    vectors = model.encode(texts, show_progress_bar=False)
    print(f"  {len(vectors)} pieces -> {len(vectors[0])} numbers each")

    print("\n=== STEP 4: INDEX - save pieces + numbers + metadata to disk ===")
    client = chromadb.PersistentClient(path=INDEX_DIR)
    try:
        client.delete_collection(COLLECTION)     # wipe and rebuild each run
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[f"p{i}" for i in range(len(records))],
        documents=texts,                                    # the words
        embeddings=[v.tolist() for v in vectors],           # the numbers
        metadatas=[{"title": r["title"], "source": r["source"],
                    "version": r["version"], "chunk_index": r["chunk_index"]}
                   for r in records],                       # where it came from
    )
    print(f"  stored {col.count()} pieces in '{INDEX_DIR}/'")

    print("\nThe filing cabinet is full.")
    print("Next: claims_agent_policy_search.py to look things up.\n")
