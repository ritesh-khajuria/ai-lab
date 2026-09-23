"""
CLAIMS AGENT - the SAME RAG index, but in a real SQL table you can query.

Chroma hides everything inside its own files. This does the identical job with
an ordinary database table, so you can SELECT the rows and see what is actually
stored. Uses SQLite (already on your Mac). The pgvector version is nearly
identical - the differences are printed at the end.

Run from the ai-lab folder:
    .venv/bin/python claims_agent_policy_sqldemo.py

Afterwards you can poke at it yourself:
    sqlite3 policy_demo.db "SELECT source_document, section FROM policy_chunks;"
"""
import os
import json
import sqlite3
from datetime import datetime

from sentence_transformers import SentenceTransformer

from claims_agent_policy_ingest import read_policy_files, chunk

# Anchor every path to THIS file's folder, so the script works
# no matter which directory it is run from (Airflow runs it from elsewhere).
HERE = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(HERE, "policy_demo.db")
EMBED_MODEL = "all-MiniLM-L6-v2"

line = "=" * 78


# ===========================================================================
# 1. THE SCHEMA - this is the metadata contract, as an actual table.
# ===========================================================================
SCHEMA = """
CREATE TABLE policy_chunks (
    chunk_id          INTEGER PRIMARY KEY,
    source_document   TEXT    NOT NULL,   -- which file it came from
    document_version  TEXT    NOT NULL,   -- WHICH VERSION was in force
    plan_id           TEXT    NOT NULL,   -- filter before you search
    section           TEXT,               -- precise citation
    chunk_index       INTEGER NOT NULL,   -- which piece of the document
    content           TEXT    NOT NULL,   -- the actual words
    embedding         TEXT    NOT NULL,   -- 384 numbers (pgvector: vector(384))
    ingested_at       TEXT    NOT NULL    -- audit: when did we load this
);
"""


def cosine(a, b):
    """How similar are two vectors? 1.0 = identical meaning, 0 = unrelated."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb)


if __name__ == "__main__":
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)                     # start fresh each run

    con = sqlite3.connect(DB_FILE)

    # -------------------------------------------------------------------
    print("\n" + line)
    print("STEP 1 - CREATE THE TABLE")
    print(line)
    con.execute(SCHEMA)
    print(SCHEMA.strip())

    # -------------------------------------------------------------------
    print("\n" + line)
    print("STEP 2 - INGEST: read files, chunk, embed, INSERT rows")
    print(line)
    model = SentenceTransformer(EMBED_MODEL)
    now = datetime.now().isoformat(timespec="seconds")

    rows = 0
    for doc in read_policy_files():
        for i, piece in enumerate(chunk(doc["text"])):
            vector = model.encode(piece).tolist()
            section = piece.split("\n")[0][:60]          # first line = the heading
            con.execute(
                """INSERT INTO policy_chunks
                   (source_document, document_version, plan_id, section,
                    chunk_index, content, embedding, ingested_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (doc["filename"], doc["version"], "PLAN-A", section,
                 i, piece, json.dumps(vector), now),
            )
            rows += 1
    con.commit()
    print(f"  INSERT ... -> {rows} rows committed into policy_chunks")

    # -------------------------------------------------------------------
    print("\n" + line)
    print("STEP 3 - IT'S JUST A TABLE. Look at it.")
    print(line)
    print("\n  SELECT chunk_id, source_document, section FROM policy_chunks LIMIT 6;\n")
    for r in con.execute("""SELECT chunk_id, source_document, section
                            FROM policy_chunks ORDER BY chunk_id LIMIT 6"""):
        print(f"   {r[0]:>3} | {r[1]:<32} | {r[2][:38]}")

    print("\n  SELECT source_document, count(*) FROM policy_chunks GROUP BY 1;\n")
    for r in con.execute("""SELECT source_document, count(*)
                            FROM policy_chunks GROUP BY 1 ORDER BY 1"""):
        print(f"   {r[0]:<34} {r[1]} chunks")

    print("\n  -- one full row, so you can see every column:\n")
    r = con.execute("SELECT * FROM policy_chunks WHERE chunk_id = 8").fetchone()
    cols = [d[0] for d in con.execute("SELECT * FROM policy_chunks LIMIT 1").description]
    for name, value in zip(cols, r):
        shown = str(value).replace("\n", " ")
        if name == "embedding":
            nums = json.loads(value)
            shown = f"[{nums[0]:+.4f}, {nums[1]:+.4f}, ... ] ({len(nums)} numbers)"
        print(f"   {name:<18} : {shown[:88]}")

    # -------------------------------------------------------------------
    print("\n" + line)
    print("STEP 4 - SEARCH: filter first, then rank by similarity")
    print(line)
    question = "What are the prior authorization criteria for semaglutide?"
    print(f"\n  question: {question!r}\n")
    qvec = model.encode(question).tolist()

    # NOTE: we filter on plan_id in SQL FIRST, then score only what survives.
    # This is the "filter before you search" rule, in actual SQL.
    candidates = con.execute(
        """SELECT chunk_id, source_document, document_version, section, content, embedding
           FROM policy_chunks
           WHERE plan_id = ?""", ("PLAN-A",)).fetchall()
    print(f"  WHERE plan_id = 'PLAN-A'  ->  {len(candidates)} candidate rows")

    scored = []
    for cid, src, ver, sec, content, emb in candidates:
        scored.append((cosine(qvec, json.loads(emb)), cid, src, ver, sec, content))
    scored.sort(reverse=True)

    print("\n  top 3 by cosine similarity:\n")
    for score, cid, src, ver, sec, content in scored[:3]:
        print(f"   {score:.3f}  chunk {cid:<3} {src}  (v{ver.split('|')[0].strip()})")
        print(f"          {sec}")
        print(f"          {content[:120].replace(chr(10),' ')}...")
        print()

    con.close()

    # -------------------------------------------------------------------
    print(line)
    print("WHAT WOULD CHANGE WITH POSTGRES + PGVECTOR")
    print(line)
    print("""
  Schema - one column type changes:
      embedding  TEXT              -- SQLite: we stored JSON
      embedding  vector(384)       -- pgvector: a real vector type

  Search - the whole scoring loop above collapses into ONE SQL statement,
  because Postgres can do the distance maths itself:

      SELECT chunk_id, source_document, document_version, section, content
      FROM   policy_chunks
      WHERE  plan_id = 'PLAN-A'
        AND  '2026-08-18' BETWEEN effective_date AND expiry_date
      ORDER  BY embedding <=> %s        -- <=> is cosine distance
      LIMIT  3;

  That single query does filtering, ranking and limiting in the database -
  no Python loop, and it scales to millions of rows with an index:

      CREATE INDEX ON policy_chunks USING hnsw (embedding vector_cosine_ops);

  THAT is why pgvector exists. Not a different concept - the same table,
  with the similarity search pushed down into the database engine.
""")
    print(line + "\n")
