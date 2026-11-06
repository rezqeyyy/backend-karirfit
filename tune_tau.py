"""
Sets the retrieval threshold tau on development queries (thesis Section 3.5.4).

Step 1 – make the labelling sheet (2 queries per career = 30 by default):
    python tune_tau.py sheet            -> kb/tau_labels.csv
  For every query it lists the top-5 passages and the passages at ranks 8, 15 and 30, with
  their cosine similarity. Fill the column `relevant` with 1 or 0 using your written criteria.
  You may edit the `query` column first and re-run `sheet` to use your own queries.

Step 2 – compute tau from the filled sheet:
    python tune_tau.py fit              -> best tau (maximises balanced accuracy)
  Then set RAG_THRESHOLD=<tau> in backend/.env.

These queries are development data only: do not reuse them in the final evaluation.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHEET = HERE / "kb" / "tau_labels.csv"
QUERIES = HERE / "kb" / "dev_queries.csv"
RANKS = [1, 2, 3, 4, 5, 8, 15, 30]


def default_queries(model):
    out = []
    for c in model["careers"]:
        skills = {s["id"]: s["name"] for s in model["skills"]}
        top = [skills[r["skill"]] for r in c["requirements"][:6]]
        out.append((c["id"], f"{c['name_en']}. {', '.join(top[:3])}"))
        out.append((c["id"], f"{c['name_id']}: kompetensi {', '.join(top[3:6]) or ', '.join(top[:2])}"))
    return out


def make_sheet():
    import faiss  # noqa: F401
    from sentence_transformers import SentenceTransformer
    import faiss as F
    from rag import EMBED_MODEL, KB_DIR
    model = json.load(open(HERE / "model.json", encoding="utf-8"))
    if QUERIES.exists():
        queries = [(r["career"], r["query"]) for r in csv.DictReader(open(QUERIES, encoding="utf-8"))]
    else:
        queries = default_queries(model)
        with open(QUERIES, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["career", "query"]); w.writerows(queries)
    index = F.read_index(str(KB_DIR / "index.faiss"))
    chunks = json.load(open(KB_DIR / "chunks.json", encoding="utf-8"))
    enc = SentenceTransformer(EMBED_MODEL)
    with open(SHEET, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["q", "career", "query", "rank", "score", "chunk_id", "passage", "relevant"])
        for qi, (career, q) in enumerate(queries, 1):
            v = enc.encode([f"query: {q}"], normalize_embeddings=True)
            scores, ids = index.search(v, 400)
            ranked = [(s, chunks[i]) for s, i in zip(scores[0], ids[0])
                      if i >= 0 and chunks[i].get("retrievable", True) and career in chunks[i].get("careers", [])
                      and chunks[i].get("access") != "private"]
            for r in RANKS:
                if r <= len(ranked):
                    s, c = ranked[r - 1]
                    w.writerow([qi, career, q, r, f"{s:.4f}", c["id"], c["text"][:400].replace("\n", " "), ""])
    print(f"{len(queries)} queries -> {SHEET}. Fill `relevant` (1/0), then run: python tune_tau.py fit")


def fit():
    rows = [r for r in csv.DictReader(open(SHEET, encoding="utf-8")) if r["relevant"].strip() in ("0", "1")]
    if not rows:
        sys.exit("No labelled rows: fill the `relevant` column with 1/0 first.")
    data = sorted((float(r["score"]), int(r["relevant"])) for r in rows)
    pos = sum(y for _, y in data); neg = len(data) - pos
    if not pos or not neg:
        sys.exit("Need both relevant (1) and irrelevant (0) labels.")
    best = (0, None)
    for tau in sorted({s for s, _ in data}):
        tp = sum(1 for s, y in data if s >= tau and y == 1)
        tn = sum(1 for s, y in data if s < tau and y == 0)
        bacc = (tp / pos + tn / neg) / 2
        if bacc > best[0]:
            best = (bacc, tau)
    print(f"labelled pairs: {len(data)} (relevant {pos}, not relevant {neg})")
    print(f"best tau = {best[1]:.4f}  (balanced accuracy {best[0]:.3f})")
    print(f"-> put RAG_THRESHOLD={best[1]:.2f} in backend/.env")


if __name__ == "__main__":
    {"sheet": make_sheet, "fit": fit}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
