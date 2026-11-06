"""
Retrieval module (thesis Section 3.2.4 / 3.5.4).

Knowledge base: JSONL, one chunk per line:
  {"id": "SKKNI-J.62-001", "text": "...", "source": "SKKNI 282/2016 unit J.620100.001.02",
   "doc_type": "skkni|occupation_map|career_profile|rps|course", "careers": ["software_developer"],
   "access": "public|private", "retrievable": true}

Embeddings: intfloat/multilingual-e5-base (768-d), normalised, FAISS IndexFlatIP (cosine, exact).
Public mode (FR11): chunks with access=private are never retrieved.

Build:  python rag.py build kb/chunks.jsonl
Both sentence-transformers and faiss are OPTIONAL: if missing or no index exists, the app runs
without retrieval and explanations fall back to the template (NFR6).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KB_DIR = HERE / "kb"
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
TOP_K = int(os.getenv("RAG_TOP_K", 5))
TAU = float(os.getenv("RAG_THRESHOLD", 0.80))  # set on the 30 development queries
PUBLIC_MODE = os.getenv("PUBLIC_MODE", "1") == "1"


class Retriever:
    def __init__(self):
        self.ready = False
        try:
            import faiss  # noqa: F401
            from sentence_transformers import SentenceTransformer
            idx_path, meta_path = KB_DIR / "index.faiss", KB_DIR / "chunks.json"
            if not idx_path.exists():
                return
            import faiss as _f
            self.index = _f.read_index(str(idx_path))
            self.chunks = json.load(open(meta_path, encoding="utf-8"))
            self.encoder = SentenceTransformer(EMBED_MODEL)
            self.ready = True
        except Exception as e:  # optional dependency
            print("[rag] retrieval disabled:", e)

    def search(self, career: str, matched: list[str], gaps: list[str], career_name: str) -> list[dict]:
        if not self.ready:
            return []
        q = f"query: {career_name}. " + ", ".join(matched + gaps[:3])
        v = self.encoder.encode([q], normalize_embeddings=True)
        scores, ids = self.index.search(v, 200)
        out = []
        for s, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            c = self.chunks[i]
            if not c.get("retrievable", True) or career not in c.get("careers", []):
                continue
            if PUBLIC_MODE and c.get("access") == "private":
                continue
            if c.get("doc_type") == "roadmap" or s < TAU:
                continue
            out.append({k: c[k] for k in ("id", "text", "source")} | {"score": float(s)})
            if len(out) == TOP_K:
                break
        return out


def build(jsonl_path: str):
    import faiss
    from sentence_transformers import SentenceTransformer
    chunks = [json.loads(line) for line in open(jsonl_path, encoding="utf-8") if line.strip()]
    enc = SentenceTransformer(EMBED_MODEL)
    vecs = enc.encode([f"passage: {c['text']}" for c in chunks], normalize_embeddings=True,
                      batch_size=32, show_progress_bar=True)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    KB_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(KB_DIR / "index.faiss"))
    json.dump(chunks, open(KB_DIR / "chunks.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"indexed {len(chunks)} chunks -> {KB_DIR}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "build":
        build(sys.argv[2])
    else:
        print(__doc__)
