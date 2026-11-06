"""
Builds the RAG knowledge base and FAISS index (thesis Sections 3.2.4 and 3.5.4).

  1. reads every document listed in kb/docs/mapping.csv (PDF, DOCX or TXT in kb/docs/)
  2. cleans the text (page numbers, repeated headers/footers)
  3. chunks by document structure:
       skkni        -> one chunk per competency unit (KODE UNIT ...). Only the description,
                       elements and performance criteria are retrievable. The "BATASAN VARIABEL"
                       and "PANDUAN PENILAIAN" part is kept but marked retrievable=false.
       other types  -> paragraphs packed into chunks of <= 1,500 characters, 200 overlap
  4. adds the derived career-profile chunks from model.json (public)
  5. writes kb/chunks.jsonl and embeds everything with intfloat/multilingual-e5-base into
     kb/index.faiss + kb/chunks.json (read by rag.py at startup)

Usage:
  pip install -r requirements-rag.txt
  python build_rag.py              # chunk + index
  python build_rag.py --no-index   # only chunk (check kb/chunks.jsonl first)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
KB = HERE / "kb"
DOCS = KB / "docs"
MAX_CHARS, OVERLAP = 1500, 200


# ----------------------------------------------------------------------------- reading
def read_pages(path: Path) -> list[str]:
    suf = path.suffix.lower()
    if suf == ".pdf":
        from pypdf import PdfReader
        return [(p.extract_text() or "") for p in PdfReader(str(path)).pages]
    if suf == ".docx":
        import docx
        return ["\n".join(p.text for p in docx.Document(str(path)).paragraphs)]
    if suf in (".txt", ".md"):
        return [path.read_text(encoding="utf-8", errors="ignore")]
    raise ValueError(f"unsupported file type: {path.name}")


STRUCTURE_RE = re.compile(r"KODE\s+UNIT|JUDUL\s+UNIT|DESKRIPSI\s+UNIT|ELEMEN\s+KOMPETENSI|KRITERIA\s+UNJUK|"
                          r"BATASAN\s+VARIABEL|PANDUAN\s+PENILAIAN|KOMPETENSI\s+KUNCI", re.I)


def clean(pages: list[str]) -> str:
    """Removes page numbers and headers/footers: short lines that sit in the first or last two
    lines of a page and repeat on many pages. SKKNI structure headings are never removed."""
    lines_per_page = [[l.strip() for l in p.splitlines() if l.strip()] for p in pages]
    repeated = set()
    if len(pages) >= 3:
        edge = Counter(l for lines in lines_per_page for l in set(lines[:2] + lines[-2:]) if len(l) < 120)
        repeated = {l for l, n in edge.items() if n >= max(3, len(pages) * 0.4) and not STRUCTURE_RE.search(l)}
    out = []
    for lines in lines_per_page:
        for l in lines:
            if l in repeated or re.fullmatch(r"(?:-\s*)?\d{1,4}(?:\s*-)?|halaman \d+.*|page \d+.*", l, re.I):
                continue
            out.append(l)
    text = "\n".join(out)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ----------------------------------------------------------------------------- chunking
def pack(text: str, max_chars=MAX_CHARS, overlap=OVERLAP) -> list[str]:
    """Packs paragraphs/sentences into chunks of <= max_chars with a character overlap."""
    parts = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[A-Z0-9][^\n]{0,80}:)", text) if p.strip()]
    pieces = []
    for p in parts:  # split very long paragraphs on sentence boundaries
        while len(p) > max_chars:
            cut = p.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 2 else max_chars
            pieces.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            pieces.append(p)
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + 1 + len(p) > max_chars:
            chunks.append(cur)
            tail = cur[-overlap:]
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            cur = (tail + " " + p).strip()
        else:
            cur = (cur + "\n" + p).strip()
    if cur:
        chunks.append(cur)
    return chunks


UNIT_RE = re.compile(r"KODE\s+UNIT\s*:?\s*", re.I)
NONRETRIEVABLE_RE = re.compile(r"BATASAN\s+VARIABEL|PANDUAN\s+PENILAIAN", re.I)


def chunk_skkni(text: str) -> list[dict]:
    """One chunk per unit. Returns dicts: text, unit_code, title, retrievable."""
    starts = [m.start() for m in UNIT_RE.finditer(text)]
    if not starts:
        return [{"text": c, "unit_code": None, "title": None, "retrievable": True} for c in pack(text)]
    out = []
    for i, a in enumerate(starts):
        unit = text[a: starts[i + 1] if i + 1 < len(starts) else len(text)]
        code = re.search(r"KODE\s+UNIT\s*:?\s*([A-Z]\.?\s?[\d.]+\d)", unit, re.I)
        title = re.search(r"JUDUL\s+UNIT\s*:?\s*(.+)", unit, re.I)
        code = re.sub(r"\s", "", code.group(1)) if code else None
        title = title.group(1).strip()[:150] if title else None
        cut = NONRETRIEVABLE_RE.search(unit)
        main, rest = (unit[:cut.start()], unit[cut.start():]) if cut else (unit, "")
        head = f"{code or ''} {title or ''}".strip()
        for j, c in enumerate(pack(main)):
            out.append({"text": c if j == 0 or not head else f"[{head}] {c}", "unit_code": code, "title": title, "retrievable": True})
        for c in pack(rest):
            out.append({"text": f"[{head}] {c}" if head else c, "unit_code": code, "title": title, "retrievable": False})
    return out


# ----------------------------------------------------------------------------- main
def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:40] or "DOC"


def build_chunks() -> list[dict]:
    model = json.load(open(HERE / "model.json", encoding="utf-8"))
    names = {s["id"]: s["name"] for s in model["skills"]}
    valid_careers = {c["id"] for c in model["careers"]}
    chunks: list[dict] = []

    # 1) derived career profiles (public)
    for c in model["careers"]:
        reqs = ", ".join(f"{names[r['skill']]} ({r['demand']:.0%})" for r in c["requirements"])
        label = model.get("data_label") or model.get("demand_source")
        chunks.append({"id": f"CP-{c['id']}", "source": f"Profil karier dari data lowongan ({label})",
                       "doc_type": "career_profile", "careers": [c["id"]], "access": "public", "retrievable": True,
                       "text": f"Profil karier {c['name_id']} ({c['name_en']}). Keterampilan yang diminta di minimal "
                               f"10% lowongan untuk karier ini, dengan proporsi lowongan yang memintanya: {reqs}."})

    # 2) documents from kb/docs/mapping.csv
    mapping = DOCS / "mapping.csv"
    if not mapping.exists():
        print(f"[!] {mapping} not found: only career-profile chunks are built")
        return chunks
    for row in csv.DictReader(open(mapping, encoding="utf-8-sig")):
        if not row.get("file") or row["file"].startswith("#"):
            continue
        path = DOCS / row["file"].strip()
        if not path.exists():
            print(f"[!] skipped (file not found): {path.name}")
            continue
        careers = [c.strip() for c in row.get("careers", "").split("|") if c.strip()]
        unknown = set(careers) - valid_careers
        if unknown:
            raise SystemExit(f"{path.name}: unknown career id(s) {unknown}. Valid: {sorted(valid_careers)}")
        dtype = (row.get("doc_type") or "other").strip()
        access = (row.get("access") or "public").strip()
        source = (row.get("source") or path.stem).strip()
        prefix = (row.get("id_prefix") or slug(path.stem)).strip()
        text = clean(read_pages(path))
        if not text:
            print(f"[!] no text extracted from {path.name} (scanned PDF? run OCR first)")
            continue
        pieces = chunk_skkni(text) if dtype == "skkni" else [{"text": c, "unit_code": None, "title": None, "retrievable": True} for c in pack(text)]
        n_ret = 0
        seen_ids = Counter()
        for k, p in enumerate(pieces, 1):
            base = f"{prefix}-{p['unit_code']}" if p["unit_code"] else f"{prefix}-{k:03d}"
            seen_ids[base] += 1
            cid = base if seen_ids[base] == 1 else f"{base}-{seen_ids[base]}"
            src = f"{source}, unit {p['unit_code']} {p['title'] or ''}".strip() if p["unit_code"] else source
            chunks.append({"id": cid, "source": src, "doc_type": dtype, "careers": careers, "access": access,
                           "retrievable": p["retrievable"] and dtype != "roadmap", "text": p["text"]})
            n_ret += p["retrievable"]
        print(f"  {path.name}: {len(pieces)} chunks ({n_ret} retrievable) -> {', '.join(careers)}")
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-index", action="store_true", help="only write kb/chunks.jsonl")
    a = ap.parse_args()
    chunks = build_chunks()
    KB.mkdir(exist_ok=True)
    with open(KB / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    by_type = Counter(c["doc_type"] for c in chunks)
    print(f"\n{len(chunks)} chunks ({sum(c['retrievable'] for c in chunks)} retrievable) -> kb/chunks.jsonl")
    print("  per type:", dict(by_type))
    print("  per career:", dict(Counter(k for c in chunks if c["retrievable"] for k in c["careers"])))
    if not a.no_index:
        from rag import build
        build(str(KB / "chunks.jsonl"))


if __name__ == "__main__":
    main()
