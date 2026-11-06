"""
Creates kb/chunks.jsonl with the derived "career profile" chunks (public layer, Table 3.5)
from model.json, and appends any extra chunk files you place in kb/sources/*.jsonl
(SKKNI units, occupation-map profiles, and – only for the PNJ deployment – RPS chunks with
access="private").

  python make_kb.py && python rag.py build kb/chunks.jsonl
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
model = json.load(open(HERE / "model.json", encoding="utf-8"))
names = {s["id"]: s["name"] for s in model["skills"]}
out = HERE / "kb"
out.mkdir(exist_ok=True)
n = 0
with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
    for c in model["careers"]:
        reqs = ", ".join(f"{names[r['skill']]} ({r['demand']:.0%})" for r in c["requirements"])
        text = (f"Career profile: {c['name_en']} ({c['name_id']}). Skills requested in at least 10% of "
                f"Indonesian job advertisements for this career, with the share of ads requesting each: {reqs}.")
        f.write(json.dumps({"id": f"CP-{c['id']}", "text": text, "source": f"Career profile derived from job ads ({model.get('demand_source')})",
                            "doc_type": "career_profile", "careers": [c["id"]], "access": "public",
                            "retrievable": True}, ensure_ascii=False) + "\n")
        n += 1
    for extra in sorted((out / "sources").glob("*.jsonl")) if (out / "sources").exists() else []:
        for line in open(extra, encoding="utf-8"):
            if line.strip():
                f.write(line if line.endswith("\n") else line + "\n")
                n += 1
print(f"wrote {n} chunks -> {out / 'chunks.jsonl'}")
