"""
RQ5 offline evaluation: explanation quality with vs without retrieval (thesis Section 3.6.5).

For 60 test profiles (4 per career, random, fixed seed) taken from heldout_ads.csv (exported by the
Colab notebook), two explanations are generated for the profile's true career with the SAME LLM,
prompt, scores, skill lists, format and language (English); the only difference is whether the
retrieved passages are included. Each explanation is checked automatically by claims.py.

Outputs (folder eval_rq5/):
  results.csv          one row per profile x condition (evidence support, four parts, correct gaps)
  explanations.jsonl   full texts, passages and claim lists (reproducibility)
  manual_coding.csv    30 random pairs, blinded (A/B in random order) for manual coding
  manual_key.csv       which of A/B is RAG (keep away from the coder until coding is done)
  summary.json         means, paired Wilcoxon test and the acceptance criteria

Acceptance (Table 3.13): RAG evidence support >= 0.90 and higher than non-RAG (p < 0.05);
>= 90% of explanations complete (four parts); 100% correct gaps.

Usage:
  export GEMINI_API_KEY=...      # same model for both conditions (GEMINI_MODEL)
  python eval_rq5.py heldout_ads.csv [--n-per-career 4] [--lang en] [--sleep 4]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model"))
sys.path.insert(0, str(HERE))
try:
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
except ImportError:
    pass

import llm  # noqa: E402
from career_model.extract import SkillExtractor  # noqa: E402
from career_model.scoring import score  # noqa: E402
from claims import check_explanation  # noqa: E402
from rag import Retriever  # noqa: E402

SEED = 42


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("heldout")
    ap.add_argument("--n-per-career", type=int, default=4)
    ap.add_argument("--n-manual", type=int, default=30)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--sleep", type=float, default=4.0, help="seconds between LLM calls (free-tier rate limit)")
    ap.add_argument("--out", default=str(HERE / "eval_rq5"))
    a = ap.parse_args(argv)

    model = json.load(open(HERE / "model.json", encoding="utf-8"))
    names = {s["id"]: s["name"] for s in model["skills"]}
    extractor = SkillExtractor(model["skills"])
    retriever = Retriever()
    if not retriever.ready:
        print("[!] retrieval index not available: the RAG condition would equal non-RAG. Build it first (build_rag.py).")

    by_career = defaultdict(list)
    for row in csv.DictReader(open(a.heldout, encoding="utf-8")):
        sk = [s for s in row["skills"].split("|") if s]
        if len(sk) >= 3:
            by_career[row["career"]].append(sk)
    rng = random.Random(SEED)
    profiles = []
    for c in [x["id"] for x in model["careers"]]:
        pool = by_career.get(c, [])
        for sk in rng.sample(pool, min(a.n_per_career, len(pool))):
            profiles.append((c, sk))
    print(f"{len(profiles)} test profiles")

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    rows, pairs = [], []
    with open(out / "explanations.jsonl", "w", encoding="utf-8") as jf:
        for i, (career, skills) in enumerate(profiles, 1):
            prof = {"skills": {s: "advanced" for s in skills}, "interests": []}
            result = next(r for r in score(prof, model, use_interest=False) if r["career"] == career)
            passages = retriever.search(career, [names[r["skill"]] for r in result["matched"]],
                                        [names[r["skill"]] for r in result["gaps"]], result["name_en"])
            texts = {}
            for cond, ctx in (("rag", passages), ("non_rag", [])):
                try:
                    text, model_id = llm.explain(result, names, ctx, a.lang)
                except Exception as e:
                    text, model_id = "", f"error: {e}"
                chk = check_explanation(text, result, model, ctx, prof, extractor) if text else None
                texts[cond] = text
                rows.append({"profile": i, "career": career, "condition": cond, "model": model_id,
                             "prompt_version": llm.PROMPT_VERSION, "n_passages": len(ctx),
                             "n_claims": chk["n_claims"] if chk else 0, "n_supported": chk["n_supported"] if chk else 0,
                             "evidence_support": chk["evidence_support"] if chk else None,
                             "four_parts": chk["four_parts"] if chk else False,
                             "gaps_correct": chk["gaps_correct"] if chk else False})
                jf.write(json.dumps({"profile": i, "career": career, "condition": cond, "model": model_id,
                                     "skills": skills, "passages": ctx, "text": text,
                                     "claims": chk["claims"] if chk else []}, ensure_ascii=False) + "\n")
                time.sleep(a.sleep)
            pairs.append((i, career, texts["rag"], texts["non_rag"]))
            print(f"  [{i}/{len(profiles)}] {career}: RAG {rows[-2]['evidence_support']}  non-RAG {rows[-1]['evidence_support']}")

    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # blinded sample for manual coding (single coder, Section 3.6.5)
    sample = rng.sample(pairs, min(a.n_manual, len(pairs)))
    with open(out / "manual_coding.csv", "w", newline="", encoding="utf-8") as fc, \
         open(out / "manual_key.csv", "w", newline="", encoding="utf-8") as fk:
        wc, wk = csv.writer(fc), csv.writer(fk)
        wc.writerow(["pair", "career", "explanation_A", "explanation_B",
                     "A_claims_total", "A_claims_supported", "B_claims_total", "B_claims_supported", "notes"])
        wk.writerow(["pair", "A_is", "B_is"])
        for k, (i, career, rag, norag) in enumerate(sample, 1):
            flip = rng.random() < 0.5
            A, B = (norag, rag) if flip else (rag, norag)
            wc.writerow([k, career, A, B, "", "", "", "", ""])
            wk.writerow([k, "non_rag" if flip else "rag", "rag" if flip else "non_rag"])

    rag = [r for r in rows if r["condition"] == "rag"]
    nrag = [r for r in rows if r["condition"] == "non_rag"]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    es_r = [r["evidence_support"] for r in rag if r["evidence_support"] is not None]
    es_n = [r["evidence_support"] for r in nrag if r["evidence_support"] is not None]
    paired = [(x["evidence_support"], y["evidence_support"]) for x, y in zip(rag, nrag)
              if x["evidence_support"] is not None and y["evidence_support"] is not None]
    p = None
    try:
        from scipy.stats import wilcoxon
        if paired and any(x != y for x, y in paired):
            p = float(wilcoxon([x for x, _ in paired], [y for _, y in paired], alternative="greater").pvalue)
    except ImportError:
        print("scipy not installed: Wilcoxon test skipped (pip install scipy)")
    allx = rag + nrag
    summary = {
        "n_profiles": len(profiles),
        "evidence_support_rag": mean(es_r), "evidence_support_non_rag": mean(es_n), "wilcoxon_p": p,
        "complete_four_parts": mean([r["four_parts"] for r in allx]),
        "correct_gaps": mean([r["gaps_correct"] for r in allx]),
    }
    summary["accept_evidence"] = bool(summary["evidence_support_rag"] is not None and summary["evidence_support_rag"] >= 0.90
                                      and es_n and summary["evidence_support_rag"] > summary["evidence_support_non_rag"]
                                      and p is not None and p < 0.05)
    summary["accept_completeness"] = bool(summary["complete_four_parts"] is not None and summary["complete_four_parts"] >= 0.90)
    summary["accept_gaps"] = summary["correct_gaps"] == 1.0
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
