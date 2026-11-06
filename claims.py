"""
Automatic claim checker for generated explanations (thesis Sections 3.6.1 and 3.6.5).

Every factual claim the checker can recognise is verified against the system's own data:

  citation      [ID] cites a passage that was actually retrieved for this explanation
  unit_code     an SKKNI unit code (e.g. J.620100.021.02) appears in a retrieved passage
  fit_score     "NN/100" equals the computed fit score (rounded)
  competency    "kompetensi/competency ... NN%" equals the computed competency fit K
  demand        "<skill> ... NN%" equals the demand of that skill for the career (±1 point)
  skill_status  a skill named in part 2 is matched; a skill named in part 3 is a computed gap
  course        a course named as evidence (parts 1-3) was taken by the user and teaches the skill on
                that line; a course suggested in part 4 teaches at least one of the gaps

Output: claims list, number supported, evidence support = supported / checked, whether the
explanation has the four parts, and whether every gap it names is a real gap ("correct gaps").
Claims the checker cannot recognise are left for manual coding (Section 3.6.5).
"""
from __future__ import annotations

import re

from career_model.extract import SkillExtractor

PCT_RE = re.compile(r"(\d{1,3}(?:[.,]\d+)?)\s*%")
FIT_RE = re.compile(r"\b(\d{1,3})\s*/\s*100\b")
CITE_RE = re.compile(r"\[([A-Za-z0-9_\-:.]+)\]")
UNIT_RE = re.compile(r"\b([A-Z]\.\d{6}\.\d{3}\.\d{2})\b")
PART_RE = re.compile(r"^\s*(?:#+\s*)?\**\s*([1-4])[.)]\s*\**\s*\S", re.M)
COMP_WORDS = ("kompetensi", "competency")
COURSE_WORDS = ("mata kuliah", "matakuliah", "kuliah", "course")


def _sections(text: str) -> dict[int, str]:
    """Splits the explanation into its numbered parts 1-4 (text before part 1 goes to 0)."""
    marks = [(m.start(), int(m.group(1))) for m in PART_RE.finditer(text)]
    out, prev_pos, prev_n = {}, 0, 0
    for pos, n in marks:
        out[prev_n] = out.get(prev_n, "") + text[prev_pos:pos]
        prev_pos, prev_n = pos, n
    out[prev_n] = out.get(prev_n, "") + text[prev_pos:]
    return out


def check_explanation(text: str, result: dict, model: dict, passages: list[dict],
                      profile: dict | None = None, extractor: SkillExtractor | None = None) -> dict:
    extractor = extractor or SkillExtractor(model["skills"])
    names = {s["id"]: s["name"] for s in model["skills"]}
    demand = {r["skill"]: r["demand"] for r in result["matched"] + result["gaps"]}
    matched = {r["skill"] for r in result["matched"]}
    gaps = {r["skill"] for r in result["gaps"]}
    passage_ids = {p["id"] for p in passages}
    passage_text = " ".join(p.get("text", "") for p in passages)
    taken = set((profile or {}).get("courses", {}) or {})
    courses = {c["id"]: c for c in model.get("courses", [])}
    course_names = {c["id"]: [c["name_id"].lower(), c["name_en"].lower()] for c in courses.values()}

    career_names = sorted({n.lower() for c in model["careers"] for n in (c["name_id"], c["name_en"])}, key=len, reverse=True)

    def skills_in(line: str) -> list[str]:
        """Dictionary skills in a line; career names are removed first and a match lying inside a
        longer match of another skill is ignored (e.g. 'SQL' inside 'SQL Server')."""
        low = line.lower()
        for n in career_names:
            low = low.replace(n, " " * len(n))
        spans = {}
        for sid, pats in extractor.compiled:
            found = [m.span() for p in pats for m in p.finditer(low)]
            if found:
                spans[sid] = found
        keep = []
        for sid, sp in spans.items():
            def inside(a, b):
                return any(o_sid != sid and oa <= a and b <= ob and (ob - oa) > (b - a)
                           for o_sid, osp in spans.items() for oa, ob in osp)
            if any(not inside(a, b) for a, b in sp):
                keep.append((min(a for a, _ in sp), sid))
        return [sid for _, sid in sorted(keep)]

    claims: list[dict] = []

    def add(kind, claim, ok, detail=""):
        claims.append({"type": kind, "claim": claim[:160], "supported": bool(ok), "detail": detail})

    parts = _sections(text)
    for part_no, part in parts.items():
        for line in [l.strip() for l in part.splitlines() if l.strip()]:
            low = line.lower()
            skills_here = skills_in(line)
            pcts = [float(p.replace(",", ".")) for p in PCT_RE.findall(line)]

            for cid in CITE_RE.findall(line):
                if not UNIT_RE.fullmatch(cid) and cid not in ("1", "2", "3", "4"):
                    add("citation", f"[{cid}]", cid in passage_ids, "retrieved" if cid in passage_ids else "not retrieved")
            for code in UNIT_RE.findall(re.sub(r"\[[^\]]*\]", "", line)):
                add("unit_code", code, code in passage_text, "")

            for m in FIT_RE.finditer(line):
                v = int(m.group(1))
                add("fit_score", m.group(0), abs(v - round(result["fit"])) <= 1, f"computed {result['fit']:.1f}")

            used_pct = set()
            if any(w in low for w in COMP_WORDS) and pcts and not skills_here:
                v = pcts[0]
                add("competency", f"{v:g}%", abs(v - result["K"] * 100) <= 1, f"computed {result['K'] * 100:.1f}%")
                used_pct.add(0)
            remaining = [p for i, p in enumerate(pcts) if i not in used_pct]
            # pair skills with percentages in reading order (one-to-one when counts match)
            if skills_here and remaining:
                ordered = skills_here
                pairs = list(zip(ordered, remaining)) if len(ordered) == len(remaining) else \
                        ([(ordered[0], remaining[0])] if len(ordered) == 1 or len(remaining) == 1 else [])
                for sk, v in pairs:
                    if sk in demand:
                        add("demand", f"{names.get(sk, sk)} {v:g}%", abs(v - demand[sk] * 100) <= 1,
                            f"data {demand[sk] * 100:.1f}%")
                    else:
                        add("demand", f"{names.get(sk, sk)} {v:g}%", False, "skill not required for this career")

            if part_no == 2:
                for sk in set(skills_here):
                    add("skill_status", f"{names.get(sk, sk)} = matched", sk in matched, "")
            elif part_no == 3:
                for sk in set(skills_here):
                    add("skill_status", f"{names.get(sk, sk)} = gap", sk in gaps, "")

            # course claims only where the line talks about a course (course names can equal skill names)
            if any(w in low for w in COURSE_WORDS):
                for cid, ns in course_names.items():
                    if any(n and n in low for n in ns):
                        if part_no == 4:   # advice: the course must exist and teach one of the gaps
                            ok = bool(set(courses[cid]["skills"]) & gaps)
                            add("course", courses[cid]["name_id"], ok, "teaches a gap" if ok else "teaches no gap skill")
                        else:              # statement about the user: the course must have been taken
                            teaches = not skills_here or any(s in courses[cid]["skills"] for s in skills_here)
                            add("course", courses[cid]["name_id"], cid in taken and teaches,
                                "taken" if cid in taken else "not taken by the user")

    gap_claims = [c for c in claims if c["type"] == "skill_status" and c["claim"].endswith("= gap")]
    supported = sum(c["supported"] for c in claims)
    return {
        "n_claims": len(claims),
        "n_supported": supported,
        "evidence_support": (supported / len(claims)) if claims else None,
        "four_parts": all(n in parts for n in (1, 2, 3, 4)),
        "gaps_named": len(gap_claims),
        "gaps_correct": all(c["supported"] for c in gap_claims),
        "claims": claims,
    }
