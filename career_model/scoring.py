"""
Deterministic scoring mechanism (thesis Section 3.5.2). No LLM is involved.

    d(c,s)  = n(c,s) / N(c)                                  (3.1)
    w(c,s)  = d(c,s) * ln(1 + |C| / m(s))                    (3.2)
    e(s)    = max(e_course(s), e_self(s), e_exp(s))          (3.3)
    K(c)    = sum_{s in R(c)} w(c,s) e(s) / sum w(c,s)       (3.4)
    Fit(c)  = 100 * [ alpha K(c) + (1 - alpha) M(c) ]        (3.5)
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

GRADE_POINTS = {"A": 1.00, "B": 0.75, "C": 0.50, "D": 0.0, "E": 0.0}
SELF_LEVEL = {"advanced": 1.00, "intermediate": 0.67, "beginner": 0.33}
DEMAND_THRESHOLD = 0.10
DEFAULT_ALPHA = 0.5


# --------------------------------------------------------------------------- model build
def build_model(demand_rows, careers, interests, skills, courses, threshold=DEMAND_THRESHOLD,
                soft_rows=None, soft_skills=None, course_categories=None):
    """demand_rows: iterable of dicts with career, skill, demand (0-1) [, n_ads].
    soft_rows: same for soft skills (advice only, never scored).
    Returns the model dict consumed by score() and by the web front end."""
    req = defaultdict(dict)
    for r in demand_rows:
        d = float(r["demand"])
        if d >= threshold:
            req[r["career"]][r["skill"]] = d
    soft_req = defaultdict(dict)
    for r in soft_rows or []:
        if float(r["demand"]) >= threshold:
            soft_req[r["career"]][r["skill"]] = float(r["demand"])
    n_careers = len(careers)
    m = defaultdict(int)
    for c in req.values():
        for s in c:
            m[s] += 1
    model_careers = []
    for c in careers:
        reqs = req.get(c["id"], {})
        items = []
        for s, d in reqs.items():
            w = d * math.log(1 + n_careers / m[s])
            items.append({"skill": s, "demand": round(d, 4), "weight": round(w, 6)})
        items.sort(key=lambda x: -x["weight"])
        soft = sorted(({"skill": s, "demand": round(d, 4)} for s, d in soft_req.get(c["id"], {}).items()),
                      key=lambda x: -x["demand"])
        model_careers.append({k: c[k] for k in ("id", "name_en", "name_id", "group")}
                             | {"requirements": items, "soft_skills": soft})
    return {
        "version": 1,
        "params": {"alpha": DEFAULT_ALPHA, "threshold": threshold, "n_careers": n_careers,
                   "grade_points": GRADE_POINTS, "self_level": SELF_LEVEL},
        "careers": model_careers,
        "interests": interests,
        "skills": [{k: s[k] for k in ("id", "name", "category", "patterns")} for s in skills],
        "soft_skills": soft_skills or [],
        "courses": courses,
        "course_categories": course_categories or [],
    }


def load_model_from_dir(data_dir: str | Path, demand_file="demand_seed.csv"):
    data_dir = Path(data_dir)
    j = lambda n: json.load(open(data_dir / n, encoding="utf-8")) if (data_dir / n).exists() else []
    rows = list(csv.DictReader(open(data_dir / demand_file, encoding="utf-8")))
    soft_file = data_dir / demand_file.replace("demand", "soft_demand", 1)
    soft_rows = list(csv.DictReader(open(soft_file, encoding="utf-8"))) if soft_file.exists() else []
    return build_model(rows, j("careers.json"), j("interests.json"), j("skills.json"), j("courses.json"),
                       soft_rows=soft_rows, soft_skills=j("soft_skills.json"), course_categories=j("course_categories.json"))


# --------------------------------------------------------------------------- scoring
def evidence(profile: dict, model: dict) -> dict[str, dict]:
    """Returns {skill: {"value": e(s), "source": "course|self|experience", "detail": str}}."""
    course_skills = {c["id"]: c for c in model["courses"]}
    ev: dict[str, dict] = {}

    def put(s, v, src, detail, course=None):
        if v > ev.get(s, {"value": -1})["value"]:
            ev[s] = {"value": v, "source": src, "detail": detail} | ({"course": course} if course else {})

    for cid, grade in (profile.get("courses") or {}).items():
        course = course_skills.get(cid)
        if not course:
            continue
        v = GRADE_POINTS.get(str(grade).upper(), 0.0)
        for s in course["skills"]:
            put(s, v, "course", f"{course['name_en']} ({grade})", cid)
    for s, level in (profile.get("skills") or {}).items():
        put(s, SELF_LEVEL.get(level, 0.0), "self", level)
    for s in profile.get("experience_skills") or []:
        put(s, 1.0, "experience", "confirmed from experience")
    return ev


def score(profile: dict, model: dict, alpha: float | None = None, use_interest: bool = True):
    """Rank all careers. Returns list of dicts sorted by fit (desc), ties broken by career order
    (repeatable: same input -> same ranking, NFR1)."""
    alpha = model["params"]["alpha"] if alpha is None else alpha
    ev = evidence(profile, model)
    chosen = set(profile.get("interests") or [])
    linked = {c for i in model["interests"] if i["id"] in chosen for c in i["careers"]}
    results = []
    for order, c in enumerate(model["careers"]):
        reqs = c["requirements"]
        tot = sum(r["weight"] for r in reqs) or 1.0
        k = sum(r["weight"] * ev.get(r["skill"], {"value": 0})["value"] for r in reqs) / tot
        m = 1.0 if c["id"] in linked else 0.0
        fit = 100 * (alpha * k + (1 - alpha) * m) if use_interest else 100 * k
        matched = [r | {"evidence": ev[r["skill"]]} for r in reqs if ev.get(r["skill"], {"value": 0})["value"] > 0]
        gaps = [r for r in reqs if ev.get(r["skill"], {"value": 0})["value"] == 0]
        own_soft = profile.get("soft_skills") or {}
        soft = [r | {"have": SELF_LEVEL.get(own_soft.get(r["skill"]), 0) >= SELF_LEVEL["intermediate"]}
                for r in c.get("soft_skills", [])]
        results.append({
            "career": c["id"], "name_en": c["name_en"], "name_id": c["name_id"], "group": c["group"],
            "fit": round(fit, 2), "K": round(k, 4), "M": m,
            "matched": matched, "gaps": gaps, "soft": soft, "_order": order,
        })
    results.sort(key=lambda r: (-r["fit"], r["_order"]))
    for i, r in enumerate(results):
        r["rank"] = i + 1
        r.pop("_order")
    return results


def recommend(profile: dict, model: dict, alpha: float | None = None, n_alt: int = 3, n_top: int = 5):
    """FR6: four graduate-profile careers ranked + top-3 IT alternatives (thesis), plus the top-3
    general (non-IT) careers and the overall top-5 across all careers.
    Soft skills never enter the score; each result carries `soft` for development advice."""
    ranked = score(profile, model, alpha)
    primary = [r for r in ranked if r["group"] == "primary"]
    alternative = [r for r in ranked if r["group"] == "alternative"][:n_alt]
    non_it = [r for r in ranked if r["group"] == "non_it"][:n_alt]
    return {"top": ranked[:n_top], "primary": primary, "alternative": alternative, "non_it": non_it, "all": ranked}


# --------------------------------------------------------------------------- baseline
def cosine_rank(profile_skills: set[str], model: dict):
    """Skill-overlap baseline: cosine between binary profile vector and binary R(c)."""
    out = []
    for order, c in enumerate(model["careers"]):
        R = {r["skill"] for r in c["requirements"]}
        inter = len(R & profile_skills)
        denom = math.sqrt(len(R)) * math.sqrt(len(profile_skills)) or 1.0
        out.append((-(inter / denom), order, c["id"]))
    out.sort()
    return [cid for _, _, cid in out]
