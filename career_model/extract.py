"""Dictionary-based skill extraction (used for job ads, and as the no-LLM fallback for the
experience text). Only dictionary skills are returned, each with the phrase that supports it
(same output contract as LLM task 1 in Table 3.11)."""
from __future__ import annotations

import re


class SkillExtractor:
    def __init__(self, skills: list[dict]):
        self.compiled = [(s["id"], [re.compile(p, re.I) for p in s["patterns"]]) for s in skills]

    def extract(self, text: str) -> list[dict]:
        text = (text or "").lower()
        found = []
        for sid, pats in self.compiled:
            for p in pats:
                m = p.search(text)
                if m:
                    a, b = max(0, m.start() - 30), min(len(text), m.end() + 30)
                    phrase = text[a:b]
                    if a > 0:
                        phrase = re.sub(r"^\S*\s", "", phrase)   # drop cut-off first word
                    if b < len(text):
                        phrase = re.sub(r"\s\S*$", "", phrase)   # drop cut-off last word
                    found.append({"skill": sid, "phrase": phrase.strip()})
                    break
        return found

    def skill_set(self, text: str) -> set[str]:
        return {f["skill"] for f in self.extract(text)}


class CareerClassifier:
    """Classifies a job-ad title into one of the careers by title patterns (Section 3.2.2).
    Order matters: Senior Programmer is checked before Software Developer, specific roles
    before generic ones."""

    PRIORITY = ["mobile_developer", "data_engineer", "data_scientist", "data_analyst",
                "database_administrator", "devops_cloud", "cyber_security", "qa_tester",
                "ui_ux_designer", "system_analyst", "network_sysadmin", "it_support", "it_pm",
                "senior_programmer", "software_developer"]

    def __init__(self, careers: list[dict]):
        by_id = {c["id"]: c for c in careers}
        self.rules = [(cid, [re.compile(p, re.I) for p in by_id[cid]["title_patterns"]])
                      for cid in self.PRIORITY if cid in by_id]
        self.non_it = [re.compile(p, re.I) for p in careers[0].get("non_it_title", [])]

    def classify(self, title: str) -> str | None:
        t = (title or "").lower()
        if any(p.search(t) for p in self.non_it):
            return None
        for cid, pats in self.rules:
            if any(p.search(t) for p in pats):
                return cid
        return None
