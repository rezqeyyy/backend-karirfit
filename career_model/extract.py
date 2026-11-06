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
    IT careers are checked first (specific roles before generic ones, Senior Programmer before
    Software Developer), then general careers. A title containing a non-IT word (NON_IT_TITLE,
    e.g. 'supplier', 'sales', 'accounting') is never assigned to an IT career."""

    IT_PRIORITY = ["mobile_developer", "data_engineer", "data_scientist", "data_analyst",
                   "database_administrator", "devops_cloud", "cyber_security", "qa_tester",
                   "ui_ux_designer", "system_analyst", "network_sysadmin", "it_support", "it_pm",
                   "senior_programmer", "software_developer"]
    GENERAL_PRIORITY = ["tax_staff", "accountant", "financial_analyst", "digital_marketing", "content_writer",
                        "graphic_designer", "hr_recruiter", "teacher", "supply_chain", "civil_engineer",
                        "electrical_engineer", "qc_production", "customer_service", "sales_bd", "admin_staff"]

    def __init__(self, careers: list[dict]):
        by_id = {c["id"]: c for c in careers}
        compile_ = lambda cid: (cid, [re.compile(p, re.I) for p in by_id[cid]["title_patterns"]])
        it_ids = [c for c in self.IT_PRIORITY if c in by_id]
        general = [c for c in self.GENERAL_PRIORITY if c in by_id]
        others = [c for c in by_id if c not in it_ids and c not in general]   # careers added later
        self.it_rules = [compile_(c) for c in it_ids]
        self.general_rules = [compile_(c) for c in general + others]
        self.non_it = [re.compile(p, re.I) for p in careers[0].get("non_it_title", [])] if careers else []

    def classify(self, title: str) -> str | None:
        t = (title or "").lower()
        rules = self.general_rules if any(p.search(t) for p in self.non_it) else self.it_rules + self.general_rules
        for cid, pats in rules:
            if any(p.search(t) for p in pats):
                return cid
        return None
