"""
LLM tasks (thesis Section 3.5.3, Table 3.11). The LLM never computes or changes a score.

  Task 1  extract_skills(text)          -> [{"skill", "phrase"}]   (dictionary skills only)
  Task 2  explain(career, ..., passages) -> four-part explanation citing passage ids

Primary: Gemini Flash (GEMINI_API_KEY), temperature 0, versioned prompt.
Fallback: Groq open-weight model (GROQ_API_KEY) – keeps the app available, not used for evaluation.
No name / e-mail / student number is ever sent (NFR5): only experience text, scores, skill
lists and retrieved passages.
Uses only the standard library (urllib) so no SDK is required.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

PROMPT_VERSION = "v1.0"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TIMEOUT = 30


class LLMUnavailable(RuntimeError):
    pass


def _post(url, payload, headers):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def _gemini(prompt: str, json_mode: bool) -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise LLMUnavailable("GEMINI_API_KEY not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={key}"
    cfg = {"temperature": 0}
    if json_mode:
        cfg["responseMimeType"] = "application/json"
    out = _post(url, {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": cfg},
                {"Content-Type": "application/json"})
    return out["candidates"][0]["content"]["parts"][0]["text"]


def _groq(prompt: str, json_mode: bool) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise LLMUnavailable("GROQ_API_KEY not set")
    body = {"model": GROQ_MODEL, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    out = _post("https://api.groq.com/openai/v1/chat/completions", body,
                {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    return out["choices"][0]["message"]["content"]


def complete(prompt: str, json_mode=False) -> tuple[str, str]:
    """Returns (text, model_id). Raises LLMUnavailable if no provider works."""
    errors = []
    for name, fn, model in (("gemini", _gemini, GEMINI_MODEL), ("groq", _groq, GROQ_MODEL)):
        try:
            return fn(prompt, json_mode), f"{name}:{model}"
        except Exception as e:  # network, quota, missing key
            errors.append(f"{name}: {e}")
    raise LLMUnavailable("; ".join(errors))


# --------------------------------------------------------------------------- task 1
EXTRACT_PROMPT = """You extract technical skills from a person's description of their experience
(projects, internships, jobs, certificates). Use ONLY skill ids from this dictionary:
{dictionary}

Return JSON: {{"skills": [{{"skill": "<id>", "phrase": "<exact words from the text that show it>"}}]}}
Include a skill only if the text clearly shows the person used it. Do not guess.

Text:
\"\"\"{text}\"\"\""""


def extract_skills(text: str, skills: list[dict]) -> tuple[list[dict], str]:
    dictionary = ", ".join(f"{s['id']} ({s['name']})" for s in skills)
    raw, model_id = complete(EXTRACT_PROMPT.format(dictionary=dictionary, text=text[:1500]), json_mode=True)
    valid = {s["id"] for s in skills}
    data = json.loads(re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M))
    out, seen = [], set()
    for item in data.get("skills", []):
        sid = item.get("skill")
        if sid in valid and sid not in seen:  # safeguard: dictionary skills only
            seen.add(sid)
            out.append({"skill": sid, "phrase": str(item.get("phrase", ""))[:200]})
    return out, model_id


# --------------------------------------------------------------------------- task 2
EXPLAIN_PROMPT = """You are a career guidance assistant. Write in {language}.
Explain why the career "{career}" was recommended to this person. Use ONLY the facts below and
the CONTEXT passages. Cite passages with their id in square brackets, e.g. [{example_id}].
If something is not supported by the facts or context, do not say it.

FACTS (computed by the system, do not change them):
- Fit score: {fit:.0f}/100 (competency fit {k:.0%}, interest match: {interest})
- Matched skills (skill: demand in job ads, evidence): {matched}
- Skill gaps ordered by importance (skill: demand in job ads): {gaps}
- Soft skills often requested for this career that the person has not marked as strong (advice only): {soft}

CONTEXT:
{context}

Answer in exactly four short sections with these headings:
1. {h1}
2. {h2}
3. {h3}
4. {h4}
"""

HEADINGS = {
    "id": ("Dasar kecocokan", "Kompetensi yang sudah sesuai", "Kompetensi yang perlu ditingkatkan", "Saran pengembangan"),
    "en": ("Basis of fit", "Matched competencies", "Competencies to improve", "Development advice"),
}


def explain(result: dict, names: dict, passages: list[dict], lang="id", soft_names: dict | None = None) -> tuple[str, str]:
    fmt = lambda r: f"{names.get(r['skill'], r['skill'])}: {r['demand']:.0%}"
    matched = "; ".join(f"{fmt(r)}, {r['evidence']['source']}" for r in result["matched"]) or "none"
    gaps = "; ".join(fmt(r) for r in result["gaps"][:6]) or "none"
    soft_names = soft_names or {}
    soft = ", ".join(soft_names.get(r["skill"], r["skill"]) for r in result.get("soft", []) if not r.get("have"))[:300] or "none"
    context = "\n\n".join(f"[{p['id']}] ({p['source']}) {p['text']}" for p in passages) or "(no passages)"
    h = HEADINGS.get(lang, HEADINGS["en"])
    prompt = EXPLAIN_PROMPT.format(
        language="Indonesian" if lang == "id" else "English",
        career=result["name_id"] if lang == "id" else result["name_en"],
        example_id=passages[0]["id"] if passages else "P1",
        fit=result["fit"], k=result["K"], interest="yes" if result["M"] else "no",
        matched=matched, gaps=gaps, soft=soft, context=context, h1=h[0], h2=h[1], h3=h[2], h4=h[3])
    return complete(prompt)


def check_citations(text: str, passages: list[dict]) -> dict:
    """Lightweight automatic check: every cited id must exist among retrieved passages."""
    cited = set(re.findall(r"\[([A-Za-z0-9_\-:.]+)\]", text))
    known = {p["id"] for p in passages}
    return {"cited": sorted(cited), "unknown_citations": sorted(cited - known),
            "supported_ratio": (len(cited & known) / len(cited)) if cited else None}


# --------------------------------------------------------------------------- career assistant chat
CHAT_SYSTEM = """You are KarirFit's career assistant. Answer in {language}, briefly and concretely.
Help the person understand their career test results and plan how to learn the skills they lack.
Rules:
- Use the RESULTS and CONTEXT below for anything about their scores, skills or job-ad demand. Never
  invent scores, percentages, salaries or statistics; if the data is not here, say you do not know.
- When you use a CONTEXT passage, cite its id in square brackets, e.g. [{example_id}].
- You may give general study advice (what to learn first, project ideas, kinds of courses or
  certifications) but do not claim specific prices, dates or that a particular course exists.
- Stay on careers, skills, learning and job searching. Politely decline unrelated requests.
- The ranking was computed by a fixed formula; you cannot change it. Recommendations are
  suggestions to explore, not decisions.

RESULTS (computed by the system):
{results}

CONTEXT:
{context}"""


def chat(messages: list[dict], results_text: str, passages: list[dict], lang: str = "id") -> tuple[str, str]:
    """Groq chat completion (GROQ_MODEL); falls back to Gemini if Groq is unavailable."""
    context = "\n\n".join(f"[{p['id']}] ({p['source']}) {p['text'][:900]}" for p in passages) or "(no passages)"
    system = CHAT_SYSTEM.format(language="Indonesian" if lang == "id" else "English",
                                example_id=passages[0]["id"] if passages else "P1",
                                results=results_text, context=context)
    history = [{"role": m["role"], "content": str(m["content"])[:2000]}
               for m in messages[-12:] if m.get("role") in ("user", "assistant")]
    key = os.getenv("GROQ_API_KEY")
    if key:
        try:
            out = _post("https://api.groq.com/openai/v1/chat/completions",
                        {"model": GROQ_MODEL, "temperature": 0.3, "max_tokens": 700,
                         "messages": [{"role": "system", "content": system}] + history},
                        {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            return out["choices"][0]["message"]["content"], f"groq:{GROQ_MODEL}"
        except Exception as e:
            err = e
    else:
        err = LLMUnavailable("GROQ_API_KEY not set")
    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
    try:   # fallback keeps the assistant available
        text = _gemini(system + "\n\nCONVERSATION:\n" + transcript + "\nASSISTANT:", json_mode=False)
        return text, f"gemini:{GEMINI_MODEL}"
    except Exception as e2:
        raise LLMUnavailable(f"groq: {err}; gemini: {e2}")
