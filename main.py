"""
KarirFit API (FastAPI) – application layer of the thesis architecture (Section 3.5.5).

  GET  /api/model            catalogue: careers, skills, courses, interests (for the UI)
  POST /api/extract-skills   LLM task 1 (fallback: dictionary matching)
  POST /api/recommend        deterministic ranking (FR6/FR7), never uses the LLM
  POST /api/explain          RAG + LLM explanation with citations (FR8); template fallback
  POST /api/rating           relevance rating 1-5 (FR9)
  GET  /api/export           evaluation log export (FR10) – needs ADMIN_TOKEN

The Next.js front end (../frontend) calls these endpoints; set ALLOWED_ORIGINS to its URL.

Run:  uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model"))
from career_model.explain import template_explanation  # noqa: E402
from career_model.extract import SkillExtractor  # noqa: E402
from career_model.scoring import recommend  # noqa: E402

try:  # load backend/.env before llm/rag read their settings
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
except ImportError:
    pass

import llm  # noqa: E402
from rag import Retriever  # noqa: E402

MODEL = json.load(open(HERE / "model.json", encoding="utf-8"))
NAMES = {s["id"]: s["name"] for s in MODEL["skills"]}
EXTRACTOR = SkillExtractor(MODEL["skills"])
RETRIEVER = Retriever()
DB_PATH = os.getenv("DB_PATH", str(HERE / "karirfit.db"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")


def db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS log(id INTEGER PRIMARY KEY, ts REAL, kind TEXT,
                   session TEXT, payload TEXT)""")
    return con


def log(kind: str, session: str, payload: dict):
    with db() as con:
        con.execute("INSERT INTO log(ts, kind, session, payload) VALUES (?,?,?,?)",
                    (time.time(), kind, session, json.dumps(payload, ensure_ascii=False)))


app = FastAPI(title="KarirFit API", version="1.0")
ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])


class Profile(BaseModel):
    courses: dict[str, str] = Field(default_factory=dict)        # course_id -> grade A-E
    skills: dict[str, str] = Field(default_factory=dict)         # skill_id -> beginner|intermediate|advanced
    experience_skills: list[str] = Field(default_factory=list)   # confirmed skill ids
    interests: list[str] = Field(default_factory=list, max_length=3)
    target_career: str | None = None


class RecommendReq(BaseModel):
    profile: Profile
    alpha: float | None = Field(default=None, ge=0, le=1)   # sensitivity view; default 0.5
    include_all: bool = False                                # also return all 15 careers
    session: str = "anon"
    consent: bool = False   # only logged when the user consents (FR1 / Section 3.8)


class ExtractReq(BaseModel):
    text: str = Field(max_length=1500)


class ExplainReq(BaseModel):
    profile: Profile
    career: str
    alpha: float | None = Field(default=None, ge=0, le=1)
    lang: str = "id"
    session: str = "anon"
    consent: bool = False


class RatingReq(BaseModel):
    session: str
    career: str
    rating: int = Field(ge=1, le=5)


@app.get("/api/model")
def get_model():
    return MODEL


@app.post("/api/extract-skills")
def extract_skills(req: ExtractReq):
    try:
        skills, model_id = llm.extract_skills(req.text, MODEL["skills"])
    except Exception as e:
        skills, model_id = EXTRACTOR.extract(req.text), f"dictionary ({e.__class__.__name__})"
    return {"skills": skills, "model": model_id}


@app.post("/api/recommend")
def do_recommend(req: RecommendReq):
    t0 = time.time()
    res = recommend(req.profile.model_dump(), MODEL, alpha=req.alpha)
    if not req.include_all:
        res.pop("all")
    if req.consent:
        log("recommend", req.session, {"profile": req.profile.model_dump(),
                                       "ranking": [r["career"] for r in res["primary"] + res["alternative"]]})
    return res | {"ms": round((time.time() - t0) * 1000, 1)}


@app.post("/api/explain")
def do_explain(req: ExplainReq):
    ranked = recommend(req.profile.model_dump(), MODEL, alpha=req.alpha)["all"]
    result = next((r for r in ranked if r["career"] == req.career), None)
    if not result:
        raise HTTPException(404, "unknown career")
    passages = RETRIEVER.search(result["career"], [NAMES[r["skill"]] for r in result["matched"]],
                                [NAMES[r["skill"]] for r in result["gaps"]], result["name_en"])
    out = {"career": req.career, "passages": passages, "prompt_version": llm.PROMPT_VERSION}
    try:
        text, model_id = llm.explain(result, NAMES, passages, req.lang)
        out |= {"text": text, "model": model_id, "mode": "rag" if passages else "llm_no_context",
                "check": llm.check_citations(text, passages)}
        if not passages:
            out["notice"] = ("Bukti dari basis pengetahuan tidak cukup; penjelasan hanya memakai data sistem."
                             if req.lang == "id" else
                             "Insufficient evidence in the knowledge base; explanation uses system data only.")
    except Exception as e:  # NFR6: ranking and skill lists remain available
        out |= {"text": template_explanation(result, NAMES, req.lang), "model": "template",
                "mode": "template", "error": str(e)[:200]}
    if req.consent:
        log("explain", req.session, {"career": req.career, "mode": out["mode"], "model": out["model"],
                                     "passages": [p["id"] for p in passages], "check": out.get("check")})
    return out


@app.post("/api/rating")
def rate(req: RatingReq):
    log("rating", req.session, {"career": req.career, "rating": req.rating})
    return {"ok": True}


@app.get("/api/export")
def export(x_admin_token: str = Header(default="")):
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(403)
    with db() as con:
        rows = con.execute("SELECT ts, kind, session, payload FROM log ORDER BY id").fetchall()
    return [{"ts": t, "kind": k, "session": s, **json.loads(p)} for t, k, s, p in rows]


@app.get("/api/health")
def health():
    return {"ok": True, "rag": RETRIEVER.ready, "public_mode": os.getenv("PUBLIC_MODE", "1") == "1",
            "llm_keys": {"gemini": bool(os.getenv("GEMINI_API_KEY")), "groq": bool(os.getenv("GROQ_API_KEY"))},
            "demand_source": MODEL.get("demand_source")}

