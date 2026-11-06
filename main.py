"""
KarirFit API (FastAPI) – application layer of the thesis architecture (Section 3.5.5).

  GET  /api/model            catalogue: careers, skills, courses, interests (for the UI)
  POST /api/extract-skills   LLM task 1 (fallback: dictionary matching)
  POST /api/recommend        deterministic ranking (FR6/FR7), never uses the LLM
  POST /api/explain          RAG + LLM explanation with citations (FR8); template fallback
  POST /api/rating           relevance rating 1-5 (FR9)
  POST /api/chat             career assistant (Groq), grounded in the user's results
  GET  /api/export           evaluation log export (FR10) – needs ADMIN_TOKEN

  With SUPABASE_URL set, every user endpoint needs a Supabase access token
  (Authorization: Bearer ...); the Supabase user id becomes the session id (auth.py).

  Evaluation (EVAL_MODE=1, Sections 3.6.4-3.6.6):
  POST /api/compare          normal vs alpha=1 ranking, random order, no labels
  POST /api/compare/choice   participant's preference
  GET  /api/sus  POST /api/sus   System Usability Scale items / answers
  Every explanation is checked by claims.py; with EVAL_MODE=1 the same prompt is also sent
  without passages (non-RAG twin), checked and logged but never shown.

  Administrator (X-Admin-Token, FR10):
  GET /api/admin/kb   POST /api/admin/kb/upload   DELETE /api/admin/kb/{file}
  POST /api/admin/kb/rebuild   GET /api/admin/kb/status

The Next.js front end (../frontend) calls these endpoints; set ALLOWED_ORIGINS to its URL.

Run:  uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
# career_model is found either next to main.py (stand-alone backend repo, e.g. on Render)
# or in ../model (full project layout)
sys.path.insert(0, str(HERE.parent / "model"))
sys.path.insert(0, str(HERE))
from career_model.explain import template_explanation  # noqa: E402
from career_model.extract import SkillExtractor  # noqa: E402
from career_model.scoring import recommend  # noqa: E402

try:  # load backend/.env before llm/rag read their settings
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
except ImportError:
    pass

import evaluation  # noqa: E402
from auth import REQUIRE_AUTH, current_user  # noqa: E402
import llm  # noqa: E402
from claims import check_explanation  # noqa: E402
from storage import Storage  # noqa: E402
from rag import Retriever  # noqa: E402

MODEL = json.load(open(HERE / "model.json", encoding="utf-8"))
NAMES = {s["id"]: s["name"] for s in MODEL["skills"]}
SOFT_NAMES = {"id": {s["id"]: s["name_id"] for s in MODEL.get("soft_skills", [])},
              "en": {s["id"]: s["name_en"] for s in MODEL.get("soft_skills", [])}}
EXTRACTOR = SkillExtractor(MODEL["skills"])
RETRIEVER = Retriever()
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
EVAL_MODE = os.getenv("EVAL_MODE", "0") == "1"
STORE = Storage()


def log(kind: str, session: str, payload: dict):
    try:
        STORE.log(kind, session, payload)
    except Exception as e:  # logging must never break a recommendation
        print("[log] failed:", e)


app = FastAPI(title="KarirFit API", version="1.0")
ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])


class Profile(BaseModel):
    courses: dict[str, str] = Field(default_factory=dict)        # course_id -> grade A-E
    skills: dict[str, str] = Field(default_factory=dict)         # skill_id -> beginner|intermediate|advanced
    soft_skills: dict[str, str] = Field(default_factory=dict)    # soft skill -> level (advice only, never scored)
    experience_skills: list[str] = Field(default_factory=list)   # confirmed skill ids
    interests: list[str] = Field(default_factory=list, max_length=30)   # any number of areas (UI: no limit)
    major: str | None = None                                     # e.g. "Akuntansi" (context only)
    target_career: str | None = None


class RecommendReq(BaseModel):
    profile: Profile
    alpha: float | None = Field(default=None, ge=0, le=1)   # evaluation only: honoured with X-Admin-Token
    include_all: bool = False                                # also return all 15 careers
    session: str = "anon"
    consent: bool = False   # only logged when the user consents (FR1 / Section 3.8)


class ExtractReq(BaseModel):
    text: str = Field(max_length=1500)


class ExplainReq(BaseModel):
    profile: Profile
    career: str
    alpha: float | None = Field(default=None, ge=0, le=1)   # evaluation only: honoured with X-Admin-Token
    lang: str = "id"
    session: str = "anon"
    consent: bool = False


class RatingReq(BaseModel):
    session: str
    career: str
    rating: int = Field(ge=1, le=5)


class CompareReq(BaseModel):
    profile: Profile
    session: str
    consent: bool = False


class ChoiceReq(BaseModel):
    session: str
    token: str
    choice: str          # "A" | "B" | "same"


class SusReq(BaseModel):
    session: str
    answers: list[int] = Field(min_length=10, max_length=10)


class ChatMessage(BaseModel):
    role: str
    content: str = Field(max_length=4000)


class ChatReq(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)
    profile: Profile
    career: str | None = None          # career the user is looking at (focus of retrieval)
    lang: str = "id"
    session: str = "anon"
    consent: bool = False


def who(user: dict | None, session: str, consent: bool) -> tuple[str, bool]:
    """Session id and consent: from the Supabase account when logged in, else from the request."""
    if user:
        return user["id"], consent or user.get("consent", False)
    return session, consent


def require_admin(token: str):
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(403, "admin token required")


def require_eval(consent: bool = True):
    if not EVAL_MODE:
        raise HTTPException(404, "evaluation mode is off")
    if not consent:
        raise HTTPException(400, "consent required")


def effective_alpha(requested: float | None, token: str) -> float | None:
    """alpha is fixed at 0.5 before testing (Section 3.5.2). Other values are only for the
    researcher's sensitivity analysis / alpha=1 comparison (Sections 3.6.3-3.6.4)."""
    if requested is None:
        return None
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(403, "alpha is fixed; custom alpha requires the admin token")
    return requested


@app.get("/api/model")
def get_model():
    return MODEL


@app.post("/api/extract-skills")
def extract_skills(req: ExtractReq, user: dict | None = Depends(current_user)):
    try:
        skills, model_id = llm.extract_skills(req.text, MODEL["skills"])
    except Exception as e:
        skills, model_id = EXTRACTOR.extract(req.text), f"dictionary ({e.__class__.__name__})"
    return {"skills": skills, "model": model_id}


@app.post("/api/recommend")
def do_recommend(req: RecommendReq, x_admin_token: str = Header(default=""), user: dict | None = Depends(current_user)):
    t0 = time.time()
    session, consent = who(user, req.session, req.consent)
    res = recommend(req.profile.model_dump(), MODEL, alpha=effective_alpha(req.alpha, x_admin_token))
    if not req.include_all:
        res.pop("all")
    if consent:
        log("recommend", session, {"profile": req.profile.model_dump(),
                                   "ranking": [r["career"] for r in res["primary"] + res["alternative"]],
                                   "top": [r["career"] for r in res["top"]]})
    return res | {"ms": round((time.time() - t0) * 1000, 1)}


def _check_summary(chk: dict) -> dict:
    return {k: chk[k] for k in ("n_claims", "n_supported", "evidence_support", "four_parts", "gaps_named", "gaps_correct")}


def _non_rag_twin(result: dict, profile: dict, lang: str, session: str, rag_passages: list[str]):
    """Section 3.5.6: same prompt without passages; checked and logged, never shown."""
    try:
        text, model_id = llm.explain(result, NAMES, [], lang, SOFT_NAMES.get(lang, SOFT_NAMES["en"]))
        chk = check_explanation(text, result, MODEL, [], profile, EXTRACTOR)
        log("explain_norag", session, {"career": result["career"], "condition": "non_rag", "model": model_id,
                                       "prompt_version": llm.PROMPT_VERSION, "text": text,
                                       "rag_passages": rag_passages, "check": _check_summary(chk), "claims": chk["claims"]})
    except Exception as e:
        log("explain_norag", session, {"career": result["career"], "condition": "non_rag", "error": str(e)[:200]})


@app.post("/api/explain")
def do_explain(req: ExplainReq, background: BackgroundTasks, x_admin_token: str = Header(default=""),
               user: dict | None = Depends(current_user)):
    session, consent = who(user, req.session, req.consent)
    soft_names = SOFT_NAMES.get(req.lang, SOFT_NAMES["en"])
    profile = req.profile.model_dump()
    ranked = recommend(profile, MODEL, alpha=effective_alpha(req.alpha, x_admin_token))["all"]
    result = next((r for r in ranked if r["career"] == req.career), None)
    if not result:
        raise HTTPException(404, "unknown career")
    passages = RETRIEVER.search(result["career"], [NAMES[r["skill"]] for r in result["matched"]],
                                [NAMES[r["skill"]] for r in result["gaps"]], result["name_en"])
    out = {"career": req.career, "passages": passages, "prompt_version": llm.PROMPT_VERSION}
    try:
        text, model_id = llm.explain(result, NAMES, passages, req.lang, soft_names)
        out |= {"text": text, "model": model_id, "mode": "rag" if passages else "llm_no_context"}
        if not passages:
            out["notice"] = ("Bukti dari basis pengetahuan tidak cukup; penjelasan hanya memakai data sistem."
                             if req.lang == "id" else
                             "Insufficient evidence in the knowledge base; explanation uses system data only.")
    except Exception as e:  # NFR6: ranking and skill lists remain available
        out |= {"text": template_explanation(result, NAMES, req.lang, soft_names), "model": "template",
                "mode": "template", "error": str(e)[:200]}
    chk = check_explanation(out["text"], result, MODEL, passages, profile, EXTRACTOR)
    out["check"] = _check_summary(chk)
    if consent:
        log("explain", session, {"career": req.career, "condition": "rag", "mode": out["mode"],
                                     "model": out["model"], "prompt_version": llm.PROMPT_VERSION,
                                     "passages": [p["id"] for p in passages], "text": out["text"],
                                     "check": out["check"], "claims": chk["claims"]})
        if EVAL_MODE and out["mode"] != "template":
            background.add_task(_non_rag_twin, result, profile, req.lang, session, [p["id"] for p in passages])
    return out


@app.post("/api/rating")
def rate(req: RatingReq, user: dict | None = Depends(current_user)):
    session, _ = who(user, req.session, True)
    log("rating", session, {"career": req.career, "rating": req.rating})
    return {"ok": True}


@app.post("/api/compare")
def compare(req: CompareReq, user: dict | None = Depends(current_user)):
    session, consent = who(user, req.session, req.consent)
    require_eval(consent)
    out = evaluation.make_comparison(req.profile.model_dump(), MODEL, session)
    log("compare_shown", session, {"same": out["same"]})
    return out


@app.post("/api/compare/choice")
def compare_choice(req: ChoiceReq, user: dict | None = Depends(current_user)):
    require_eval()
    session, _ = who(user, req.session, True)
    res = evaluation.resolve_choice(req.token, session, req.choice)
    if not res:
        raise HTTPException(400, "invalid comparison token")
    log("compare_choice", session, res)
    return {"ok": True}


@app.get("/api/sus")
def sus_items(lang: str = "id"):
    require_eval()
    return {"items": evaluation.SUS_ITEMS.get(lang, evaluation.SUS_ITEMS["en"])}


@app.post("/api/sus")
def sus_submit(req: SusReq, user: dict | None = Depends(current_user)):
    require_eval()
    req.session, _ = who(user, req.session, True)
    try:
        score = evaluation.sus_score(req.answers)
    except ValueError as e:
        raise HTTPException(400, str(e))
    log("sus", req.session, {"answers": req.answers, "score": score})
    return {"ok": True}


def _results_text(res: dict, lang: str, focus: dict | None) -> str:
    name = (lambda r: r["name_id"]) if lang == "id" else (lambda r: r["name_en"])
    soft = SOFT_NAMES.get(lang, SOFT_NAMES["en"])
    lines = ["Top careers (fit score 0-100): " + "; ".join(f"{name(r)} {r['fit']:.0f}" for r in res["top"])]
    lines.append("IT graduate-profile careers: " + "; ".join(f"{name(r)} {r['fit']:.0f}" for r in res["primary"]))
    for r in ([focus] if focus else []) + [x for x in res["top"][:2] if not focus or x["career"] != focus["career"]]:
        has = ", ".join(NAMES[m["skill"]] for m in r["matched"][:8]) or "none"
        todo = ", ".join("%s %.0f%%" % (NAMES[g["skill"]], g["demand"] * 100) for g in r["gaps"][:6]) or "none"
        soft_todo = ", ".join(soft.get(x["skill"], x["skill"]) for x in r.get("soft", []) if not x["have"]) or "none"
        lines.append("- %s: fit %.0f, competency %.0f%%, interest match %s; has: %s; to learn (most important first, "
                     "%% of job ads): %s; soft skills to develop: %s"
                     % (name(r), r["fit"], r["K"] * 100, "yes" if r["M"] else "no", has, todo, soft_todo))
    return "\n".join(lines)


@app.post("/api/chat")
def do_chat(req: ChatReq, user: dict | None = Depends(current_user)):
    session, consent = who(user, req.session, req.consent)
    p = req.profile
    tested = bool(p.interests or p.skills or p.courses or p.experience_skills)
    res = recommend(p.model_dump(), MODEL)
    focus = next((r for r in res["all"] if r["career"] == req.career), None) if req.career else None
    target = focus or res["top"][0]
    if tested or focus:
        passages = RETRIEVER.search(target["career"], [NAMES[r["skill"]] for r in target["matched"]],
                                    [NAMES[r["skill"]] for r in target["gaps"]], target["name_en"])
        results_text = _results_text(res, req.lang, focus)
    else:   # chat opened before the test: general career help, no personal scores exist yet
        passages = []
        name = (lambda c: c["name_id"]) if req.lang == "id" else (lambda c: c["name_en"])
        results_text = ("The user has NOT taken the test yet, so there are no personal scores. Answer general "
                        "questions about careers, skills and learning, and suggest taking the KarirFit test for a "
                        "personal recommendation. Careers covered by KarirFit: "
                        + ", ".join(name(c) for c in MODEL["careers"]))
    try:
        reply, model_id = llm.chat([m.model_dump() for m in req.messages], results_text, passages, req.lang)
    except Exception as e:
        raise HTTPException(503, f"assistant unavailable: {str(e)[:150]}")
    if consent:
        log("chat", session, {"career": target["career"], "question": req.messages[-1].content[:1000],
                              "answer": reply[:3000], "model": model_id, "passages": [p["id"] for p in passages]})
    return {"reply": reply, "model": model_id, "passages": [{"id": p["id"], "source": p["source"]} for p in passages]}


@app.get("/api/export")
def export(x_admin_token: str = Header(default="")):
    require_admin(x_admin_token)
    return STORE.export()


# ----------------------------------------------------------------------------- admin: knowledge base (FR10)
KB_DOCS = HERE / "kb" / "docs"
MAPPING = KB_DOCS / "mapping.csv"
MAP_FIELDS = ["file", "doc_type", "careers", "source", "access", "id_prefix"]
REBUILD = {"state": "idle", "message": "", "finished": None}


def _read_mapping() -> tuple[list[str], list[dict]]:
    if not MAPPING.exists():
        return [], []
    lines = MAPPING.read_text(encoding="utf-8-sig").splitlines()
    comments = [l for l in lines[1:] if l.startswith("#")]
    rows = [r for r in csv.DictReader(io.StringIO("\n".join([lines[0]] + [l for l in lines[1:] if l and not l.startswith("#")])))]
    return comments, rows


def _write_mapping(comments: list[str], rows: list[dict]):
    KB_DOCS.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=MAP_FIELDS, lineterminator="\n")
    w.writeheader()
    for c in comments:
        buf.write(c + "\n")
    for r in rows:
        w.writerow({k: r.get(k, "") for k in MAP_FIELDS})
    MAPPING.write_text(buf.getvalue(), encoding="utf-8")


@app.get("/api/admin/kb")
def kb_list(x_admin_token: str = Header(default="")):
    require_admin(x_admin_token)
    _, rows = _read_mapping()
    chunks = HERE / "kb" / "chunks.json"
    return {"documents": [r | {"present": (KB_DOCS / r["file"]).exists()} for r in rows],
            "index_ready": RETRIEVER.ready, "n_chunks": len(json.load(open(chunks))) if chunks.exists() else 0,
            "careers": [c["id"] for c in MODEL["careers"]], "rebuild": REBUILD,
            "note": "Render free: uploaded files are lost on restart; commit documents to the repo for permanence."}


@app.post("/api/admin/kb/upload")
async def kb_upload(file: UploadFile = File(...), doc_type: str = Form("skkni"), careers: str = Form(""),
                    source: str = Form(""), access: str = Form("public"), id_prefix: str = Form(""),
                    x_admin_token: str = Header(default="")):
    require_admin(x_admin_token)
    name = Path(file.filename or "").name
    if not name.lower().endswith((".pdf", ".docx", ".txt", ".md")):
        raise HTTPException(400, "only PDF, DOCX, TXT or MD")
    valid = {c["id"] for c in MODEL["careers"]}
    cs = [c.strip() for c in careers.replace(",", "|").split("|") if c.strip()]
    if not cs or set(cs) - valid:
        raise HTTPException(400, f"careers must be one or more of: {sorted(valid)}")
    if access not in ("public", "private") or doc_type not in ("skkni", "occupation_map", "rps", "course", "other"):
        raise HTTPException(400, "invalid access or doc_type")
    KB_DOCS.mkdir(parents=True, exist_ok=True)
    (KB_DOCS / name).write_bytes(await file.read())
    comments, rows = _read_mapping()
    rows = [r for r in rows if r["file"] != name] + [{"file": name, "doc_type": doc_type, "careers": "|".join(cs),
                                                      "source": source or Path(name).stem, "access": access,
                                                      "id_prefix": id_prefix}]
    _write_mapping(comments, rows)
    log("admin_kb", "admin", {"action": "upload", "file": name})
    return {"ok": True, "file": name}


@app.delete("/api/admin/kb/{filename}")
def kb_delete(filename: str, x_admin_token: str = Header(default="")):
    require_admin(x_admin_token)
    name = Path(filename).name
    comments, rows = _read_mapping()
    if not any(r["file"] == name for r in rows):
        raise HTTPException(404, "not in mapping")
    _write_mapping(comments, [r for r in rows if r["file"] != name])
    (KB_DOCS / name).unlink(missing_ok=True)
    log("admin_kb", "admin", {"action": "delete", "file": name})
    return {"ok": True}


def _rebuild():
    global RETRIEVER
    REBUILD.update(state="running", message="chunking documents…", finished=None)
    try:
        import build_rag
        chunks = build_rag.build_chunks()
        with open(HERE / "kb" / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        REBUILD["message"] = f"{len(chunks)} chunks; embedding…"
        from rag import build
        build(str(HERE / "kb" / "chunks.jsonl"))
        RETRIEVER = Retriever()
        REBUILD.update(state="done", message=f"{len(chunks)} chunks indexed; retrieval ready: {RETRIEVER.ready}")
    except ImportError as e:
        REBUILD.update(state="error", message=f"RAG libraries not installed on this server ({e}); build locally with build_rag.py")
    except Exception as e:
        REBUILD.update(state="error", message=str(e)[:300])
    REBUILD["finished"] = time.time()


@app.post("/api/admin/kb/rebuild")
def kb_rebuild(x_admin_token: str = Header(default="")):
    require_admin(x_admin_token)
    if REBUILD["state"] == "running":
        raise HTTPException(409, "rebuild already running")
    threading.Thread(target=_rebuild, daemon=True).start()
    log("admin_kb", "admin", {"action": "rebuild"})
    return {"ok": True}


@app.get("/api/admin/kb/status")
def kb_status(x_admin_token: str = Header(default="")):
    require_admin(x_admin_token)
    return REBUILD | {"index_ready": RETRIEVER.ready}


@app.get("/api/health")
def health():
    return {"ok": True, "rag": RETRIEVER.ready, "public_mode": os.getenv("PUBLIC_MODE", "1") == "1",
            "eval_mode": EVAL_MODE, "auth_required": REQUIRE_AUTH, "chat": bool(os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")),
            "llm_keys": {"gemini": bool(os.getenv("GEMINI_API_KEY")), "groq": bool(os.getenv("GROQ_API_KEY"))},
            "demand_source": MODEL.get("demand_source"), "data_label": MODEL.get("data_label"),
            "storage": STORE.kind, "storage_error": STORE.error}
