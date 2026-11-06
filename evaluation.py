"""
Helpers for the user evaluation (thesis Sections 3.6.4 and 3.6.6). Active only when EVAL_MODE=1.

  * A/B comparison: each participant sees the normal ranking (alpha=0.5) and the ranking without
    interest fit (alpha=1) side by side, in random order and without labels, and picks the one that
    fits them better. The order is carried in an HMAC-signed token so the client cannot see or
    forge which side is which.
  * SUS: standard 10-item System Usability Scale, score 0-100 (Brooke, 1996).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets

from career_model.scoring import recommend

_SECRET = (os.getenv("EVAL_SECRET") or os.getenv("ADMIN_TOKEN") or secrets.token_hex(16)).encode()
VARIANTS = ("normal", "no_interest")   # alpha 0.5 vs alpha 1


def _sign(payload: str) -> str:
    mac = hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()[:16]
    return base64.urlsafe_b64encode(f"{payload}|{mac}".encode()).decode()


def _unsign(token: str) -> str | None:
    try:
        payload, mac = base64.urlsafe_b64decode(token.encode()).decode().rsplit("|", 1)
    except Exception:
        return None
    good = hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()[:16]
    return payload if hmac.compare_digest(mac, good) else None


def _short(res: dict) -> dict:
    pick = lambda rs: [{"career": r["career"], "name_id": r["name_id"], "name_en": r["name_en"]} for r in rs]
    return {"primary": pick(res["primary"]), "alternative": pick(res["alternative"])}


def make_comparison(profile: dict, model: dict, session: str) -> dict:
    """Returns {"A": ranking, "B": ranking, "token": signed order, "same": bool} without labels."""
    normal = _short(recommend(profile, model, alpha=model["params"]["alpha"]))
    no_int = _short(recommend(profile, model, alpha=1.0))
    order = list(VARIANTS)
    if secrets.randbelow(2):
        order.reverse()
    ranks = {"normal": normal, "no_interest": no_int}
    token = _sign(f"{session}:{order[0]}:{order[1]}")
    return {"A": ranks[order[0]], "B": ranks[order[1]], "token": token, "same": normal == no_int}


def resolve_choice(token: str, session: str, choice: str) -> dict | None:
    """Maps the participant's choice ('A', 'B' or 'same') back to the variant."""
    payload = _unsign(token)
    if not payload:
        return None
    sess, a, b = payload.split(":")
    if sess != session or choice not in ("A", "B", "same"):
        return None
    chosen = {"A": a, "B": b}.get(choice, "no_preference")
    return {"order": [a, b], "choice": choice, "preferred": chosen, "prefers_normal": chosen == "normal"}


SUS_ITEMS = {
    "id": [
        "Saya rasa saya ingin sering menggunakan sistem ini.",
        "Saya merasa sistem ini terlalu rumit tanpa alasan.",
        "Saya merasa sistem ini mudah digunakan.",
        "Saya rasa saya butuh bantuan orang teknis untuk bisa menggunakan sistem ini.",
        "Saya merasa berbagai fungsi dalam sistem ini terintegrasi dengan baik.",
        "Saya merasa ada terlalu banyak ketidakkonsistenan dalam sistem ini.",
        "Saya rasa kebanyakan orang akan cepat belajar menggunakan sistem ini.",
        "Saya merasa sistem ini sangat merepotkan untuk digunakan.",
        "Saya merasa sangat percaya diri menggunakan sistem ini.",
        "Saya perlu mempelajari banyak hal sebelum bisa menggunakan sistem ini.",
    ],
    "en": [
        "I think that I would like to use this system frequently.",
        "I found the system unnecessarily complex.",
        "I thought the system was easy to use.",
        "I think that I would need the support of a technical person to be able to use this system.",
        "I found the various functions in this system were well integrated.",
        "I thought there was too much inconsistency in this system.",
        "I would imagine that most people would learn to use this system very quickly.",
        "I found the system very cumbersome to use.",
        "I felt very confident using the system.",
        "I needed to learn a lot of things before I could get going with this system.",
    ],
}


def sus_score(answers: list[int]) -> float:
    """answers: 10 values 1-5 in item order. Odd items: a-1, even items: 5-a; sum x 2.5."""
    if len(answers) != 10 or any(a not in (1, 2, 3, 4, 5) for a in answers):
        raise ValueError("SUS needs 10 answers between 1 and 5")
    return 2.5 * sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(answers))
