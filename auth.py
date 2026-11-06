"""
Supabase Auth check for the API (thesis FR1: register, log in and consent before data are stored).

The Next.js app signs users in with Supabase and sends the access token as
`Authorization: Bearer <token>`. The backend validates it by asking Supabase who the token belongs
to (GET {SUPABASE_URL}/auth/v1/user), which works for every Supabase JWT signing method, and caches
the answer for a few minutes. The Supabase user id is then used as the (pseudonymous) session id in
the evaluation log; names and e-mails are never sent to the LLM (NFR5).

Env: SUPABASE_URL, SUPABASE_ANON_KEY, REQUIRE_AUTH (default 1 when SUPABASE_URL is set).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from fastapi import Header, HTTPException

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "1" if SUPABASE_URL else "0") == "1"
_CACHE: dict[str, tuple[dict, float]] = {}
TTL = 300


def verify_token(token: str) -> dict | None:
    """Returns {"id", "email", "consent", ...} for a valid Supabase access token, else None."""
    if not token or not SUPABASE_URL:
        return None
    hit = _CACHE.get(token)
    if hit and hit[1] > time.time():
        return hit[0]
    req = urllib.request.Request(f"{SUPABASE_URL}/auth/v1/user",
                                 headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_ANON_KEY})
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
    except (urllib.error.URLError, ValueError):
        return None
    meta = data.get("user_metadata") or {}
    user = {"id": data.get("id"), "email": data.get("email"), "consent": bool(meta.get("consent"))}
    if not user["id"]:
        return None
    _CACHE[token] = (user, time.time() + TTL)
    if len(_CACHE) > 5000:
        _CACHE.clear()
    return user


def current_user(authorization: str = Header(default="")) -> dict | None:
    """FastAPI dependency. With REQUIRE_AUTH=1 a valid Supabase session is mandatory (401 otherwise);
    with REQUIRE_AUTH=0 (local development) it returns None and the request continues anonymously."""
    token = authorization[7:] if authorization.lower().startswith("bearer ") else ""
    user = verify_token(token)
    if REQUIRE_AUTH and not user:
        raise HTTPException(401, "login required")
    return user
