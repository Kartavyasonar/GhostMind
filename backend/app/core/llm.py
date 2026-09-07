"""LLM client v3 — multi-key rotation, per-key 429 cooldown, provider fallback.

WHY: Groq free tier = ~30 req/min. v2 pipeline makes several calls per query,
so a single key 429s constantly. This client:
  1. Rotates across ALL configured keys (GROQ_API_KEY, GROQ_API_KEY_2..5,
     GEMINI_API_KEY, GEMINI_API_KEY_1..10, OPENAI, ANTHROPIC).
  2. On 429, puts that key on a 30s cooldown and instantly tries the next key
     (no sleeping in front of the user).
  3. Makes a second pass after a 3s pause only if every key is exhausted.
  4. Uses only Python stdlib (urllib) — zero new dependencies, Render-safe.
"""
import asyncio
import json
import os
import time
import urllib.request
import urllib.error
from typing import Callable, List, Tuple
import structlog

from app.core.config import settings

log = structlog.get_logger()

class RateLimited(Exception):
    pass

class LLMUnavailable(Exception):
    pass

_KEY_COOLDOWN: dict = {}   # key -> epoch seconds until usable again
COOLDOWN_SECONDS = 30

def _on_cooldown(key: str) -> bool:
    return _KEY_COOLDOWN.get(key, 0) > time.time()

def _cool(key: str):
    _KEY_COOLDOWN[key] = time.time() + COOLDOWN_SECONDS

# ── Raw HTTP calls (stdlib) ────────────────────────────────────────────────

def _post_json(url: str, headers: dict, payload: dict, timeout: int = 90) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")[:300]
        if e.code == 429:
            raise RateLimited(body)
        raise RuntimeError(f"HTTP {e.code}: {body}")

def _call_groq(key, system, user, max_tokens, temperature):
    data = _post_json(
        "https://api.groq.com/openai/v1/chat/completions",
        {"Authorization": f"Bearer {key}"},
        {"model": settings.GROQ_MODEL, "temperature": temperature,
         "max_tokens": max_tokens,
         "messages": [{"role": "system", "content": system},
                      {"role": "user", "content": user}]})
    return data["choices"][0]["message"]["content"]

def _call_gemini(key, system, user, max_tokens, temperature):
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{settings.LLM_MODEL}:generateContent?key={key}")
    data = _post_json(url, {}, {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": temperature,
                             "maxOutputTokens": max_tokens}})
    return data["candidates"][0]["content"]["parts"][0]["text"]

def _call_openai(key, system, user, max_tokens, temperature):
    data = _post_json(
        "https://api.openai.com/v1/chat/completions",
        {"Authorization": f"Bearer {key}"},
        {"model": settings.OPENAI_MODEL, "temperature": temperature,
         "max_tokens": max_tokens,
         "messages": [{"role": "system", "content": system},
                      {"role": "user", "content": user}]})
    return data["choices"][0]["message"]["content"]

def _call_anthropic(key, system, user, max_tokens, temperature):
    data = _post_json(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": key, "anthropic-version": "2023-06-01"},
        {"model": settings.ANTHROPIC_MODEL, "max_tokens": max_tokens,
         "temperature": temperature, "system": system,
         "messages": [{"role": "user", "content": user}]})
    return data["content"][0]["text"]

# ── Key discovery ──────────────────────────────────────────────────────────

def _build_chain() -> List[Tuple[str, str, Callable]]:
    chain: List[Tuple[str, str, Callable]] = []
    # Groq keys (primary + rotations from env)
    if settings.GROQ_API_KEY:
        chain.append(("groq", settings.GROQ_API_KEY, _call_groq))
    for i in range(2, 6):
        k = os.getenv(f"GROQ_API_KEY_{i}", "")
        if k:
            chain.append((f"groq#{i}", k, _call_groq))
    # Gemini keys (fallback pool)
    gem_keys = [settings.GEMINI_API_KEY] + [
        getattr(settings, f"GEMINI_API_KEY_{i}", "") for i in range(1, 11)]
    for i, k in enumerate(gem_keys):
        if k:
            chain.append((f"gemini#{i}", k, _call_gemini))
    if settings.OPENAI_API_KEY:
        chain.append(("openai", settings.OPENAI_API_KEY, _call_openai))
    if settings.ANTHROPIC_API_KEY:
        chain.append(("anthropic", settings.ANTHROPIC_API_KEY, _call_anthropic))
    return chain

# ── Client ─────────────────────────────────────────────────────────────────

class _Client:
    def _complete_sync(self, system: str, user: str,
                       max_tokens: int, temperature: float) -> str:
        chain = _build_chain()
        if not chain:
            raise LLMUnavailable(
                "No LLM keys configured. Set GROQ_API_KEY in Render → Environment.")
        last_err = "no providers configured"
        for pass_no in range(2):                      # pass 2 = after brief pause
            for name, key, fn in chain:
                if _on_cooldown(key):
                    continue
                try:
                    return fn(key, system, user, max_tokens, temperature)
                except RateLimited:
                    _cool(key)
                    last_err = f"{name} rate-limited (429)"
                    log.warning("LLM 429, rotating key", provider=name)
                    continue
                except Exception as e:
                    last_err = f"{name}: {e}"
                    continue
            if pass_no == 0:
                time.sleep(3)                          # one short pause, then retry
        raise LLMUnavailable(
            f"All LLM providers unavailable ({last_err}). "
            "Add more keys: GROQ_API_KEY_2 / GEMINI_API_KEY in Render → Environment.")

    async def complete(self, system: str, user: str,
                       max_tokens: int = 512, temperature: float = 0.2) -> str:
        return await asyncio.to_thread(
            self._complete_sync, system, user, max_tokens, temperature)

_client = _Client()

def get_llm(*_args, **_kwargs) -> _Client:
    """Singleton client; temperature is passed per-call."""
    return _client
