"""
GhostMind LLM Client — Multi-provider round-robin with automatic fallback.

Strategy:
  - Distributes calls across ALL configured providers + models in rotation
  - On quota/rate-limit: skips to next provider immediately (no waiting)
  - On transient rate limit (per-minute): waits the hint delay, then continues
  - Providers exhausted: raises clearly so the agent can return a graceful error

Provider priority & free tiers:
  GEMINI_API_KEY_1..N  — gemini-2.0-flash  15 req/min, 1500 req/day per key
  GEMINI_API_KEY_1..N  — gemini-1.5-flash  15 req/min, 1500 req/day per key
  OPENAI_API_KEY       — gpt-4o-mini       (paid, fallback)
  ANTHROPIC_API_KEY    — claude-haiku      (paid, fallback)

Add multiple Gemini keys in .env like:
  GEMINI_API_KEY_1=AIza...
  GEMINI_API_KEY_2=AIza...
  GEMINI_API_KEY_3=AIza...
"""
import asyncio
import re
import time
from dataclasses import dataclass
from typing import List, Optional
import structlog

from app.core.config import settings

log = structlog.get_logger()

DAILY_QUOTA_COOLDOWN = 3600  # seconds before retrying a daily-exhausted provider


@dataclass
class Provider:
    name: str
    backend: str   # "gemini" | "openai" | "anthropic"
    api_key: str
    model: str
    daily_exhausted_at: float = 0.0
    consecutive_failures: int = 0

    def is_daily_exhausted(self) -> bool:
        if self.daily_exhausted_at == 0.0:
            return False
        return (time.time() - self.daily_exhausted_at) < DAILY_QUOTA_COOLDOWN

    def mark_daily_exhausted(self):
        self.daily_exhausted_at = time.time()
        log.warning("Provider daily-quota exhausted, cooling down",
                    provider=self.name, model=self.model)

    def reset(self):
        self.consecutive_failures = 0
        self.daily_exhausted_at = 0.0


def _parse_retry_delay(err: str) -> Optional[float]:
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", err)
    if m:
        return float(m.group(1)) + 2.0
    m = re.search(r"retry in\s+([\d.]+)s", err, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 2.0
    m = re.search(r"Please try again in ([\d.]+)s", err, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 2.0
    return None


def _is_daily_quota(err: str) -> bool:
    signals = [
        "GenerateRequestsPerDayPerProjectPerModel",
        "per_day", "daily",
        "insufficient_quota",
        "credit balance is too low",
        "limit: 0",
    ]
    lower = err.lower()
    return any(s.lower() in lower for s in signals)


def _is_rate_limit(err: str) -> bool:
    signals = ["429", "rate", "quota", "retry", "resource_exhausted",
               "too many requests", "overloaded"]
    lower = err.lower()
    return any(s in lower for s in signals)


class LLMClient:
    """Round-robin multi-provider LLM client with automatic quota fallback."""

    def __init__(self, providers: List[Provider]):
        self._providers = providers
        self._index = 0
        names = [f"{p.name}({p.model})" for p in providers]
        log.info("LLM multi-provider ready", providers=names, count=len(providers))

    def _next_available(self) -> Optional[Provider]:
        n = len(self._providers)
        for _ in range(n):
            p = self._providers[self._index % n]
            self._index = (self._index + 1) % n
            if not p.is_daily_exhausted():
                return p
        return None

    async def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        n = len(self._providers)

        for _ in range(n * 3):  # generous loop to handle waits + retries
            provider = self._next_available()
            if provider is None:
                raise RuntimeError(
                    "All LLM providers have exhausted their daily quota. "
                    "Solutions: add more GEMINI_API_KEY_N keys in .env, "
                    "wait for quota reset (~midnight PT), or add credits."
                )

            try:
                result = await self._call(provider, system, user, max_tokens, temperature)
                provider.reset()
                log.info("LLM success", provider=provider.name, model=provider.model)
                return result

            except Exception as e:
                err = str(e)
                provider.consecutive_failures += 1

                if _is_daily_quota(err):
                    provider.mark_daily_exhausted()
                    continue  # skip immediately to next provider

                elif _is_rate_limit(err):
                    wait = _parse_retry_delay(err) or 20.0
                    log.warning("Rate limited — waiting, then trying next provider",
                                provider=provider.name, wait_s=round(wait, 1))
                    await asyncio.sleep(wait)
                    continue

                else:
                    log.error("Provider hard error, skipping",
                              provider=provider.name, error=err[:300])
                    provider.mark_daily_exhausted()  # treat as unusable
                    continue

        raise RuntimeError("All LLM providers failed after exhausting all retries.")

    async def _call(
        self, p: Provider, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        if p.backend == "gemini":
            return await self._gemini(p, system, user, max_tokens, temperature)
        if p.backend == "openai":
            return await self._openai(p, system, user, max_tokens, temperature)
        if p.backend == "anthropic":
            return await self._anthropic(p, system, user, max_tokens, temperature)
        raise ValueError(f"Unknown backend: {p.backend}")

    async def _gemini(self, p, system, user, max_tokens, temperature):
        import google.generativeai as genai
        genai.configure(api_key=p.api_key)
        model = genai.GenerativeModel(p.model, system_instruction=system)
        cfg = genai.types.GenerationConfig(
            max_output_tokens=max_tokens, temperature=temperature
        )
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: model.generate_content(user, generation_config=cfg)
        )
        return response.text

    async def _openai(self, p, system, user, max_tokens, temperature):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=p.api_key)
        resp = await client.chat.completions.create(
            model=p.model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""

    async def _anthropic(self, p, system, user, max_tokens, temperature):
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=p.api_key)
        resp = await client.messages.create(
            model=p.model, max_tokens=max_tokens, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def _build_providers() -> List[Provider]:
    providers: List[Provider] = []

    # Gemini — collect all numbered keys + plain key
    gemini_keys: List[str] = []
    plain = getattr(settings, "GEMINI_API_KEY", "") or ""
    if plain.strip():
        gemini_keys.append(plain.strip())
    for i in range(1, 11):
        key = getattr(settings, f"GEMINI_API_KEY_{i}", "") or ""
        if key.strip() and key.strip() not in gemini_keys:
            gemini_keys.append(key.strip())

    # Register each Gemini key with TWO models (doubles daily capacity per key)
    for i, key in enumerate(gemini_keys, start=1):
        providers.append(Provider(
            name=f"gemini-{i}-flash2",
            backend="gemini", api_key=key, model="gemini-2.0-flash",
        ))
        providers.append(Provider(
            name=f"gemini-{i}-flash15",
            backend="gemini", api_key=key, model="gemini-1.5-flash",
        ))

    # OpenAI
    openai_key = getattr(settings, "OPENAI_API_KEY", "") or ""
    if openai_key.strip():
        model = getattr(settings, "OPENAI_MODEL", None) or "gpt-4o-mini"
        providers.append(Provider(
            name="openai", backend="openai",
            api_key=openai_key.strip(), model=model,
        ))

    # Anthropic
    anthropic_key = getattr(settings, "ANTHROPIC_API_KEY", "") or ""
    if anthropic_key.strip():
        model = getattr(settings, "ANTHROPIC_MODEL", None) or "claude-haiku-4-5-20251001"
        providers.append(Provider(
            name="anthropic", backend="anthropic",
            api_key=anthropic_key.strip(), model=model,
        ))

    if not providers:
        raise RuntimeError(
            "No LLM providers configured! Set at least one of:\n"
            "  GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY\n"
            "in your .env file."
        )

    return providers


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_llm_client: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient(_build_providers())
    return _llm_client


def reset_llm():
    global _llm_client
    _llm_client = None
