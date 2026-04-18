"""Application configuration via environment variables."""
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    ENV: str = "development"
    DATABASE_URL: str = "sqlite+aiosqlite:///./ghostmind.db"

    # ── Groq (free tier, no daily quota — RECOMMENDED primary provider) ──
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # ── LLM API Keys ──────────────────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_API_KEY_1: str = ""
    GEMINI_API_KEY_2: str = ""
    GEMINI_API_KEY_3: str = ""
    GEMINI_API_KEY_4: str = ""
    GEMINI_API_KEY_5: str = ""
    GEMINI_API_KEY_6: str = ""
    GEMINI_API_KEY_7: str = ""
    GEMINI_API_KEY_8: str = ""
    GEMINI_API_KEY_9: str = ""
    GEMINI_API_KEY_10: str = ""

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-haiku-4-5-20251001"

    LLM_BACKEND: str = "gemini"
    LLM_MODEL: str = "gemini-2.0-flash"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 2048

    # ── Embeddings (local, no API key needed) ─────────────────────────────
    EMBED_MODEL: str = "all-MiniLM-L6-v2"
    EMBED_DIM: int = 384

    # ── arXiv ingestion ───────────────────────────────────────────────────
    ARXIV_MAX_RESULTS: int = 15
    ARXIV_CATEGORIES: List[str] = ["cs.AI", "cs.LG", "cs.CL", "cs.IR"]

    # ── Memory / MemRL ────────────────────────────────────────────────────
    # ALPHA raised from 0.1 → 0.3: makes Q-value movement visible in 5-7 sessions.
    # EPSILON at 0.2: 20% exploration ensures all strategies get tried quickly.
    # Set MEMRL_EPSILON=0.0 in .env to disable exploration (pure exploitation).
    MEMRL_ALPHA: float = 0.3        # FIX: was 0.1 — too slow for demo-visible learning
    MEMRL_GAMMA: float = 0.9
    MEMRL_EPSILON: float = 0.2      # FIX: raised from 0.15 for faster strategy diversity
    MEMORY_MAX_TRIPLETS: int = 10000

    # ── CORS ──────────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
    ]

    LOG_LEVEL: str = "INFO"


settings = Settings()
