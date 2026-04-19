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
    LLM_MAX_TOKENS: int = 1024   # FIX: reduced from 2048 — halves peak LLM response RAM

    # ── Embeddings ────────────────────────────────────────────────────────
    # FIX: paraphrase-MiniLM-L3-v2 is 61MB vs all-MiniLM-L6-v2 at 90MB.
    # Both produce 384-dim vectors. L3 is slightly less accurate but fits
    # the 512MB Render free tier without OOM.
    # Set EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 in .env for
    # better accuracy if you upgrade to a paid Render instance.
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBED_DIM: int = 384

    # ── arXiv ingestion ───────────────────────────────────────────────────
    # FIX: reduced from 15 → 5. Each paper = ~2KB embedding stored in JSON.
    # 15 papers × 2 embedding calls/paper was the main OOM trigger during ingest.
    # 5 papers is sufficient for RAG retrieval quality.
    ARXIV_MAX_RESULTS: int = 5
    ARXIV_CATEGORIES: List[str] = ["cs.AI", "cs.LG", "cs.CL", "cs.IR"]

    # ── Memory / MemRL ────────────────────────────────────────────────────
    MEMRL_ALPHA: float = 0.3
    MEMRL_GAMMA: float = 0.9
    MEMRL_EPSILON: float = 0.2
    MEMORY_MAX_TRIPLETS: int = 10000

    # ── CORS ──────────────────────────────────────────────────────────────
    # FIX: Added CORS_EXTRA_ORIGINS env var so you can inject your exact
    # Render/Vercel URLs at deploy time without editing code.
    # In Render dashboard: CORS_EXTRA_ORIGINS=https://your-app.vercel.app
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "https://ghostmind-api.onrender.com",
        "https://ghost-mind.vercel.app",
        "https://ghostmind.vercel.app",
    ]
    # Extra origins injected via env at deploy time (comma-separated string)
    CORS_EXTRA_ORIGINS: str = ""

    def get_all_cors_origins(self) -> List[str]:
        """Merge static CORS_ORIGINS with any runtime CORS_EXTRA_ORIGINS."""
        origins = list(self.CORS_ORIGINS)
        if self.CORS_EXTRA_ORIGINS.strip():
            for o in self.CORS_EXTRA_ORIGINS.split(","):
                o = o.strip()
                if o and o not in origins:
                    origins.append(o)
        return origins

    LOG_LEVEL: str = "INFO"


settings = Settings()
