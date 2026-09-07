"""Application configuration via environment variables."""
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ENV: str = "development"
    DATABASE_URL: str = "sqlite+aiosqlite:///./ghostmind.db"

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
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
    LLM_MAX_TOKENS: int = 1024

    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBED_DIM: int = 384

    ARXIV_MAX_RESULTS: int = 5
    ARXIV_CATEGORIES: List[str] = ["cs.AI", "cs.LG", "cs.CL", "cs.IR"]

    MEMRL_ALPHA: float = 0.3
    MEMRL_GAMMA: float = 0.9
    MEMRL_EPSILON: float = 0.2
    MEMORY_MAX_TRIPLETS: int = 10000

    # ── NEW: Evaluation v2 (grounded, calibrated) ─────────────────────────
    FAITHFULNESS_THRESHOLD: float = 0.50     # claim↔source cosine ≥ this = grounded
    CITATION_VERIFY_THRESHOLD: float = 0.45  # citation sentence↔doc cosine ≥ this = verified
    JUDGE_SAMPLES: int = 2                   # self-consistency samples for rubric judge

    # ── NEW: Retrieval v2 (RRF / MMR / CRAG / HyDE) ───────────────────────
    MMR_LAMBDA: float = 0.70                 # relevance vs diversity balance
    RRF_K: int = 60                          # RRF fusion constant
    CRAG_REWRITE_THRESHOLD: float = 0.25     # raw top-3 cosine below this → corrective rewrite
    RELEVANCE_TRUST_THRESHOLD: float = 0.30  # raw cosine above this → skip LLM grader

    # ── NEW: MemRL v2 (UCB1 bandit) ───────────────────────────────────────
    UCB_C: float = 0.6                       # exploration bonus strength

    CORS_ORIGINS: List[str] = [
        "http://localhost:3000", "http://localhost:5173", "http://localhost:5174",
        "https://ghostmind-api.onrender.com",
        "https://ghost-mind.vercel.app", "https://ghostmind.vercel.app",
    ]
    CORS_EXTRA_ORIGINS: str = ""

    def get_all_cors_origins(self) -> List[str]:
        origins = list(self.CORS_ORIGINS)
        if self.CORS_EXTRA_ORIGINS.strip():
            for o in self.CORS_EXTRA_ORIGINS.split(","):
                o = o.strip()
                if o and o not in origins:
                    origins.append(o)
        return origins

    LOG_LEVEL: str = "INFO"

settings = Settings()
