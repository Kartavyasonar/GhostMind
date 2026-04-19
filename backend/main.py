"""GhostMind backend — FastAPI entry point."""
from contextlib import asynccontextmanager
import structlog
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.core.config import settings
from app.core.database import init_db
from app.api import api_router

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("GhostMind starting", env=settings.ENV)
    await init_db()
    yield
    log.info("GhostMind shutting down")


app = FastAPI(
    title="GhostMind API",
    description="Self-evolving research agent with episodic memory and MemRL",
    version="1.0.0",
    lifespan=lifespan,
)

# FIX: use get_all_cors_origins() so CORS_EXTRA_ORIGINS env var is honoured.
# In Render dashboard set: CORS_EXTRA_ORIGINS=https://your-frontend.vercel.app
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_all_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
@app.head("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
