"""Self-evaluation v2 — grounded, calibrated, reference-free RAG evaluation.

WHY v1 WAS BROKEN:
  v1 asked the LLM "rate your own confidence/hallucination" with no rubric.
  LLaMA 3.3 returns its biased prior (0.8 / 0.2) on nearly every call, and
  hallucination was never measured against sources at all.

WHAT v2 DOES (modern, free, deterministic where possible):
  1. CLAIM EXTRACTION — split the answer into atomic factual claims.
  2. EMBEDDING-NLI FAITHFULNESS (RAGAS-style) — each claim is embedded and
     matched against source embeddings. Hallucination = fraction of claims
     with NO source support. This VARIES per session and is actually true.
  3. RUBRIC-ANCHORED JUDGE with SELF-CONSISTENCY — 1-5 rubric, N samples
     averaged, used only as a secondary signal (weight 0.10-0.30).
  4. CALIBRATED CONFIDENCE — blended from faithfulness, citation precision,
     retrieval support and judge score. Never stuck at a constant.
"""
import re
import math
from typing import List, Dict, Any, Tuple
import structlog

from app.core.llm import get_llm
from app.core.embeddings import embed, batch_cosine_similarity
from app.core.models import Paper
from app.core.config import settings

log = structlog.get_logger()

_CLAIM_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")
_META_RE = re.compile(
    r"(in summary|in conclusion|references\s*:|sources\s*:|as an ai|i cannot|"
    r"note\s*:|overall\s*,)", re.I,
)

# ── 1. Claim extraction ────────────────────────────────────────────────────

def split_claims(answer: str, max_claims: int = 12) -> List[str]:
    """Split an answer into atomic, factual, checkable claims."""
    text = re.sub(r"```.*?```", " ", answer, flags=re.S)          # drop code blocks
    text = re.sub(r"^[#>\-*•\d\.]+\s*", "", text, flags=re.M)     # drop bullets/headers
    sentences = [s.strip() for s in _CLAIM_SPLIT_RE.split(text) if s]
    claims = [s for s in sentences if len(s) >= 30 and not _META_RE.search(s)]
    return (claims or sentences)[:max_claims]

# ── 2. Embedding-NLI faithfulness (hallucination that is MEASURED) ──────────

def measure_faithfulness(answer: str, source_docs: List[Paper]) -> Dict[str, Any]:
    """
    RAGAS-style faithfulness without an NLI model:
    embed every claim, take max cosine vs every source; grounded if ≥ threshold.
    Deterministic, free (local ONNX), and varies per session/strategy.
    """
    claims = split_claims(answer)
    if not claims or not source_docs:
        return {"faithfulness": 0.0, "claims": len(claims), "grounded": 0,
                "ungrounded_claims": claims[:4], "mean_claim_similarity": 0.0}

    corpus = [f"{d.title}. {d.abstract}" for d in source_docs[:8]]
    try:
        claim_embs = embed(claims)
        corpus_embs = embed(corpus)
    except Exception as e:
        log.warning("Faithfulness embedding failed", error=str(e))
        return {"faithfulness": 0.5, "claims": len(claims), "grounded": 0,
                "ungrounded_claims": [], "mean_claim_similarity": 0.0}

    grounded, sims_best, ungrounded = 0, [], []
    for claim, ce in zip(claims, claim_embs):
        sims = batch_cosine_similarity(ce, corpus_embs)
        best = max(sims) if sims else 0.0
        sims_best.append(best)
        if best >= settings.FAITHFULNESS_THRESHOLD:
            grounded += 1
        else:
            ungrounded.append(claim[:140])

    return {
        "faithfulness": round(grounded / len(claims), 4),
        "claims": len(claims),
        "grounded": grounded,
        "ungrounded_claims": ungrounded[:5],
        "mean_claim_similarity": round(sum(sims_best) / len(sims_best), 4),
    }

# ── 3. Rubric-anchored judge with self-consistency ──────────────────────────

JUDGE_PROMPT = """You are a strict academic peer reviewer judging a grounded answer.
Rubric (anchor yourself to these definitions):
5 = every major claim traceable to sources; complete; precise terminology
4 = grounded but minor gaps or one unsupported detail
3 = partially grounded; noticeable unsupported statements
2 = mostly unsupported; contradicts or ignores sources
1 = fabricated / irrelevant
Respond ONLY: SCORE: <integer 1-5>"""

async def llm_judge_score(query: str, answer: str, source_docs: List[Paper]) -> float:
    """Self-consistency judge: N samples at temp 0.3, averaged. Secondary signal only."""
    context = "\n".join(f"[{i+1}] {d.title}: {d.abstract[:220]}"
                        for i, d in enumerate(source_docs[:4]))
    user = (f"Question: {query}\n\nAnswer:\n{answer[:900]}\n\nSources:\n{context}")
    scores = []
    for _ in range(max(1, settings.JUDGE_SAMPLES)):
        try:
            llm = get_llm()
            resp = await llm.complete(system=JUDGE_PROMPT, user=user,
                                      max_tokens=12, temperature=0.3)
            m = re.search(r"SCORE:\s*([1-5])", resp)
            if m:
                scores.append((int(m.group(1)) - 1) / 4.0)   # 1-5 → 0-1
        except Exception:
            continue
    return round(sum(scores) / len(scores), 4) if scores else 0.5

# ── 4. Calibrated confidence / hallucination ────────────────────────────────

def calibrate(faithfulness: float, citation_precision: float,
              retrieval_support: float, judge: float) -> Tuple[float, float]:
    """
    Confidence = weighted blend of MEASURED signals (varies every session).
    Hallucination = measured ungrounded-claim fraction, softened by judge.
    """
    confidence = (
        0.45 * faithfulness
        + 0.20 * citation_precision
        + 0.20 * min(1.0, retrieval_support * 1.0)   # raw cosine, no inflation
        + 0.15 * judge
    )
    hallucination = 0.75 * (1.0 - faithfulness) + 0.25 * (1.0 - judge)
    return round(max(0.0, min(1.0, confidence)), 4), round(max(0.0, min(1.0, hallucination)), 4)

def compute_outcome_quality(faithfulness: float, citation_precision: float,
                            retrieval_score: float, answer: str) -> float:
    """Q-learning reward: fully measured, varies per strategy & session."""
    length = len(answer or "")
    length_score = 1.0 / (1.0 + math.exp(-0.005 * (length - 400)))
    quality = (
        0.35 * faithfulness
        + 0.25 * citation_precision
        + 0.25 * min(1.0, retrieval_score)
        + 0.15 * length_score
    )
    return round(max(0.05, min(0.99, quality)), 4)

# Backwards-compatible shim (old callers)
async def evaluate_answer(query: str, answer: str, source_docs: List[Paper]):
    f = measure_faithfulness(answer, source_docs)
    j = await llm_judge_score(query, answer, source_docs)
    conf, hall = calibrate(f["faithfulness"], 0.5, 0.5, j)
    return conf, hall
