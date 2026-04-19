"""Self-evaluation: confidence scoring and hallucination detection."""
from typing import List, Tuple
import re
import structlog

from app.core.llm import get_llm
from app.core.models import Paper

log = structlog.get_logger()


async def evaluate_answer(
    query: str,
    answer: str,
    source_docs: List[Paper],
) -> Tuple[float, float]:
    if not answer.strip():
        return 0.0, 1.0

    context = "\n\n".join(
        f"[{i+1}] {d.title}: {d.abstract[:300]}" for i, d in enumerate(source_docs[:4])
    )

    prompt = (
        f"Question: {query}\n\n"
        f"Answer: {answer[:800]}\n\n"
        f"Source documents:\n{context}\n\n"
        "Evaluate:\n"
        "1. CONFIDENCE (0-1): How well does the answer match the sources?\n"
        "2. HALLUCINATION (0-1): Fraction of claims NOT grounded in sources?\n"
        "Respond ONLY in this exact format:\n"
        "CONFIDENCE: <number>\n"
        "HALLUCINATION: <number>"
    )

    try:
        llm = get_llm()
        resp = await llm.complete(
            system="You are a critical evaluator. Respond ONLY in the specified format.",
            user=prompt,
            max_tokens=32,
            temperature=0.0,
        )

        confidence = 0.5
        hallucination = 0.3

        for line in resp.splitlines():
            line = line.strip()
            if "CONFIDENCE" in line.upper():
                m = re.search(r"[\d.]+", line)
                if m:
                    confidence = max(0.0, min(1.0, float(m.group())))
            elif "HALLUCINATION" in line.upper():
                m = re.search(r"[\d.]+", line)
                if m:
                    hallucination = max(0.0, min(1.0, float(m.group())))

        return confidence, hallucination

    except Exception as e:
        log.warning("Evaluation failed", error=str(e))
        return 0.5, 0.3


def compute_outcome_quality(confidence: float, hallucination: float) -> float:
    return max(0.0, confidence * (1.0 - hallucination))
