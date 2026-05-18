import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from backend.app.core.config import Settings, get_settings
from backend.app.rag.retrieval_models import RetrievedChunk

REFUSAL_ANSWER = "I don't know based on the provided documents."
PROMPT_INJECTION_PATTERNS = (
    "ignore all previous instructions",
    "ignore previous instructions",
    "ignore your rules",
    "ignore the rules",
    "reveal the system prompt",
    "system prompt",
    "developer message",
    "hidden metadata",
    "private configuration",
    "api key",
    "secret key",
    "retrieved document instruction",
    "jailbreak",
)
OUT_OF_SCOPE_PATTERNS = (
    "what did i eat",
    "who won eurovision",
)
PERSONAL_HISTORY_PATTERN = re.compile(
    r"\bwhat\s+did\s+i\b.*\b(yesterday|today|last night|last week|last month)\b"
)


class RefusalInfo(BaseModel):
    reason: Literal[
        "no_retrieved_chunks",
        "low_retrieval_confidence",
        "unsafe_question",
        "out_of_scope_question",
    ]
    top_score: float | None
    threshold: float


def normalize_question_for_guard(question: str) -> str:
    return " ".join(question.casefold().split())


def should_refuse_question(question: str) -> RefusalInfo | None:
    normalized_question = normalize_question_for_guard(question)
    if not normalized_question:
        raise ValueError("question must not be blank")

    if any(pattern in normalized_question for pattern in PROMPT_INJECTION_PATTERNS):
        return RefusalInfo(
            reason="unsafe_question",
            top_score=None,
            threshold=0.0,
        )

    if any(pattern in normalized_question for pattern in OUT_OF_SCOPE_PATTERNS):
        return RefusalInfo(
            reason="out_of_scope_question",
            top_score=None,
            threshold=0.0,
        )

    if PERSONAL_HISTORY_PATTERN.search(normalized_question):
        return RefusalInfo(
            reason="out_of_scope_question",
            top_score=None,
            threshold=0.0,
        )

    return None


def get_top_score(chunks: Sequence[RetrievedChunk]) -> float | None:
    if not chunks:
        return None
    return max(chunk.score for chunk in chunks)


def should_refuse(
    chunks: Sequence[RetrievedChunk],
    *,
    threshold: float,
) -> RefusalInfo | None:
    if threshold < 0:
        raise ValueError("threshold must not be negative")

    top_score = get_top_score(chunks)
    if top_score is None:
        return RefusalInfo(
            reason="no_retrieved_chunks",
            top_score=None,
            threshold=threshold,
        )

    if top_score < threshold:
        return RefusalInfo(
            reason="low_retrieval_confidence",
            top_score=top_score,
            threshold=threshold,
        )

    return None


def refusal_from_settings(
    chunks: Sequence[RetrievedChunk],
    settings: Settings | None = None,
) -> RefusalInfo | None:
    settings = settings or get_settings()
    return should_refuse(
        chunks,
        threshold=settings.refusal_score_threshold,
    )
