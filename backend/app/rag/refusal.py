from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from backend.app.core.config import Settings, get_settings
from backend.app.rag.retrieval_models import RetrievedChunk

REFUSAL_ANSWER = "I don't know based on the provided documents."


class RefusalInfo(BaseModel):
    reason: Literal["no_retrieved_chunks", "low_retrieval_confidence"]
    top_score: float | None
    threshold: float


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

