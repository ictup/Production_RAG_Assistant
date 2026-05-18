import uuid

import pytest

from backend.app.core.config import Settings
from backend.app.rag.refusal import (
    REFUSAL_ANSWER,
    RefusalInfo,
    get_top_score,
    refusal_from_settings,
    should_refuse,
)
from backend.app.rag.retrieval_models import RetrievedChunk


def make_chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(uuid.uuid4()),
        document_id=str(uuid.uuid4()),
        text="FlashAttention reduces memory traffic.",
        title="FlashAttention Notes",
        section_title="FlashAttention",
        source_uri="llm_systems/flashattention.md",
        score=score,
        rank=1,
        retrieval_mode="hybrid_rrf",
        metadata={},
    )


def test_get_top_score_returns_none_for_empty_chunks() -> None:
    assert get_top_score([]) is None


def test_get_top_score_uses_maximum_score() -> None:
    assert get_top_score([make_chunk(0.1), make_chunk(0.3), make_chunk(0.2)]) == 0.3


def test_should_refuse_when_no_chunks_exist() -> None:
    refusal = should_refuse([], threshold=0.25)

    assert refusal == RefusalInfo(
        reason="no_retrieved_chunks",
        top_score=None,
        threshold=0.25,
    )


def test_should_refuse_when_top_score_below_threshold() -> None:
    refusal = should_refuse([make_chunk(0.12)], threshold=0.25)

    assert refusal is not None
    assert refusal.reason == "low_retrieval_confidence"
    assert refusal.top_score == 0.12
    assert refusal.threshold == 0.25


def test_should_not_refuse_when_top_score_meets_threshold() -> None:
    assert should_refuse([make_chunk(0.25)], threshold=0.25) is None
    assert should_refuse([make_chunk(0.30)], threshold=0.25) is None


def test_should_refuse_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        should_refuse([make_chunk(0.1)], threshold=-0.1)


def test_refusal_from_settings_uses_configured_threshold() -> None:
    refusal = refusal_from_settings(
        [make_chunk(0.1)],
        Settings(refusal_score_threshold=0.2),
    )

    assert refusal is not None
    assert refusal.threshold == 0.2


def test_refusal_answer_matches_blueprint_contract() -> None:
    assert REFUSAL_ANSWER == "I don't know based on the provided documents."

