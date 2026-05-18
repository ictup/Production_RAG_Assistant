import uuid

import pytest

from backend.app.rag.fusion import reciprocal_rank_fusion
from backend.app.rag.retrieval_models import RetrievedChunk


def make_chunk(
    chunk_id: str,
    *,
    rank: int,
    score: float,
    mode: str,
    source_uri: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=str(uuid.uuid4()),
        text=f"text for {chunk_id}",
        title=f"title for {chunk_id}",
        section_title=None,
        source_uri=source_uri or f"{chunk_id}.md",
        score=score,
        rank=rank,
        retrieval_mode=mode,  # type: ignore[arg-type]
        metadata={},
    )


def test_rrf_merges_duplicate_chunks_and_reranks_by_fused_score() -> None:
    shared = "shared"
    vector_results = [
        make_chunk(shared, rank=1, score=0.9, mode="vector"),
        make_chunk("vector-only", rank=2, score=0.8, mode="vector"),
    ]
    sparse_results = [
        make_chunk("sparse-only", rank=1, score=10.0, mode="sparse"),
        make_chunk(shared, rank=2, score=9.0, mode="sparse"),
    ]

    fused = reciprocal_rank_fusion(
        [vector_results, sparse_results],
        k=60,
        top_n=3,
    )

    assert [chunk.chunk_id for chunk in fused] == [
        shared,
        "sparse-only",
        "vector-only",
    ]
    assert fused[0].rank == 1
    assert fused[0].retrieval_mode == "hybrid_rrf"
    assert fused[0].score == pytest.approx((1 / 61) + (1 / 62))


def test_rrf_respects_top_n() -> None:
    results = [
        make_chunk("a", rank=1, score=0.9, mode="vector"),
        make_chunk("b", rank=2, score=0.8, mode="vector"),
        make_chunk("c", rank=3, score=0.7, mode="vector"),
    ]

    fused = reciprocal_rank_fusion([results], top_n=2)

    assert [chunk.chunk_id for chunk in fused] == ["a", "b"]


def test_rrf_does_not_mutate_input_chunks() -> None:
    chunk = make_chunk("a", rank=1, score=0.9, mode="vector")

    fused = reciprocal_rank_fusion([[chunk]], top_n=1)

    assert chunk.rank == 1
    assert chunk.score == 0.9
    assert chunk.retrieval_mode == "vector"
    assert fused[0] is not chunk
    assert fused[0].retrieval_mode == "hybrid_rrf"


def test_rrf_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="k"):
        reciprocal_rank_fusion([], k=0)

    with pytest.raises(ValueError, match="top_n"):
        reciprocal_rank_fusion([], top_n=0)

