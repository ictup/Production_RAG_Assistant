import uuid

import pytest

from backend.app.core.config import Settings
from backend.app.rag.reranking import NoOpReranker, Reranker, build_reranker
from backend.app.rag.retrieval_models import RetrievedChunk


def make_chunk(chunk_id: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=str(uuid.uuid4()),
        text=f"text for {chunk_id}",
        title=f"title for {chunk_id}",
        source_uri=f"{chunk_id}.md",
        score=1.0 / rank,
        rank=rank,
        retrieval_mode="hybrid_rrf",
        metadata={},
    )


@pytest.mark.asyncio
async def test_noop_reranker_keeps_order_and_truncates_to_top_n() -> None:
    reranker = NoOpReranker()
    chunks = [make_chunk("a", 1), make_chunk("b", 2), make_chunk("c", 3)]

    reranked = await reranker.rerank(
        query="FlashAttention",
        chunks=chunks,
        top_n=2,
    )

    assert [chunk.chunk_id for chunk in reranked] == ["a", "b"]
    assert [chunk.rank for chunk in reranked] == [1, 2]
    assert reranked[0] is not chunks[0]


@pytest.mark.asyncio
async def test_noop_reranker_reassigns_ranks_after_truncation() -> None:
    reranker = NoOpReranker()
    chunks = [make_chunk("a", 7), make_chunk("b", 9)]

    reranked = await reranker.rerank(query="KV cache", chunks=chunks, top_n=2)

    assert [chunk.rank for chunk in reranked] == [1, 2]


@pytest.mark.asyncio
async def test_noop_reranker_rejects_blank_query() -> None:
    reranker = NoOpReranker()

    with pytest.raises(ValueError, match="query must not be blank"):
        await reranker.rerank(query="  ", chunks=[make_chunk("a", 1)], top_n=1)


@pytest.mark.asyncio
async def test_noop_reranker_rejects_invalid_top_n() -> None:
    reranker = NoOpReranker()

    with pytest.raises(ValueError, match="top_n"):
        await reranker.rerank(query="FlashAttention", chunks=[], top_n=0)


def test_build_reranker_uses_settings() -> None:
    reranker = build_reranker(Settings(reranker_provider="none"))

    assert isinstance(reranker, Reranker)
    assert reranker.provider_name == "none"

