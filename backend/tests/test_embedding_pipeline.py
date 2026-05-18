from collections.abc import Sequence

import pytest

from backend.app.rag.embedding_pipeline import embed_chunks
from backend.app.rag.embeddings import FakeEmbeddingClient
from ingestion.chunking import chunk_document
from ingestion.models import RawDocument


class BadCountEmbeddingClient:
    model_name = "bad-count"
    dimension = 4

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4]]

    async def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class BadDimensionEmbeddingClient:
    model_name = "bad-dimension"
    dimension = 4

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    async def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2]


def make_chunks():
    raw_document = RawDocument(
        title="PagedAttention",
        source_uri="data/raw/pagedattention.md",
        text="# PagedAttention\n\nPagedAttention stores KV cache in pages.",
    )
    return chunk_document(raw_document, chunk_size_tokens=40, chunk_overlap_tokens=5)


@pytest.mark.asyncio
async def test_embed_chunks_returns_one_embedding_per_chunk_in_order() -> None:
    chunks = make_chunks()
    client = FakeEmbeddingClient(dimension=8)

    embeddings = await embed_chunks(chunks, client)

    assert len(embeddings) == len(chunks)
    assert all(len(embedding) == 8 for embedding in embeddings)
    assert embeddings[0] == await client.embed_query(chunks[0].text)


@pytest.mark.asyncio
async def test_embed_chunks_allows_empty_chunk_list() -> None:
    embeddings = await embed_chunks([], FakeEmbeddingClient(dimension=8))

    assert embeddings == []


@pytest.mark.asyncio
async def test_embed_chunks_rejects_provider_count_mismatch() -> None:
    chunks = make_chunks() * 2

    with pytest.raises(ValueError, match="embedding count"):
        await embed_chunks(chunks, BadCountEmbeddingClient())


@pytest.mark.asyncio
async def test_embed_chunks_rejects_provider_dimension_mismatch() -> None:
    chunks = make_chunks()

    with pytest.raises(ValueError, match="does not match"):
        await embed_chunks(chunks, BadDimensionEmbeddingClient())
