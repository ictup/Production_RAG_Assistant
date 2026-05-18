import math

import pytest

from backend.app.core.config import Settings
from backend.app.rag.embeddings import (
    EmbeddingClient,
    EmbeddingDimensionError,
    FakeEmbeddingClient,
    build_embedding_client,
    validate_embedding_batch,
    validate_embedding_dimension,
)


@pytest.mark.asyncio
async def test_fake_embedding_client_returns_deterministic_vectors() -> None:
    client = FakeEmbeddingClient(dimension=8)

    first = await client.embed_query("FlashAttention reduces HBM traffic")
    second = await client.embed_query("FlashAttention reduces HBM traffic")

    assert first == second
    assert len(first) == 8
    assert math.isclose(sum(value * value for value in first), 1.0)


@pytest.mark.asyncio
async def test_fake_embedding_client_preserves_input_order() -> None:
    client = FakeEmbeddingClient(dimension=6)

    embeddings = await client.embed_texts(["KV cache", "PagedAttention", "KV cache"])

    assert len(embeddings) == 3
    assert embeddings[0] == embeddings[2]
    assert embeddings[0] != embeddings[1]


@pytest.mark.asyncio
async def test_fake_embedding_client_rejects_blank_text() -> None:
    client = FakeEmbeddingClient(dimension=4)

    with pytest.raises(ValueError, match="must not be blank"):
        await client.embed_query("   ")


def test_validate_embedding_dimension_rejects_wrong_size() -> None:
    with pytest.raises(EmbeddingDimensionError, match="does not match"):
        validate_embedding_dimension([0.1, 0.2], expected_dimension=3)


def test_validate_embedding_dimension_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite numeric"):
        validate_embedding_dimension([0.1, float("nan")], expected_dimension=2)


def test_validate_embedding_batch_reports_failing_index() -> None:
    with pytest.raises(EmbeddingDimensionError, match=r"embedding\[1\]"):
        validate_embedding_batch([[0.1, 0.2], [0.3]], expected_dimension=2)


def test_build_embedding_client_uses_settings() -> None:
    client = build_embedding_client(
        Settings(
            embedding_provider="fake",
            embedding_model="test-fake",
            embedding_dimension=12,
        )
    )

    assert isinstance(client, EmbeddingClient)
    assert client.model_name == "test-fake"
    assert client.dimension == 12

