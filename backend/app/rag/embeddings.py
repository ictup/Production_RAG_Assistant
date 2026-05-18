import hashlib
import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from backend.app.core.config import Settings, get_settings


class EmbeddingDimensionError(ValueError):
    pass


@runtime_checkable
class EmbeddingClient(Protocol):
    model_name: str
    dimension: int

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        pass

    async def embed_query(self, query: str) -> list[float]:
        pass


def validate_embedding_dimension(
    embedding: Sequence[float],
    *,
    expected_dimension: int,
    label: str = "embedding",
) -> None:
    if len(embedding) != expected_dimension:
        raise EmbeddingDimensionError(
            f"{label} dimension {len(embedding)} does not match "
            f"expected dimension {expected_dimension}"
        )

    has_only_finite_numbers = all(
        isinstance(value, int | float) and math.isfinite(value)
        for value in embedding
    )
    if not has_only_finite_numbers:
        raise ValueError(f"{label} must contain only finite numeric values")


def validate_embedding_batch(
    embeddings: Sequence[Sequence[float]],
    *,
    expected_dimension: int,
) -> None:
    for index, embedding in enumerate(embeddings):
        validate_embedding_dimension(
            embedding,
            expected_dimension=expected_dimension,
            label=f"embedding[{index}]",
        )


class FakeEmbeddingClient:
    def __init__(
        self,
        *,
        dimension: int = 1536,
        model_name: str = "fake-embedding",
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")

        self.dimension = dimension
        self.model_name = model_name

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = [self._embed_one(text) for text in texts]
        validate_embedding_batch(embeddings, expected_dimension=self.dimension)
        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        return (await self.embed_texts([query]))[0]

    def _embed_one(self, text: str) -> list[float]:
        normalized = " ".join(text.split())
        if not normalized:
            raise ValueError("text must not be blank")

        values: list[float] = []
        seed = f"{self.model_name}:{normalized}".encode()
        counter = 0

        while len(values) < self.dimension:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            values.extend((byte / 127.5) - 1.0 for byte in digest)
            counter += 1

        vector = values[: self.dimension]
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector

        return [value / norm for value in vector]


def build_embedding_client(settings: Settings | None = None) -> EmbeddingClient:
    settings = settings or get_settings()

    if settings.embedding_provider == "fake":
        return FakeEmbeddingClient(
            dimension=settings.embedding_dimension,
            model_name=settings.embedding_model,
        )

    raise ValueError(f"unsupported embedding provider: {settings.embedding_provider}")
