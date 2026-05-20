import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx

from backend.app.core.config import Settings, get_settings
from backend.app.rag.openai_provider import OpenAIProviderError, post_with_retries

OPENAI_EMBEDDINGS_PATH = "/embeddings"


class EmbeddingDimensionError(ValueError):
    pass


class OpenAIEmbeddingError(OpenAIProviderError):
    pass


@dataclass(frozen=True)
class EmbeddingUsage:
    input_tokens: int = 0
    total_tokens: int = 0


@runtime_checkable
class EmbeddingClient(Protocol):
    provider_name: str
    model_name: str
    dimension: int
    last_usage: EmbeddingUsage

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
    provider_name = "fake"

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
        self.last_usage = EmbeddingUsage()

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = [self._embed_one(text) for text in texts]
        validate_embedding_batch(embeddings, expected_dimension=self.dimension)
        self.last_usage = EmbeddingUsage()
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


class OpenAIEmbeddingClient:
    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.25,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("api_key must not be blank")
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must not be negative")

        self.api_key = api_key
        self.model_name = model_name
        self.dimension = dimension
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.http_client = http_client
        self.last_usage = EmbeddingUsage()

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        normalized_texts = normalize_embedding_inputs(texts)
        payload = {
            "model": self.model_name,
            "input": normalized_texts,
            "encoding_format": "float",
            "dimensions": self.dimension,
        }
        response_data = await self._create_embeddings(payload)
        embeddings = parse_openai_embeddings_response(
            response_data,
            expected_count=len(normalized_texts),
        )
        validate_embedding_batch(embeddings, expected_dimension=self.dimension)
        self.last_usage = parse_openai_embeddings_usage(response_data)
        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        return (await self.embed_texts([query]))[0]

    async def _create_embeddings(self, payload: dict) -> dict:
        if self.http_client is not None:
            return await self._post_embeddings(self.http_client, payload)

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            return await self._post_embeddings(client, payload)

    async def _post_embeddings(
        self,
        client: httpx.AsyncClient,
        payload: dict,
    ) -> dict:
        async def call() -> httpx.Response:
            return await client.post(
                f"{self.base_url}{OPENAI_EMBEDDINGS_PATH}",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

        response = await post_with_retries(
            operation="OpenAI embeddings request",
            call=call,
            max_retries=self.max_retries,
            retry_delay_seconds=self.retry_delay_seconds,
            error_cls=OpenAIEmbeddingError,
        )

        try:
            response_data = response.json()
        except ValueError as exc:
            raise OpenAIEmbeddingError(
                "OpenAI embeddings response was not valid JSON"
            ) from exc
        if not isinstance(response_data, dict):
            raise OpenAIEmbeddingError("OpenAI embeddings response must be an object")
        return response_data


def normalize_embedding_inputs(texts: Sequence[str]) -> list[str]:
    normalized_texts = [" ".join(text.split()) for text in texts]
    if any(not text for text in normalized_texts):
        raise ValueError("text must not be blank")
    return normalized_texts


def parse_openai_embeddings_response(
    response_data: dict,
    *,
    expected_count: int,
) -> list[list[float]]:
    raw_items = response_data.get("data")
    if not isinstance(raw_items, list):
        raise OpenAIEmbeddingError("OpenAI embeddings response missing data list")

    embeddings_by_index: dict[int, list[float]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            raise OpenAIEmbeddingError("OpenAI embedding item must be an object")
        index = item.get("index")
        embedding = item.get("embedding")
        if not isinstance(index, int):
            raise OpenAIEmbeddingError("OpenAI embedding item missing integer index")
        if not isinstance(embedding, list):
            raise OpenAIEmbeddingError("OpenAI embedding item missing embedding list")
        embeddings_by_index[index] = embedding

    expected_indexes = set(range(expected_count))
    if set(embeddings_by_index) != expected_indexes:
        raise OpenAIEmbeddingError(
            "OpenAI embeddings response indexes did not match request inputs"
        )

    return [embeddings_by_index[index] for index in range(expected_count)]


def parse_openai_embeddings_usage(response_data: dict) -> EmbeddingUsage:
    raw_usage = response_data.get("usage")
    if raw_usage is None:
        return EmbeddingUsage()
    if not isinstance(raw_usage, dict):
        raise OpenAIEmbeddingError("OpenAI embeddings usage must be an object")

    input_tokens = raw_usage.get("prompt_tokens", raw_usage.get("input_tokens", 0))
    total_tokens = raw_usage.get("total_tokens", input_tokens)
    if (
        not isinstance(input_tokens, int)
        or not isinstance(total_tokens, int)
        or input_tokens < 0
        or total_tokens < 0
    ):
        raise OpenAIEmbeddingError(
            "OpenAI embeddings usage token counts must be non-negative integers"
        )
    return EmbeddingUsage(
        input_tokens=input_tokens,
        total_tokens=total_tokens,
    )


def build_embedding_client(settings: Settings | None = None) -> EmbeddingClient:
    settings = settings or get_settings()

    if settings.embedding_provider == "fake":
        return FakeEmbeddingClient(
            dimension=settings.embedding_dimension,
            model_name=settings.embedding_model,
        )

    if settings.embedding_provider == "openai":
        if settings.openai_api_key is None or not settings.openai_api_key.strip():
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        return OpenAIEmbeddingClient(
            api_key=settings.openai_api_key,
            model_name=settings.openai_embedding_model,
            dimension=settings.embedding_dimension,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
            retry_delay_seconds=settings.openai_retry_delay_seconds,
        )

    raise ValueError(f"unsupported embedding provider: {settings.embedding_provider}")
