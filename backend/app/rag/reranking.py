import json
from collections.abc import Sequence
from json import JSONDecodeError
from typing import Protocol, runtime_checkable

import httpx

from backend.app.core.config import Settings, get_settings
from backend.app.rag.generation import extract_openai_response_text
from backend.app.rag.openai_provider import (
    OpenAIErrorInfo,
    OpenAIProviderError,
    post_with_retries,
)
from backend.app.rag.retrieval_models import RetrievedChunk

OPENAI_RESPONSES_PATH = "/responses"
OPENAI_RERANK_INSTRUCTIONS = (
    "You are a retrieval reranker. Rank candidate passages by relevance to the "
    "query. Return only strict JSON in this shape: "
    '{"ranked_indices":[1,2,3]}. Use the provided 1-based candidate numbers, '
    "most relevant first. Do not explain your reasoning."
)
MAX_RERANK_TEXT_CHARS = 2000


class OpenAIRerankingError(OpenAIProviderError):
    pass


@runtime_checkable
class Reranker(Protocol):
    provider_name: str

    async def rerank(
        self,
        *,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        pass


class NoOpReranker:
    provider_name = "none"

    async def rerank(
        self,
        *,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("query must not be blank")
        if top_n <= 0:
            raise ValueError("top_n must be greater than zero")

        return [
            chunk.model_copy(update={"rank": rank})
            for rank, chunk in enumerate(chunks[:top_n], start=1)
        ]


class OpenAIListwiseReranker:
    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.25,
        max_output_tokens: int = 512,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("api_key must not be blank")
        if not model_name.strip():
            raise ValueError("model_name must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must not be negative")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")

        self.api_key = api_key
        self.model_name = model_name.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.max_output_tokens = max_output_tokens
        self.http_client = http_client

    async def rerank(
        self,
        *,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be blank")
        if top_n <= 0:
            raise ValueError("top_n must be greater than zero")
        if not chunks:
            return []

        payload = {
            "model": self.model_name,
            "instructions": OPENAI_RERANK_INSTRUCTIONS,
            "input": build_rerank_prompt(query=query, chunks=chunks),
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        response_data = await self._create_response(payload)
        try:
            response_text = extract_openai_response_text(response_data)
        except OpenAIProviderError as exc:
            raise build_rerank_response_error(str(exc)) from exc

        order = parse_ranked_indices(response_text, candidate_count=len(chunks))
        return apply_rerank_order(chunks=chunks, order=order, top_n=top_n)

    async def _create_response(self, payload: dict) -> dict:
        if self.http_client is not None:
            return await self._post_response(self.http_client, payload)

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            return await self._post_response(client, payload)

    async def _post_response(
        self,
        client: httpx.AsyncClient,
        payload: dict,
    ) -> dict:
        async def call() -> httpx.Response:
            return await client.post(
                f"{self.base_url}{OPENAI_RESPONSES_PATH}",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

        response = await post_with_retries(
            operation="OpenAI rerank request",
            call=call,
            max_retries=self.max_retries,
            retry_delay_seconds=self.retry_delay_seconds,
            error_cls=OpenAIRerankingError,
        )

        try:
            response_data = response.json()
        except ValueError as exc:
            raise build_rerank_response_error(
                "OpenAI rerank response was not valid JSON"
            ) from exc
        if not isinstance(response_data, dict):
            raise build_rerank_response_error(
                "OpenAI rerank response must be an object"
            )
        return response_data


def build_rerank_response_error(message: str) -> OpenAIRerankingError:
    return OpenAIRerankingError(
        OpenAIErrorInfo(
            operation="OpenAI rerank response",
            category="provider_error",
            message=message,
            retryable=False,
        )
    )


def build_rerank_prompt(*, query: str, chunks: Sequence[RetrievedChunk]) -> str:
    candidate_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        section = chunk.section_title or ""
        candidate_blocks.append(
            "\n".join(
                [
                    f"[{index}]",
                    f"Title: {chunk.title}",
                    f"Section: {section}",
                    f"Source: {chunk.source_uri}",
                    f"Retrieval score: {chunk.score:.6f}",
                    "Text:",
                    truncate_rerank_text(chunk.text),
                ]
            )
        )

    return "\n\n".join(
        [
            f"Query:\n{query}",
            "Candidates:",
            *candidate_blocks,
            'Return JSON only, for example: {"ranked_indices":[2,1,3]}',
        ]
    )


def truncate_rerank_text(text: str) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= MAX_RERANK_TEXT_CHARS:
        return normalized
    return normalized[:MAX_RERANK_TEXT_CHARS].rstrip()


def parse_ranked_indices(response_text: str, *, candidate_count: int) -> list[int]:
    if candidate_count < 0:
        raise ValueError("candidate_count must not be negative")
    if candidate_count == 0:
        return []

    try:
        parsed = parse_rerank_json(response_text)
    except OpenAIRerankingError:
        return list(range(candidate_count))

    raw_indices = extract_raw_indices(parsed)
    ranked_indices: list[int] = []
    seen_indices: set[int] = set()
    for value in raw_indices:
        candidate_number = coerce_candidate_number(value)
        if candidate_number is None:
            continue
        index = candidate_number - 1
        if index < 0 or index >= candidate_count or index in seen_indices:
            continue
        ranked_indices.append(index)
        seen_indices.add(index)

    ranked_indices.extend(
        index for index in range(candidate_count) if index not in seen_indices
    )
    return ranked_indices


def parse_rerank_json(response_text: str) -> object:
    text = response_text.strip()
    if not text:
        raise OpenAIRerankingError("OpenAI rerank response text was blank")

    try:
        return json.loads(text)
    except JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except JSONDecodeError:
            pass

    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except JSONDecodeError:
            pass

    raise OpenAIRerankingError("OpenAI rerank response did not contain JSON")


def extract_raw_indices(parsed: object) -> list[object]:
    if isinstance(parsed, list):
        return list(parsed)
    if not isinstance(parsed, dict):
        return []

    for key in ("ranked_indices", "indices", "ranking"):
        value = parsed.get(key)
        if isinstance(value, list):
            return list(value)
    return []


def coerce_candidate_number(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdecimal():
            return int(stripped)
        return None
    if isinstance(value, dict):
        for key in ("candidate_id", "index", "id"):
            candidate_number = coerce_candidate_number(value.get(key))
            if candidate_number is not None:
                return candidate_number
    return None


def apply_rerank_order(
    *,
    chunks: Sequence[RetrievedChunk],
    order: Sequence[int],
    top_n: int,
) -> list[RetrievedChunk]:
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero")
    return [
        chunks[index].model_copy(update={"rank": rank})
        for rank, index in enumerate(order[:top_n], start=1)
    ]


def build_reranker(settings: Settings | None = None) -> Reranker:
    settings = settings or get_settings()

    if settings.reranker_provider == "none":
        return NoOpReranker()

    if settings.reranker_provider == "openai":
        if settings.openai_api_key is None or not settings.openai_api_key.strip():
            raise ValueError("OPENAI_API_KEY is required for OpenAI reranking")
        return OpenAIListwiseReranker(
            api_key=settings.openai_api_key,
            model_name=settings.reranker_model,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
            retry_delay_seconds=settings.openai_retry_delay_seconds,
            max_output_tokens=settings.openai_max_output_tokens,
        )

    raise ValueError(f"unsupported reranker provider: {settings.reranker_provider}")
