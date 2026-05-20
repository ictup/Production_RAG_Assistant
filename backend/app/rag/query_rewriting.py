import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from backend.app.core.config import Settings, get_settings
from backend.app.rag.generation import extract_openai_response_text
from backend.app.rag.openai_provider import (
    OpenAIErrorInfo,
    OpenAIProviderError,
    post_with_retries,
)

OPENAI_RESPONSES_PATH = "/responses"
OPENAI_QUERY_REWRITE_INSTRUCTIONS = (
    "You rewrite user questions into concise search queries for a production "
    "RAG retriever. Preserve technical entities, acronyms, product names, and "
    "constraints. Remove conversational filler. Return only the rewritten query "
    "as plain text, without quotes or explanation."
)
MAX_REWRITTEN_QUERY_CHARS = 512


class OpenAIQueryRewriteError(OpenAIProviderError):
    pass


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    rewritten_query: str
    provider_name: str
    model_name: str
    rewritten: bool


@runtime_checkable
class QueryRewriter(Protocol):
    provider_name: str
    model_name: str

    async def rewrite(
        self,
        *,
        question: str,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> QueryRewriteResult:
        pass


class NoOpQueryRewriter:
    provider_name = "none"
    model_name = "none"

    async def rewrite(
        self,
        *,
        question: str,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> QueryRewriteResult:
        normalized_question = normalize_question(question)
        return QueryRewriteResult(
            original_query=normalized_question,
            rewritten_query=normalized_question,
            provider_name=self.provider_name,
            model_name=self.model_name,
            rewritten=False,
        )


class OpenAIQueryRewriter:
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
        max_output_tokens: int = 64,
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

    async def rewrite(
        self,
        *,
        question: str,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> QueryRewriteResult:
        original_query = normalize_question(question)
        payload = {
            "model": self.model_name,
            "instructions": OPENAI_QUERY_REWRITE_INSTRUCTIONS,
            "input": build_query_rewrite_prompt(
                question=original_query,
                metadata_filter=metadata_filter,
            ),
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        response_data = await self._create_response(payload)
        try:
            response_text = extract_openai_response_text(response_data)
        except OpenAIProviderError as exc:
            raise build_query_rewrite_response_error(str(exc)) from exc

        rewritten_query = normalize_rewritten_query(
            response_text,
            fallback_query=original_query,
        )
        return QueryRewriteResult(
            original_query=original_query,
            rewritten_query=rewritten_query,
            provider_name=self.provider_name,
            model_name=self.model_name,
            rewritten=rewritten_query.casefold() != original_query.casefold(),
        )

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
            operation="OpenAI query rewrite request",
            call=call,
            max_retries=self.max_retries,
            retry_delay_seconds=self.retry_delay_seconds,
            error_cls=OpenAIQueryRewriteError,
        )

        try:
            response_data = response.json()
        except ValueError as exc:
            raise build_query_rewrite_response_error(
                "OpenAI query rewrite response was not valid JSON"
            ) from exc
        if not isinstance(response_data, dict):
            raise build_query_rewrite_response_error(
                "OpenAI query rewrite response must be an object"
            )
        return response_data


def build_query_rewrite_response_error(message: str) -> OpenAIQueryRewriteError:
    return OpenAIQueryRewriteError(
        OpenAIErrorInfo(
            operation="OpenAI query rewrite response",
            category="provider_error",
            message=message,
            retryable=False,
        )
    )


def build_query_rewrite_prompt(
    *,
    question: str,
    metadata_filter: Mapping[str, Any] | None = None,
) -> str:
    metadata = metadata_filter or {}
    metadata_text = (
        json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        if metadata
        else "{}"
    )
    return "\n\n".join(
        [
            f"Question:\n{question}",
            f"Metadata filter:\n{metadata_text}",
            "Rewrite as one concise retrieval query.",
        ]
    )


def normalize_question(question: str) -> str:
    normalized = " ".join(question.split())
    if not normalized:
        raise ValueError("question must not be blank")
    return normalized


def normalize_rewritten_query(response_text: str, *, fallback_query: str) -> str:
    normalized = strip_query_wrapping(response_text)
    normalized = " ".join(normalized.split())
    if not normalized:
        return fallback_query
    return normalized[:MAX_REWRITTEN_QUERY_CHARS].strip() or fallback_query


def strip_query_wrapping(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()

    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        stripped = stripped[1:-1].strip()

    return stripped


def build_query_rewriter(settings: Settings | None = None) -> QueryRewriter:
    settings = settings or get_settings()

    if settings.query_rewriter_provider == "none":
        return NoOpQueryRewriter()

    if settings.query_rewriter_provider == "openai":
        if settings.openai_api_key is None or not settings.openai_api_key.strip():
            raise ValueError("OPENAI_API_KEY is required for OpenAI query rewrite")
        return OpenAIQueryRewriter(
            api_key=settings.openai_api_key,
            model_name=settings.query_rewrite_model,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
            retry_delay_seconds=settings.openai_retry_delay_seconds,
            max_output_tokens=settings.query_rewrite_max_output_tokens,
        )

    raise ValueError(
        f"unsupported query rewriter provider: {settings.query_rewriter_provider}"
    )
