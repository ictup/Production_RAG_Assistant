import json

import httpx
import pytest

from backend.app.core.config import Settings
from backend.app.rag.query_rewriting import (
    NoOpQueryRewriter,
    OpenAIQueryRewriteError,
    OpenAIQueryRewriter,
    QueryRewriter,
    build_query_rewrite_prompt,
    build_query_rewriter,
    normalize_rewritten_query,
)


@pytest.mark.asyncio
async def test_noop_query_rewriter_returns_original_question() -> None:
    rewriter = NoOpQueryRewriter()

    result = await rewriter.rewrite(question="  What is FlashAttention?  ")

    assert result.original_query == "What is FlashAttention?"
    assert result.rewritten_query == "What is FlashAttention?"
    assert result.provider_name == "none"
    assert result.model_name == "none"
    assert result.rewritten is False


def test_build_query_rewrite_prompt_includes_metadata_filter() -> None:
    prompt = build_query_rewrite_prompt(
        question="What does it solve?",
        metadata_filter={"topic": "attention"},
    )

    assert "Question:\nWhat does it solve?" in prompt
    assert '"topic": "attention"' in prompt
    assert "Rewrite as one concise retrieval query." in prompt


def test_normalize_rewritten_query_strips_wrapping_and_caps_length() -> None:
    rewritten = normalize_rewritten_query(
        '```text\n"FlashAttention memory traffic"\n```',
        fallback_query="fallback",
    )

    assert rewritten == "FlashAttention memory traffic"


def test_normalize_rewritten_query_falls_back_for_blank_output() -> None:
    assert normalize_rewritten_query("  ", fallback_query="fallback") == "fallback"


@pytest.mark.asyncio
async def test_openai_query_rewriter_sends_request_and_rewrites_query() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url == "https://api.openai.test/v1/responses"
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json_from_request(request)
        assert payload["model"] == "gpt-rewrite"
        assert payload["max_output_tokens"] == 32
        assert payload["store"] is False
        assert "RAG retriever" in payload["instructions"]
        assert "FlashAttention" in payload["input"]
        return httpx.Response(
            200,
            json={"output_text": "FlashAttention memory traffic HBM SRAM"},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        rewriter = OpenAIQueryRewriter(
            api_key="test-key",
            model_name="gpt-rewrite",
            base_url="https://api.openai.test/v1/",
            max_output_tokens=32,
            http_client=http_client,
        )

        result = await rewriter.rewrite(
            question="What problem does FlashAttention solve?",
            metadata_filter={"topic": "attention"},
        )

    assert result.original_query == "What problem does FlashAttention solve?"
    assert result.rewritten_query == "FlashAttention memory traffic HBM SRAM"
    assert result.provider_name == "openai"
    assert result.model_name == "gpt-rewrite"
    assert result.rewritten is True
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_openai_query_rewriter_rejects_http_errors() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        rewriter = OpenAIQueryRewriter(
            api_key="test-key",
            model_name="gpt-rewrite",
            max_retries=0,
            http_client=http_client,
        )

        with pytest.raises(OpenAIQueryRewriteError, match="status 429") as exc_info:
            await rewriter.rewrite(question="What is FlashAttention?")

    assert exc_info.value.category == "rate_limit"
    assert exc_info.value.retryable is True
    assert exc_info.value.status_code == 429


def test_build_query_rewriter_uses_settings() -> None:
    rewriter = build_query_rewriter(Settings(query_rewriter_provider="none"))

    assert isinstance(rewriter, QueryRewriter)
    assert rewriter.provider_name == "none"


def test_build_query_rewriter_requires_openai_api_key() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_query_rewriter(
            Settings(
                query_rewriter_provider="openai",
                openai_api_key=None,
            )
        )


def test_build_query_rewriter_uses_openai_settings() -> None:
    rewriter = build_query_rewriter(
        Settings(
            query_rewriter_provider="openai",
            query_rewrite_model="gpt-rewrite",
            query_rewrite_max_output_tokens=32,
            openai_api_key="test-key",
            openai_base_url="https://api.openai.test/v1",
            openai_max_retries=3,
            openai_retry_delay_seconds=0,
        )
    )

    assert isinstance(rewriter, OpenAIQueryRewriter)
    assert rewriter.model_name == "gpt-rewrite"
    assert rewriter.max_output_tokens == 32
    assert rewriter.max_retries == 3
    assert rewriter.retry_delay_seconds == 0


def json_from_request(request: httpx.Request) -> dict:
    return json.loads(request.content.decode("utf-8"))
