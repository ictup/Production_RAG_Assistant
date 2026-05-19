import json
import uuid

import httpx
import pytest

from backend.app.core.config import Settings
from backend.app.rag.rerank_smoke import build_rerank_smoke_settings
from backend.app.rag.reranking import (
    NoOpReranker,
    OpenAIListwiseReranker,
    OpenAIRerankingError,
    Reranker,
    build_rerank_prompt,
    build_reranker,
    parse_ranked_indices,
)
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


def test_build_rerank_prompt_includes_query_and_candidate_context() -> None:
    chunk = make_chunk("flash", 1).model_copy(
        update={
            "text": "FlashAttention reduces memory traffic.",
            "section_title": "Attention kernels",
        }
    )

    prompt = build_rerank_prompt(
        query="What problem does FlashAttention solve?",
        chunks=[chunk],
    )

    assert "Query:\n\nWhat problem does FlashAttention solve?" not in prompt
    assert "Query:\nWhat problem does FlashAttention solve?" in prompt
    assert "[1]" in prompt
    assert "Title: title for flash" in prompt
    assert "Section: Attention kernels" in prompt
    assert "Source: flash.md" in prompt
    assert "FlashAttention reduces memory traffic." in prompt


def test_parse_ranked_indices_reads_json_object_and_appends_missing() -> None:
    order = parse_ranked_indices(
        '{"ranked_indices":[3,1]}',
        candidate_count=4,
    )

    assert order == [2, 0, 1, 3]


def test_parse_ranked_indices_filters_duplicates_and_invalid_values() -> None:
    order = parse_ranked_indices(
        '{"ranked_indices":[2,2,99,0,"3",{"candidate_id":1},true]}',
        candidate_count=3,
    )

    assert order == [1, 2, 0]


def test_parse_ranked_indices_accepts_json_inside_text() -> None:
    order = parse_ranked_indices(
        '```json\n{"ranked_indices":[2,1]}\n```',
        candidate_count=2,
    )

    assert order == [1, 0]


def test_parse_ranked_indices_falls_back_to_original_order_for_bad_json() -> None:
    assert parse_ranked_indices("not json", candidate_count=3) == [0, 1, 2]


@pytest.mark.asyncio
async def test_openai_listwise_reranker_sends_request_and_reorders_chunks() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url == "https://api.openai.test/v1/responses"
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json_from_request(request)
        assert payload["model"] == "gpt-rerank"
        assert payload["max_output_tokens"] == 123
        assert payload["store"] is False
        assert "retrieval reranker" in payload["instructions"]
        assert "FlashAttention" in payload["input"]
        assert "[3]" in payload["input"]
        return httpx.Response(
            200,
            json={"output_text": '{"ranked_indices":[3,1,2]}'},
        )

    chunks = [make_chunk("a", 1), make_chunk("b", 2), make_chunk("c", 3)]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        reranker = OpenAIListwiseReranker(
            api_key="test-key",
            model_name="gpt-rerank",
            base_url="https://api.openai.test/v1/",
            max_output_tokens=123,
            http_client=http_client,
        )

        reranked = await reranker.rerank(
            query="FlashAttention",
            chunks=chunks,
            top_n=2,
        )

    assert [chunk.chunk_id for chunk in reranked] == ["c", "a"]
    assert [chunk.rank for chunk in reranked] == [1, 2]
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_openai_listwise_reranker_rejects_http_errors() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as http_client:
        reranker = OpenAIListwiseReranker(
            api_key="test-key",
            model_name="gpt-rerank",
            max_retries=0,
            http_client=http_client,
        )

        with pytest.raises(OpenAIRerankingError, match="status 429") as exc_info:
            await reranker.rerank(
                query="FlashAttention",
                chunks=[make_chunk("a", 1)],
                top_n=1,
            )

    assert exc_info.value.category == "rate_limit"
    assert exc_info.value.retryable is True
    assert exc_info.value.status_code == 429


def test_build_reranker_requires_openai_api_key() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_reranker(
            Settings(
                reranker_provider="openai",
                openai_api_key=None,
            )
        )


def test_build_reranker_uses_openai_settings() -> None:
    reranker = build_reranker(
        Settings(
            reranker_provider="openai",
            reranker_model="gpt-rerank",
            openai_api_key="test-key",
            openai_base_url="https://api.openai.test/v1",
            openai_max_retries=3,
            openai_retry_delay_seconds=0,
            openai_max_output_tokens=123,
        )
    )

    assert isinstance(reranker, OpenAIListwiseReranker)
    assert reranker.model_name == "gpt-rerank"
    assert reranker.max_retries == 3
    assert reranker.retry_delay_seconds == 0
    assert reranker.max_output_tokens == 123


def test_build_rerank_smoke_settings_applies_runtime_overrides() -> None:
    settings = build_rerank_smoke_settings(
        Settings(
            reranker_provider="none",
            reranker_model="gpt-default",
            openai_api_key="test-key",
        ),
        reranker_provider="openai",
        reranker_model="gpt-rerank",
    )

    assert settings.reranker_provider == "openai"
    assert settings.reranker_model == "gpt-rerank"


def json_from_request(request: httpx.Request) -> dict:
    return json.loads(request.content.decode("utf-8"))
