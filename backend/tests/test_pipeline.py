import uuid

import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings
from backend.app.rag.embeddings import FakeEmbeddingClient
from backend.app.rag.generation import FakeGenerator
from backend.app.rag.pipeline import ChatPipelineRequest, RagPipeline
from backend.app.rag.refusal import REFUSAL_ANSWER
from backend.app.rag.reranking import NoOpReranker


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class FakeAsyncSession:
    def __init__(self, row_batches):
        self.row_batches = list(row_batches)
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        return FakeResult(self.row_batches.pop(0))


class FakeRow:
    def __init__(self, **mapping):
        self._mapping = mapping


def make_settings() -> Settings:
    return Settings(
        embedding_provider="fake",
        embedding_model="test-fake-embedding",
        embedding_dimension=1536,
        reranker_provider="none",
        rerank_top_n=1,
        vector_top_k=3,
        sparse_top_k=3,
        fused_top_k=3,
        rrf_k=60,
        generator_provider="fake",
        llm_model="test-fake-llm",
        refusal_score_threshold=0.01,
    )


def make_row(*, chunk_id: uuid.UUID, document_id: uuid.UUID) -> FakeRow:
    return FakeRow(
        chunk_id=chunk_id,
        document_id=document_id,
        text=(
            "FlashAttention reduces memory traffic between HBM and SRAM by "
            "tiling exact attention."
        ),
        title="FlashAttention Notes",
        section_title="FlashAttention",
        source_uri="llm_systems/flashattention.md",
        score=0.98,
        metadata={"topic": "attention"},
    )


@pytest.mark.asyncio
async def test_pipeline_answers_with_sources_and_valid_citations() -> None:
    chunk_id = uuid.uuid4()
    document_id = uuid.uuid4()
    session = FakeAsyncSession(
        [
            [make_row(chunk_id=chunk_id, document_id=document_id)],
            [make_row(chunk_id=chunk_id, document_id=document_id)],
        ]
    )
    pipeline = RagPipeline(
        session=session,  # type: ignore[arg-type]
        settings=make_settings(),
        embedding_client=FakeEmbeddingClient(
            dimension=1536,
            model_name="test-fake-embedding",
        ),
        reranker=NoOpReranker(),
        generator=FakeGenerator(model_name="test-fake-llm"),
    )

    response = await pipeline.answer_question(
        ChatPipelineRequest(question="What problem does FlashAttention solve?")
    )

    assert "FlashAttention reduces memory traffic" in response.answer
    assert response.answer.endswith("[1]")
    assert response.citation_valid is True
    assert response.refusal is None
    assert response.retrieval.mode == "hybrid_rrf_rerank"
    assert response.retrieval.fused_count == 1
    assert response.retrieval.used_count == 1
    assert response.usage.model == "test-fake-llm"
    assert response.usage.embedding_model == "test-fake-embedding"
    assert response.sources[0].chunk_id == str(chunk_id)
    assert len(session.statements) == 2


@pytest.mark.asyncio
async def test_pipeline_refuses_when_retrieval_returns_no_chunks() -> None:
    session = FakeAsyncSession([[], []])
    pipeline = RagPipeline(
        session=session,  # type: ignore[arg-type]
        settings=make_settings(),
        embedding_client=FakeEmbeddingClient(
            dimension=1536,
            model_name="test-fake-embedding",
        ),
        reranker=NoOpReranker(),
        generator=FakeGenerator(model_name="test-fake-llm"),
    )

    response = await pipeline.answer_question(
        ChatPipelineRequest(question="What problem does FlashAttention solve?")
    )

    assert response.answer == REFUSAL_ANSWER
    assert response.sources == []
    assert response.citation_valid is None
    assert response.refusal is not None
    assert response.refusal.reason == "no_retrieved_chunks"
    assert response.retrieval.used_count == 0
    assert response.usage.input_tokens == 0
    assert response.usage.output_tokens == 0


def test_pipeline_request_trims_question_and_workspace() -> None:
    request = ChatPipelineRequest(
        question="  What is RAG?  ",
        workspace_id="  public  ",
    )

    assert request.question == "What is RAG?"
    assert request.workspace_id == "public"


def test_pipeline_request_rejects_blank_question() -> None:
    with pytest.raises(ValidationError, match="value must not be blank"):
        ChatPipelineRequest(question="   ")
