import uuid

import pytest

from backend.app.db.models import EMBEDDING_DIMENSION
from backend.app.rag.embeddings import EmbeddingDimensionError
from backend.app.rag.vector_retrieval import (
    VectorRetriever,
    build_vector_retrieval_statement,
)


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class FakeAsyncSession:
    def __init__(self, rows):
        self.rows = rows
        self.statement = None

    async def execute(self, statement):
        self.statement = statement
        return FakeResult(self.rows)


class FakeRow:
    def __init__(self, **mapping):
        self._mapping = mapping


def make_query_embedding() -> list[float]:
    return [0.0] * (EMBEDDING_DIMENSION - 1) + [1.0]


def test_vector_statement_filters_workspace_and_orders_distance() -> None:
    statement = build_vector_retrieval_statement(
        make_query_embedding(),
        top_k=5,
        workspace_id="public",
    )
    compiled = str(statement)

    assert "document_chunks.workspace_id" in compiled
    assert "document_chunks.embedding IS NOT NULL" in compiled
    assert "document_chunks.embedding <=>" in compiled
    assert "ORDER BY document_chunks.embedding <=>" in compiled


def test_build_vector_retrieval_statement_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        build_vector_retrieval_statement(
            make_query_embedding(),
            top_k=0,
            workspace_id="public",
        )


def test_build_vector_retrieval_statement_rejects_wrong_dimension() -> None:
    with pytest.raises(EmbeddingDimensionError, match="query_embedding"):
        build_vector_retrieval_statement(
            [0.1, 0.2],
            top_k=5,
            workspace_id="public",
        )


@pytest.mark.asyncio
async def test_vector_retriever_maps_rows_to_retrieved_chunks() -> None:
    chunk_id = uuid.uuid4()
    document_id = uuid.uuid4()
    session = FakeAsyncSession(
        [
            FakeRow(
                chunk_id=chunk_id,
                document_id=document_id,
                text="FlashAttention reduces memory traffic.",
                title="FlashAttention Notes",
                section_title="FlashAttention",
                source_uri="llm_systems/flashattention.md",
                score=0.98,
                metadata={"topic": "attention"},
            )
        ]
    )
    retriever = VectorRetriever(session)  # type: ignore[arg-type]

    results = await retriever.retrieve(
        query_embedding=make_query_embedding(),
        top_k=3,
        workspace_id="public",
    )

    assert len(results) == 1
    assert results[0].chunk_id == str(chunk_id)
    assert results[0].document_id == str(document_id)
    assert results[0].rank == 1
    assert results[0].retrieval_mode == "vector"
    assert results[0].metadata == {"topic": "attention"}
    assert session.statement is not None
