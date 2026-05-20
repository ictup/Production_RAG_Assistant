import uuid

import pytest
from sqlalchemy.dialects import postgresql

from backend.app.rag.sparse_retrieval import (
    SparseRetriever,
    build_sparse_retrieval_statement,
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


def test_sparse_statement_uses_websearch_query_rank_and_workspace_filter() -> None:
    statement = build_sparse_retrieval_statement(
        "KV cache",
        top_k=5,
        workspace_id="public",
    )
    compiled = str(statement)

    assert "websearch_to_tsquery" in compiled
    assert "ts_rank_cd" in compiled
    assert "@@" in compiled
    assert "document_chunks.workspace_id" in compiled
    assert "ORDER BY score DESC" in compiled


def test_sparse_statement_applies_metadata_filter() -> None:
    statement = build_sparse_retrieval_statement(
        "KV cache",
        top_k=5,
        workspace_id="public",
        metadata_filter={"topic": "inference"},
    )
    compiled = str(statement.compile(dialect=postgresql.dialect()))

    assert "document_chunks.metadata" in compiled
    assert "@>" in compiled


def test_sparse_statement_rejects_blank_query() -> None:
    with pytest.raises(ValueError, match="query must not be blank"):
        build_sparse_retrieval_statement("   ", top_k=5, workspace_id="public")


def test_sparse_statement_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        build_sparse_retrieval_statement("KV cache", top_k=0, workspace_id="public")


@pytest.mark.asyncio
async def test_sparse_retriever_maps_rows_to_retrieved_chunks() -> None:
    chunk_id = uuid.uuid4()
    document_id = uuid.uuid4()
    session = FakeAsyncSession(
        [
            FakeRow(
                chunk_id=chunk_id,
                document_id=document_id,
                text="PagedAttention stores KV cache in pages.",
                title="PagedAttention Notes",
                section_title="PagedAttention",
                source_uri="llm_systems/pagedattention.md",
                score=0.42,
                metadata={"topic": "inference"},
            )
        ]
    )
    retriever = SparseRetriever(session)  # type: ignore[arg-type]

    results = await retriever.retrieve(
        query="KV cache",
        top_k=3,
        workspace_id="public",
    )

    assert len(results) == 1
    assert results[0].chunk_id == str(chunk_id)
    assert results[0].document_id == str(document_id)
    assert results[0].rank == 1
    assert results[0].retrieval_mode == "sparse"
    assert results[0].metadata == {"topic": "inference"}
    assert session.statement is not None
