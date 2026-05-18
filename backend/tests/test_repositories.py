import uuid
from typing import Any

import pytest

from backend.app.db.models import Document, DocumentChunk
from backend.app.db.repositories import DocumentRepository
from ingestion.chunking import chunk_document
from ingestion.hashing import compute_content_hash
from ingestion.models import RawDocument


class FakeAsyncSession:
    def __init__(self, scalar_result: uuid.UUID | None = None) -> None:
        self.scalar_result = scalar_result
        self.scalar_statement: Any | None = None
        self.added: list[Any] = []
        self.added_all: list[Any] = []
        self.flushed = False

    async def scalar(self, statement: Any) -> uuid.UUID | None:
        self.scalar_statement = statement
        return self.scalar_result

    def add(self, instance: Any) -> None:
        self.added.append(instance)

    def add_all(self, instances: list[Any]) -> None:
        self.added_all.extend(instances)

    async def flush(self) -> None:
        self.flushed = True


def make_raw_document() -> RawDocument:
    return RawDocument(
        title="FlashAttention Notes",
        source_uri="data/raw/flashattention.md",
        text="# FlashAttention\n\nFlashAttention reduces HBM traffic.",
        metadata={"topic": "attention"},
        author="Dao et al.",
    )


def make_embedding(dimension: int = 1536) -> list[float]:
    return [0.0] * (dimension - 1) + [1.0]


@pytest.mark.asyncio
async def test_get_document_id_by_hash_queries_document_hash() -> None:
    existing_id = uuid.uuid4()
    session = FakeAsyncSession(scalar_result=existing_id)
    repository = DocumentRepository(session)  # type: ignore[arg-type]

    result = await repository.get_document_id_by_hash("a" * 64)

    assert result == existing_id
    assert session.scalar_statement is not None
    assert "documents.content_hash" in str(session.scalar_statement)


@pytest.mark.asyncio
async def test_ingest_document_skips_existing_content_hash() -> None:
    existing_id = uuid.uuid4()
    session = FakeAsyncSession(scalar_result=existing_id)
    repository = DocumentRepository(session)  # type: ignore[arg-type]
    raw_document = make_raw_document()
    chunks = chunk_document(raw_document, chunk_size_tokens=40, chunk_overlap_tokens=5)

    result = await repository.ingest_document(raw_document, chunks)

    assert result.document_id == existing_id
    assert result.inserted is False
    assert result.chunks_inserted == 0
    assert result.reason == "duplicate_content_hash"
    assert session.added == []
    assert session.added_all == []
    assert session.flushed is False


@pytest.mark.asyncio
async def test_ingest_document_adds_document_and_chunks() -> None:
    session = FakeAsyncSession()
    repository = DocumentRepository(session)  # type: ignore[arg-type]
    raw_document = make_raw_document()
    chunks = chunk_document(raw_document, chunk_size_tokens=40, chunk_overlap_tokens=5)
    embeddings = [make_embedding() for _ in chunks]

    result = await repository.ingest_document(
        raw_document,
        chunks,
        chunk_embeddings=embeddings,
    )

    assert result.inserted is True
    assert result.chunks_inserted == len(chunks)
    assert result.content_hash == compute_content_hash(raw_document.text)
    assert session.flushed is True

    document = session.added[0]
    assert isinstance(document, Document)
    assert document.id == result.document_id
    assert document.title == raw_document.title
    assert document.content_hash == result.content_hash
    assert document.metadata_ == {"topic": "attention"}

    chunk_model = session.added_all[0]
    assert isinstance(chunk_model, DocumentChunk)
    assert chunk_model.document_id == result.document_id
    assert chunk_model.chunk_index == chunks[0].chunk_index
    assert chunk_model.text == chunks[0].text
    assert chunk_model.embedding == embeddings[0]
    assert chunk_model.metadata_ == {"topic": "attention"}


@pytest.mark.asyncio
async def test_ingest_document_requires_embeddings_for_new_document() -> None:
    session = FakeAsyncSession()
    repository = DocumentRepository(session)  # type: ignore[arg-type]
    raw_document = make_raw_document()
    chunks = chunk_document(raw_document, chunk_size_tokens=40, chunk_overlap_tokens=5)

    with pytest.raises(ValueError, match="chunk_embeddings are required"):
        await repository.ingest_document(raw_document, chunks)


@pytest.mark.asyncio
async def test_ingest_document_rejects_embedding_count_mismatch() -> None:
    session = FakeAsyncSession()
    repository = DocumentRepository(session)  # type: ignore[arg-type]
    raw_document = make_raw_document()
    chunks = chunk_document(raw_document, chunk_size_tokens=40, chunk_overlap_tokens=5)

    with pytest.raises(ValueError, match="does not match"):
        await repository.ingest_document(
            raw_document,
            chunks,
            chunk_embeddings=[],
        )


@pytest.mark.asyncio
async def test_ingest_document_rejects_embedding_dimension_mismatch() -> None:
    session = FakeAsyncSession()
    repository = DocumentRepository(session)  # type: ignore[arg-type]
    raw_document = make_raw_document()
    chunks = chunk_document(raw_document, chunk_size_tokens=40, chunk_overlap_tokens=5)

    with pytest.raises(ValueError, match="does not match"):
        await repository.ingest_document(
            raw_document,
            chunks,
            chunk_embeddings=[[0.1, 0.2] for _ in chunks],
        )


@pytest.mark.asyncio
async def test_ingest_document_rejects_empty_chunk_list() -> None:
    session = FakeAsyncSession()
    repository = DocumentRepository(session)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="without chunks"):
        await repository.ingest_document(make_raw_document(), [])
