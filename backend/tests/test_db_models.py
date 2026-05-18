from pgvector.sqlalchemy import Vector

from backend.app.db.models import EMBEDDING_DIMENSION, Base, Document, DocumentChunk


def test_base_metadata_contains_core_document_tables() -> None:
    assert set(Base.metadata.tables) == {"documents", "document_chunks"}


def test_document_metadata_column_uses_safe_python_attribute_name() -> None:
    assert Document.metadata_.name == "metadata"
    assert Document.__table__.c["metadata"] is Document.metadata_.property.columns[0]


def test_document_chunk_embedding_column_uses_expected_dimension() -> None:
    embedding_type = DocumentChunk.__table__.c.embedding.type

    assert isinstance(embedding_type, Vector)
    assert embedding_type.dim == EMBEDDING_DIMENSION


def test_document_chunk_has_access_and_search_indexes() -> None:
    index_names = {index.name for index in DocumentChunk.__table__.indexes}

    assert "document_chunks_embedding_hnsw" in index_names
    assert "document_chunks_search_vector_idx" in index_names
    assert "document_chunks_metadata_idx" in index_names
    assert "document_chunks_workspace_idx" in index_names


def test_document_chunk_cascades_when_document_is_deleted() -> None:
    foreign_keys = list(DocumentChunk.__table__.c.document_id.foreign_keys)

    assert len(foreign_keys) == 1
    assert foreign_keys[0].ondelete == "CASCADE"
