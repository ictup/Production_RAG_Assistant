from collections.abc import Sequence
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import EMBEDDING_DIMENSION, Document, DocumentChunk
from backend.app.rag.embeddings import validate_embedding_dimension
from backend.app.rag.metadata_filters import normalize_metadata_filter
from backend.app.rag.retrieval_models import RetrievedChunk


def build_vector_retrieval_statement(
    query_embedding: Sequence[float],
    *,
    top_k: int,
    workspace_id: str,
    metadata_filter: dict[str, Any] | None = None,
) -> Select[Any]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    validate_embedding_dimension(
        query_embedding,
        expected_dimension=EMBEDDING_DIMENSION,
        label="query_embedding",
    )

    normalized_metadata_filter = normalize_metadata_filter(metadata_filter)
    distance = DocumentChunk.embedding.cosine_distance(list(query_embedding))
    score = (1 - distance).label("score")

    statement = (
        select(
            DocumentChunk.id.label("chunk_id"),
            DocumentChunk.document_id.label("document_id"),
            DocumentChunk.text.label("text"),
            DocumentChunk.source_uri.label("source_uri"),
            DocumentChunk.section_title.label("section_title"),
            DocumentChunk.metadata_.label("metadata"),
            Document.title.label("title"),
            score,
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.workspace_id == workspace_id)
        .where(DocumentChunk.embedding.is_not(None))
        .order_by(distance)
        .limit(top_k)
    )
    if normalized_metadata_filter:
        statement = statement.where(
            DocumentChunk.metadata_.contains(normalized_metadata_filter)
        )
    return statement


class VectorRetriever:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def retrieve(
        self,
        *,
        query_embedding: Sequence[float],
        top_k: int,
        workspace_id: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        statement = build_vector_retrieval_statement(
            query_embedding,
            top_k=top_k,
            workspace_id=workspace_id,
            metadata_filter=metadata_filter,
        )
        rows = (await self.session.execute(statement)).all()

        retrieved_chunks: list[RetrievedChunk] = []
        for rank, row in enumerate(rows, start=1):
            mapping = row._mapping
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=str(mapping["chunk_id"]),
                    document_id=str(mapping["document_id"]),
                    text=mapping["text"],
                    title=mapping["title"],
                    section_title=mapping["section_title"],
                    source_uri=mapping["source_uri"],
                    score=float(mapping["score"]),
                    rank=rank,
                    retrieval_mode="vector",
                    metadata=dict(mapping["metadata"] or {}),
                )
            )

        return retrieved_chunks
