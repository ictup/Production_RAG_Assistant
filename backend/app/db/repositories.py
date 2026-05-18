import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import EMBEDDING_DIMENSION, Document, DocumentChunk
from backend.app.rag.embeddings import validate_embedding_batch
from ingestion.hashing import compute_content_hash
from ingestion.models import Chunk, RawDocument


@dataclass(frozen=True)
class IngestDocumentResult:
    document_id: uuid.UUID
    content_hash: str
    inserted: bool
    chunks_inserted: int
    reason: str | None = None


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_document_id_by_hash(self, content_hash: str) -> uuid.UUID | None:
        statement = select(Document.id).where(Document.content_hash == content_hash)
        return await self.session.scalar(statement)

    async def ingest_document(
        self,
        raw_document: RawDocument,
        chunks: Sequence[Chunk],
        *,
        content_hash: str | None = None,
        chunk_embeddings: Sequence[Sequence[float]] | None = None,
    ) -> IngestDocumentResult:
        if not chunks:
            raise ValueError("cannot ingest a document without chunks")

        resolved_hash = content_hash or compute_content_hash(raw_document.text)
        existing_document_id = await self.get_document_id_by_hash(resolved_hash)
        if existing_document_id is not None:
            return IngestDocumentResult(
                document_id=existing_document_id,
                content_hash=resolved_hash,
                inserted=False,
                chunks_inserted=0,
                reason="duplicate_content_hash",
            )

        if chunk_embeddings is None:
            raise ValueError("chunk_embeddings are required for new documents")
        if len(chunk_embeddings) != len(chunks):
            raise ValueError(
                f"chunk_embeddings count {len(chunk_embeddings)} does not match "
                f"chunks count {len(chunks)}"
            )
        validate_embedding_batch(
            chunk_embeddings,
            expected_dimension=EMBEDDING_DIMENSION,
        )

        document_id = uuid.uuid4()
        document = Document(
            id=document_id,
            workspace_id=raw_document.workspace_id,
            source_type=raw_document.source_type,
            source_uri=raw_document.source_uri,
            title=raw_document.title,
            author=raw_document.author,
            content_hash=resolved_hash,
            visibility=raw_document.visibility,
            metadata_=dict(raw_document.metadata),
        )
        chunk_models = [
            self._build_chunk_model(
                document_id=document_id,
                chunk=chunk,
                embedding=list(embedding),
            )
            for chunk, embedding in zip(chunks, chunk_embeddings, strict=True)
        ]

        self.session.add(document)
        self.session.add_all(chunk_models)
        await self.session.flush()

        return IngestDocumentResult(
            document_id=document_id,
            content_hash=resolved_hash,
            inserted=True,
            chunks_inserted=len(chunk_models),
        )

    @staticmethod
    def _build_chunk_model(
        *,
        document_id: uuid.UUID,
        chunk: Chunk,
        embedding: list[float],
    ) -> DocumentChunk:
        return DocumentChunk(
            id=uuid.uuid4(),
            document_id=document_id,
            workspace_id=chunk.workspace_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            token_count=chunk.token_count,
            section_title=chunk.section_title,
            source_uri=chunk.source_uri,
            embedding=embedding,
            metadata_=dict(chunk.metadata),
        )
