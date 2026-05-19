from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.db.repositories import (
    DocumentChunkSummary,
    DocumentDetailResult,
    DocumentListResult,
    DocumentSummary,
)
from backend.app.rag.reindex_embeddings import ReindexEmbeddingsStats
from ingestion.chunking import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_SIZE_TOKENS,
)


class CreateDocumentRequest(BaseModel):
    source_uri: str = Field(min_length=1, max_length=2048)
    markdown: str = Field(min_length=1, max_length=1_000_000)
    title: str | None = Field(default=None, max_length=512)
    author: str | None = Field(default=None, max_length=512)
    visibility: str | None = Field(default=None, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_size_tokens: int = Field(
        default=DEFAULT_CHUNK_SIZE_TOKENS,
        ge=1,
        le=4000,
    )
    chunk_overlap_tokens: int = Field(
        default=DEFAULT_CHUNK_OVERLAP_TOKENS,
        ge=0,
        le=1000,
    )

    @field_validator("source_uri", "markdown")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("title", "author", "visibility")
    @classmethod
    def optional_text_must_be_trimmed(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def chunk_overlap_must_be_smaller_than_size(self) -> "CreateDocumentRequest":
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                "chunk_overlap_tokens must be smaller than chunk_size_tokens"
            )
        return self


class CreateDocumentResponse(BaseModel):
    workspace_id: str
    document_id: str
    content_hash: str
    inserted: bool
    chunks_inserted: int = Field(ge=0)
    reason: str | None = None


class DocumentItem(BaseModel):
    id: str
    workspace_id: str
    source_type: str
    source_uri: str
    title: str
    author: str | None
    visibility: str
    metadata: dict[str, Any]
    chunk_count: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_summary(cls, document: DocumentSummary) -> "DocumentItem":
        return cls(
            id=str(document.id),
            workspace_id=document.workspace_id,
            source_type=document.source_type,
            source_uri=document.source_uri,
            title=document.title,
            author=document.author,
            visibility=document.visibility,
            metadata=dict(document.metadata),
            chunk_count=document.chunk_count,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )


class DocumentsResponse(BaseModel):
    workspace_id: str
    total: int = Field(ge=0)
    count: int = Field(ge=0)
    limit: int = Field(gt=0)
    offset: int = Field(ge=0)
    documents: list[DocumentItem]

    @classmethod
    def from_result(
        cls,
        *,
        workspace_id: str,
        limit: int,
        offset: int,
        result: DocumentListResult,
    ) -> "DocumentsResponse":
        documents = [
            DocumentItem.from_summary(document) for document in result.documents
        ]
        return cls(
            workspace_id=workspace_id,
            total=result.total,
            count=len(documents),
            limit=limit,
            offset=offset,
            documents=documents,
        )


class DocumentChunkItem(BaseModel):
    id: str
    document_id: str
    workspace_id: str
    chunk_index: int = Field(ge=0)
    text: str
    token_count: int = Field(ge=0)
    section_title: str | None
    page_number: int | None
    source_uri: str
    metadata: dict[str, Any]
    created_at: datetime

    @classmethod
    def from_summary(cls, chunk: DocumentChunkSummary) -> "DocumentChunkItem":
        return cls(
            id=str(chunk.id),
            document_id=str(chunk.document_id),
            workspace_id=chunk.workspace_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            token_count=chunk.token_count,
            section_title=chunk.section_title,
            page_number=chunk.page_number,
            source_uri=chunk.source_uri,
            metadata=dict(chunk.metadata),
            created_at=chunk.created_at,
        )


class DocumentDetailResponse(BaseModel):
    workspace_id: str
    document: DocumentItem
    chunks: list[DocumentChunkItem]

    @classmethod
    def from_result(
        cls,
        *,
        workspace_id: str,
        result: DocumentDetailResult,
    ) -> "DocumentDetailResponse":
        return cls(
            workspace_id=workspace_id,
            document=DocumentItem.from_summary(result.document),
            chunks=[DocumentChunkItem.from_summary(chunk) for chunk in result.chunks],
        )


class DeleteDocumentResponse(BaseModel):
    workspace_id: str
    document_id: str
    deleted: bool


class ReindexDocumentsRequest(BaseModel):
    source_uri: str | None = Field(default=None, max_length=2048)
    batch_size: int = Field(default=32, ge=1, le=256)
    limit: int | None = Field(default=None, ge=1, le=10000)
    dry_run: bool = True

    @field_validator("source_uri")
    @classmethod
    def source_uri_must_be_trimmed(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class ReindexDocumentsResponse(BaseModel):
    workspace_id: str
    source_uri: str | None
    model: str
    chunks_matched: int = Field(ge=0)
    chunks_embedded: int = Field(ge=0)
    chunks_updated: int = Field(ge=0)
    dry_run: bool
    elapsed_seconds: float = Field(ge=0)

    @classmethod
    def from_stats(
        cls,
        stats: ReindexEmbeddingsStats,
    ) -> "ReindexDocumentsResponse":
        return cls(
            workspace_id=stats.workspace_id,
            source_uri=stats.source_uri,
            model=stats.model_name,
            chunks_matched=stats.chunks_matched,
            chunks_embedded=stats.chunks_embedded,
            chunks_updated=stats.chunks_updated,
            dry_run=stats.dry_run,
            elapsed_seconds=stats.elapsed_seconds,
        )
