from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import Document, DocumentChunk
from backend.app.rag.retrieval_models import RetrievedChunk


def build_sparse_retrieval_statement(
    query: str,
    *,
    top_k: int,
    workspace_id: str,
) -> Select[Any]:
    query = query.strip()
    if not query:
        raise ValueError("query must not be blank")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    ts_query = func.websearch_to_tsquery("english", query)
    score = func.ts_rank_cd(DocumentChunk.search_vector, ts_query).label("score")

    return (
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
        .where(DocumentChunk.search_vector.bool_op("@@")(ts_query))
        .order_by(score.desc())
        .limit(top_k)
    )


class SparseRetriever:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int,
        workspace_id: str,
    ) -> list[RetrievedChunk]:
        statement = build_sparse_retrieval_statement(
            query,
            top_k=top_k,
            workspace_id=workspace_id,
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
                    retrieval_mode="sparse",
                    metadata=dict(mapping["metadata"] or {}),
                )
            )

        return retrieved_chunks

