from typing import Any

from pydantic import BaseModel, Field, field_validator

from backend.app.rag.metadata_filters import normalize_metadata_filter
from backend.app.rag.pipeline import ChatPipelineRequest, RagPipeline


class RAGSearchInput(BaseModel):
    query: str
    workspace_id: str = "public"
    top_k: int = Field(default=5, ge=1, le=50)
    metadata_filter: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query", "workspace_id")
    @classmethod
    def value_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("metadata_filter", mode="before")
    @classmethod
    def metadata_filter_must_be_object(cls, value: object) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("metadata_filter must be an object")
        return normalize_metadata_filter(value)


class RAGSearchOutput(BaseModel):
    sources: list[dict[str, Any]]
    context: str
    top_score: float | None
    refusal_recommended: bool
    retrieval: dict[str, Any] = Field(default_factory=dict)


async def rag_search_tool(
    inp: RAGSearchInput,
    *,
    pipeline: RagPipeline,
) -> RAGSearchOutput:
    retrieved = await pipeline.retrieve_context(
        ChatPipelineRequest(
            question=inp.query,
            workspace_id=inp.workspace_id,
            metadata_filter=inp.metadata_filter,
            vector_top_k=inp.top_k,
            sparse_top_k=inp.top_k,
            fused_top_k=inp.top_k,
            rerank_top_n=inp.top_k,
            rerank=True,
        )
    )
    return RAGSearchOutput(
        sources=[source.model_dump(mode="json") for source in retrieved.sources],
        context=retrieved.context,
        top_score=retrieved.retrieval.top_score,
        refusal_recommended=retrieved.refusal is not None,
        retrieval=retrieved.retrieval.model_dump(mode="json"),
    )

