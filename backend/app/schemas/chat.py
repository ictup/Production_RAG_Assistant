from pydantic import BaseModel, Field, field_validator

from backend.app.rag.citations import Source
from backend.app.rag.pipeline import (
    ChatPipelineRequest,
    ChatPipelineResponse,
    RetrievalInfo,
    UsageInfo,
)
from backend.app.rag.refusal import RefusalInfo


class ChatRequest(BaseModel):
    question: str
    vector_top_k: int | None = Field(default=None, gt=0)
    sparse_top_k: int | None = Field(default=None, gt=0)
    fused_top_k: int | None = Field(default=None, gt=0)
    rerank_top_n: int | None = Field(default=None, gt=0)
    rerank: bool = True

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value

    def to_pipeline_request(self, *, workspace_id: str) -> ChatPipelineRequest:
        return ChatPipelineRequest(
            question=self.question,
            workspace_id=workspace_id,
            vector_top_k=self.vector_top_k,
            sparse_top_k=self.sparse_top_k,
            fused_top_k=self.fused_top_k,
            rerank_top_n=self.rerank_top_n,
            rerank=self.rerank,
        )


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieval: RetrievalInfo
    usage: UsageInfo
    request_id: str
    citation_valid: bool | None
    refusal: RefusalInfo | None = None

    @classmethod
    def from_pipeline_response(
        cls,
        response: ChatPipelineResponse,
        *,
        request_id: str,
    ) -> "ChatResponse":
        return cls(
            answer=response.answer,
            sources=response.sources,
            retrieval=response.retrieval,
            usage=response.usage,
            request_id=request_id,
            citation_valid=response.citation_valid,
            refusal=response.refusal,
        )
