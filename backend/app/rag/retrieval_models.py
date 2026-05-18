from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    title: str
    section_title: str | None = None
    source_uri: str
    score: float
    rank: int = Field(ge=1)
    retrieval_mode: Literal["vector", "sparse", "hybrid_rrf"]
    metadata: dict[str, Any] = Field(default_factory=dict)
