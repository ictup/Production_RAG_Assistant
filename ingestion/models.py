from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class RawDocument(BaseModel):
    title: str
    source_type: Literal["markdown", "pdf", "html"] = "markdown"
    source_uri: str
    text: str
    workspace_id: str = "public"
    visibility: str = "public"
    metadata: dict[str, Any] = Field(default_factory=dict)
    author: str | None = None

    @field_validator("title", "source_uri", "text", "workspace_id", "visibility")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class Chunk(BaseModel):
    document_id: str
    chunk_index: int = Field(ge=0)
    text: str
    section_title: str | None = None
    token_count: int = Field(gt=0)
    source_uri: str
    workspace_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("document_id", "text", "source_uri", "workspace_id")
    @classmethod
    def chunk_fields_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value
