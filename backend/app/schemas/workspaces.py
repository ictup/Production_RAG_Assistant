from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from backend.app.db.models import Workspace
from backend.app.db.repositories import CreateWorkspaceResult, WorkspaceListResult


class CreateWorkspaceRequest(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    name: str | None = Field(default=None, max_length=256)
    description: str | None = Field(default=None, max_length=2048)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def workspace_id_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("workspace id must not be blank")
        return value

    @field_validator("name", "description")
    @classmethod
    def optional_text_must_be_trimmed(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class UpdateWorkspaceRequest(BaseModel):
    name: str | None = Field(default=None, max_length=256)
    description: str | None = Field(default=None, max_length=2048)
    metadata: dict[str, Any] | None = None

    @field_validator("name", "description")
    @classmethod
    def optional_text_must_be_trimmed(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class ArchiveWorkspaceRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=2048)

    @field_validator("reason")
    @classmethod
    def optional_reason_must_be_trimmed(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class WorkspaceItem(BaseModel):
    id: str
    name: str | None
    description: str | None
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None
    archived_reason: str | None

    @classmethod
    def from_model(cls, workspace: Workspace) -> "WorkspaceItem":
        return cls(
            id=workspace.id,
            name=workspace.name,
            description=workspace.description,
            metadata=dict(workspace.metadata_),
            created_at=workspace.created_at,
            updated_at=workspace.updated_at,
            archived_at=workspace.archived_at,
            archived_reason=workspace.archived_reason,
        )


class CreateWorkspaceResponse(BaseModel):
    workspace: WorkspaceItem
    created: bool

    @classmethod
    def from_result(cls, result: CreateWorkspaceResult) -> "CreateWorkspaceResponse":
        return cls(
            workspace=WorkspaceItem.from_model(result.workspace),
            created=result.created,
        )


class WorkspacesResponse(BaseModel):
    total: int = Field(ge=0)
    count: int = Field(ge=0)
    limit: int = Field(gt=0)
    offset: int = Field(ge=0)
    workspaces: list[WorkspaceItem]

    @classmethod
    def from_result(
        cls,
        *,
        limit: int,
        offset: int,
        result: WorkspaceListResult,
    ) -> "WorkspacesResponse":
        workspaces = [
            WorkspaceItem.from_model(workspace) for workspace in result.workspaces
        ]
        return cls(
            total=result.total,
            count=len(workspaces),
            limit=limit,
            offset=offset,
            workspaces=workspaces,
        )


class WorkspaceResponse(BaseModel):
    workspace: WorkspaceItem

    @classmethod
    def from_model(cls, workspace: Workspace) -> "WorkspaceResponse":
        return cls(workspace=WorkspaceItem.from_model(workspace))
