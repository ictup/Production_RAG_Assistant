import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.app.db.models import ExportJob
from backend.app.db.repositories import ExportJobListResult

ExportJobFormat = Literal["jsonl", "csv"]
ExportJobStatus = Literal["pending", "running", "succeeded", "failed"]
ExportJobType = Literal["chat_logs"]


class ChatLogExportFilters(BaseModel):
    limit: int = Field(default=1000, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    session_id: uuid.UUID | None = None
    request_id: str | None = Field(default=None, min_length=1, max_length=256)
    refusal_only: bool = False
    citation_valid: bool | None = None


class CreateExportJobRequest(BaseModel):
    export_type: ExportJobType = "chat_logs"
    format: ExportJobFormat = "jsonl"
    filters: ChatLogExportFilters = Field(default_factory=ChatLogExportFilters)


class ExportJobItem(BaseModel):
    id: str
    workspace_id: str
    request_id: str
    actor_hash: str
    export_type: str
    format: str
    status: ExportJobStatus
    filters: dict[str, Any]
    result_uri: str | None
    result_media_type: str | None
    result_size_bytes: int | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

    @classmethod
    def from_model(cls, export_job: ExportJob) -> "ExportJobItem":
        return cls(
            id=str(export_job.id),
            workspace_id=export_job.workspace_id,
            request_id=export_job.request_id,
            actor_hash=export_job.actor_hash,
            export_type=export_job.export_type,
            format=export_job.format,
            status=export_job.status,  # type: ignore[arg-type]
            filters=dict(export_job.filters_),
            result_uri=export_job.result_uri,
            result_media_type=export_job.result_media_type,
            result_size_bytes=export_job.result_size_bytes,
            error_message=export_job.error_message,
            created_at=export_job.created_at,
            updated_at=export_job.updated_at,
            started_at=export_job.started_at,
            completed_at=export_job.completed_at,
        )


class ExportJobResponse(BaseModel):
    job: ExportJobItem

    @classmethod
    def from_model(cls, export_job: ExportJob) -> "ExportJobResponse":
        return cls(job=ExportJobItem.from_model(export_job))


class ExportJobsResponse(BaseModel):
    total: int = Field(ge=0)
    count: int = Field(ge=0)
    limit: int = Field(gt=0)
    offset: int = Field(ge=0)
    jobs: list[ExportJobItem]

    @classmethod
    def from_result(
        cls,
        *,
        limit: int,
        offset: int,
        result: ExportJobListResult,
    ) -> "ExportJobsResponse":
        jobs = [ExportJobItem.from_model(job) for job in result.jobs]
        return cls(
            total=result.total,
            count=len(jobs),
            limit=limit,
            offset=offset,
            jobs=jobs,
        )
