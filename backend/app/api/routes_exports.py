import uuid
from hashlib import sha256
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.security import ApiPrincipal, require_api_key, resolve_workspace_id
from backend.app.api.workspace_validation import (
    get_workspace_repository,
    require_existing_workspace,
)
from backend.app.core.request_id import get_request_id
from backend.app.db.repositories import (
    CreateExportJobInput,
    ExportJobRepository,
    WorkspaceRepository,
)
from backend.app.db.session import get_db_session
from backend.app.schemas.export_jobs import (
    CreateExportJobRequest,
    ExportJobResponse,
    ExportJobsResponse,
    ExportJobStatus,
    ExportJobType,
)

router = APIRouter(prefix="/exports", tags=["exports"])


async def get_export_job_repository(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ExportJobRepository:
    return ExportJobRepository(session=session)


@router.post(
    "/jobs",
    response_model=ExportJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_export_job(
    request: CreateExportJobRequest,
    http_request: Request,
    principal: Annotated[ApiPrincipal, Depends(require_api_key)],
    export_job_repository: Annotated[
        ExportJobRepository,
        Depends(get_export_job_repository),
    ],
    workspace_repository: Annotated[
        WorkspaceRepository,
        Depends(get_workspace_repository),
    ],
    workspace_id: Annotated[str | None, Header(alias="X-Workspace-ID")] = None,
) -> ExportJobResponse:
    normalized_workspace_id = resolve_workspace_id(principal, workspace_id)
    await require_existing_workspace(
        workspace_id=normalized_workspace_id,
        repository=workspace_repository,
    )
    export_job = await export_job_repository.create_export_job(
        CreateExportJobInput(
            request_id=get_request_id(http_request),
            actor_hash=hash_principal_token(principal),
            workspace_id=normalized_workspace_id,
            export_type=request.export_type,
            format=request.format,
            filters=request.filters.model_dump(mode="json", exclude_none=True),
        ),
        commit=True,
    )
    return ExportJobResponse.from_model(export_job)


@router.get("/jobs", response_model=ExportJobsResponse)
async def list_export_jobs(
    principal: Annotated[ApiPrincipal, Depends(require_api_key)],
    export_job_repository: Annotated[
        ExportJobRepository,
        Depends(get_export_job_repository),
    ],
    workspace_id: Annotated[str | None, Header(alias="X-Workspace-ID")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    job_status: Annotated[ExportJobStatus | None, Query(alias="status")] = None,
    export_type: Annotated[ExportJobType | None, Query()] = None,
) -> ExportJobsResponse:
    normalized_workspace_id = resolve_workspace_id(principal, workspace_id)
    result = await export_job_repository.list_export_jobs(
        workspace_id=normalized_workspace_id,
        limit=limit,
        offset=offset,
        status=job_status,
        export_type=export_type,
    )
    return ExportJobsResponse.from_result(
        limit=limit,
        offset=offset,
        result=result,
    )


@router.get("/jobs/{job_id}", response_model=ExportJobResponse)
async def get_export_job(
    job_id: uuid.UUID,
    principal: Annotated[ApiPrincipal, Depends(require_api_key)],
    export_job_repository: Annotated[
        ExportJobRepository,
        Depends(get_export_job_repository),
    ],
    workspace_id: Annotated[str | None, Header(alias="X-Workspace-ID")] = None,
) -> ExportJobResponse:
    normalized_workspace_id = resolve_workspace_id(principal, workspace_id)
    export_job = await export_job_repository.get_export_job(
        job_id=job_id,
        workspace_id=normalized_workspace_id,
    )
    if export_job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="export job not found",
        )
    return ExportJobResponse.from_model(export_job)


def hash_principal_token(principal: ApiPrincipal) -> str:
    return sha256(principal.token.encode("utf-8")).hexdigest()
