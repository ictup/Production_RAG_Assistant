from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from backend.app.api.security import ApiPrincipal, require_api_key, resolve_workspace_id
from backend.app.api.workspace_validation import get_workspace_repository
from backend.app.db.repositories import (
    ArchiveWorkspaceInput,
    CreateWorkspaceInput,
    UpdateWorkspaceInput,
    WorkspaceRepository,
)
from backend.app.schemas.workspaces import (
    ArchiveWorkspaceRequest,
    BulkArchiveWorkspacesRequest,
    BulkRestoreWorkspacesRequest,
    BulkWorkspaceOperationResponse,
    CreateWorkspaceRequest,
    CreateWorkspaceResponse,
    UpdateWorkspaceRequest,
    WorkspaceResponse,
    WorkspacesResponse,
)

router = APIRouter(tags=["workspaces"])

WorkspaceStatusFilter = Literal["all", "active", "archived"]


@router.post(
    "/workspaces",
    response_model=CreateWorkspaceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_workspace(
    request: CreateWorkspaceRequest,
    response: Response,
    principal: Annotated[ApiPrincipal, Depends(require_api_key)],
    repository: Annotated[WorkspaceRepository, Depends(get_workspace_repository)],
) -> CreateWorkspaceResponse:
    workspace_id = resolve_workspace_id(principal, request.id)
    result = await repository.create_workspace(
        CreateWorkspaceInput(
            id=workspace_id,
            name=request.name,
            description=request.description,
            metadata=request.metadata,
        ),
        commit=True,
    )
    if not result.created:
        response.status_code = status.HTTP_200_OK
    return CreateWorkspaceResponse.from_result(result)


@router.get("/workspaces", response_model=WorkspacesResponse)
async def list_workspaces(
    principal: Annotated[ApiPrincipal, Depends(require_api_key)],
    repository: Annotated[WorkspaceRepository, Depends(get_workspace_repository)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    q: Annotated[str | None, Query(max_length=256)] = None,
    workspace_status: Annotated[
        WorkspaceStatusFilter,
        Query(alias="status"),
    ] = "all",
) -> WorkspacesResponse:
    result = await repository.list_workspaces(
        workspace_ids=principal.allowed_workspaces,
        limit=limit,
        offset=offset,
        search=q,
        archived=workspace_status_to_archived_filter(workspace_status),
    )
    return WorkspacesResponse.from_result(
        limit=limit,
        offset=offset,
        result=result,
    )


def workspace_status_to_archived_filter(
    workspace_status: WorkspaceStatusFilter,
) -> bool | None:
    if workspace_status == "active":
        return False
    if workspace_status == "archived":
        return True
    return None


@router.post(
    "/workspaces/bulk/archive",
    response_model=BulkWorkspaceOperationResponse,
)
async def archive_workspaces(
    request: BulkArchiveWorkspacesRequest,
    principal: Annotated[ApiPrincipal, Depends(require_api_key)],
    repository: Annotated[WorkspaceRepository, Depends(get_workspace_repository)],
) -> BulkWorkspaceOperationResponse:
    workspace_ids = resolve_bulk_workspace_ids(principal, request.ids)
    result = await repository.archive_workspaces(
        [
            ArchiveWorkspaceInput(id=workspace_id, reason=request.reason)
            for workspace_id in workspace_ids
        ],
        commit=True,
    )
    raise_for_missing_bulk_workspaces(result.missing_ids)
    return BulkWorkspaceOperationResponse.from_result(
        action="archive",
        requested_count=len(workspace_ids),
        result=result,
    )


@router.post(
    "/workspaces/bulk/restore",
    response_model=BulkWorkspaceOperationResponse,
)
async def restore_workspaces(
    request: BulkRestoreWorkspacesRequest,
    principal: Annotated[ApiPrincipal, Depends(require_api_key)],
    repository: Annotated[WorkspaceRepository, Depends(get_workspace_repository)],
) -> BulkWorkspaceOperationResponse:
    workspace_ids = resolve_bulk_workspace_ids(principal, request.ids)
    result = await repository.restore_workspaces(
        workspace_ids=workspace_ids,
        commit=True,
    )
    raise_for_missing_bulk_workspaces(result.missing_ids)
    return BulkWorkspaceOperationResponse.from_result(
        action="restore",
        requested_count=len(workspace_ids),
        result=result,
    )


def resolve_bulk_workspace_ids(
    principal: ApiPrincipal,
    workspace_ids: list[str],
) -> list[str]:
    resolved_ids: list[str] = []
    seen_ids: set[str] = set()
    for workspace_id in workspace_ids:
        resolved_id = resolve_workspace_id(principal, workspace_id)
        if resolved_id not in seen_ids:
            resolved_ids.append(resolved_id)
            seen_ids.add(resolved_id)
    return resolved_ids


def raise_for_missing_bulk_workspaces(missing_ids: list[str]) -> None:
    if not missing_ids:
        return
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "message": "workspace not found",
            "workspace_ids": missing_ids,
        },
    )


@router.get("/workspaces/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(
    workspace_id: str,
    principal: Annotated[ApiPrincipal, Depends(require_api_key)],
    repository: Annotated[WorkspaceRepository, Depends(get_workspace_repository)],
) -> WorkspaceResponse:
    normalized_workspace_id = resolve_workspace_id(principal, workspace_id)
    workspace = await repository.get_workspace(workspace_id=normalized_workspace_id)
    if workspace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="workspace not found",
        )
    return WorkspaceResponse.from_model(workspace)


@router.patch("/workspaces/{workspace_id}", response_model=WorkspaceResponse)
async def update_workspace(
    workspace_id: str,
    request: UpdateWorkspaceRequest,
    principal: Annotated[ApiPrincipal, Depends(require_api_key)],
    repository: Annotated[WorkspaceRepository, Depends(get_workspace_repository)],
) -> WorkspaceResponse:
    normalized_workspace_id = resolve_workspace_id(principal, workspace_id)
    updated_workspace = await repository.update_workspace(
        UpdateWorkspaceInput(
            id=normalized_workspace_id,
            name=request.name,
            description=request.description,
            metadata=request.metadata,
            update_name="name" in request.model_fields_set,
            update_description="description" in request.model_fields_set,
            update_metadata="metadata" in request.model_fields_set,
        ),
        commit=True,
    )
    if updated_workspace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="workspace not found",
        )
    return WorkspaceResponse.from_model(updated_workspace)


@router.post("/workspaces/{workspace_id}/archive", response_model=WorkspaceResponse)
async def archive_workspace(
    workspace_id: str,
    principal: Annotated[ApiPrincipal, Depends(require_api_key)],
    repository: Annotated[WorkspaceRepository, Depends(get_workspace_repository)],
    request: ArchiveWorkspaceRequest | None = None,
) -> WorkspaceResponse:
    normalized_workspace_id = resolve_workspace_id(principal, workspace_id)
    archived_workspace = await repository.archive_workspace(
        ArchiveWorkspaceInput(
            id=normalized_workspace_id,
            reason=request.reason if request is not None else None,
        ),
        commit=True,
    )
    if archived_workspace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="workspace not found",
        )
    return WorkspaceResponse.from_model(archived_workspace)


@router.post("/workspaces/{workspace_id}/restore", response_model=WorkspaceResponse)
async def restore_workspace(
    workspace_id: str,
    principal: Annotated[ApiPrincipal, Depends(require_api_key)],
    repository: Annotated[WorkspaceRepository, Depends(get_workspace_repository)],
) -> WorkspaceResponse:
    normalized_workspace_id = resolve_workspace_id(principal, workspace_id)
    restored_workspace = await repository.restore_workspace(
        workspace_id=normalized_workspace_id,
        commit=True,
    )
    if restored_workspace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="workspace not found",
        )
    return WorkspaceResponse.from_model(restored_workspace)
