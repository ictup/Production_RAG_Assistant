from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.security import ApiPrincipal, require_api_key, resolve_workspace_id
from backend.app.db.repositories import CreateWorkspaceInput, WorkspaceRepository
from backend.app.db.session import get_db_session
from backend.app.schemas.workspaces import (
    CreateWorkspaceRequest,
    CreateWorkspaceResponse,
    WorkspaceResponse,
    WorkspacesResponse,
)

router = APIRouter(tags=["workspaces"])


async def get_workspace_repository(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> WorkspaceRepository:
    return WorkspaceRepository(session=session)


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
) -> WorkspacesResponse:
    result = await repository.list_workspaces(
        workspace_ids=principal.allowed_workspaces,
        limit=limit,
        offset=offset,
    )
    return WorkspacesResponse.from_result(
        limit=limit,
        offset=offset,
        result=result,
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
