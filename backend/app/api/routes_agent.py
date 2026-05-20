from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agent.workflow import run_support_triage_skeleton
from backend.app.api.security import ApiPrincipal, require_api_key, resolve_workspace_id
from backend.app.api.workspace_validation import (
    get_workspace_repository,
    require_active_workspace,
)
from backend.app.core.request_id import get_request_id
from backend.app.core.tracing import get_trace_id
from backend.app.db.repositories import WorkspaceRepository
from backend.app.db.session import get_db_session
from backend.app.rag.pipeline import RagPipeline
from backend.app.schemas.agent import AgentTriageResponse, SupportTicketRequest

router = APIRouter(prefix="/agent", tags=["agent"])


async def get_agent_rag_pipeline(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> RagPipeline:
    return RagPipeline(session=session)


@router.post("/support-triage", response_model=AgentTriageResponse)
async def support_triage(
    ticket_request: SupportTicketRequest,
    raw_request: Request,
    principal: Annotated[ApiPrincipal, Depends(require_api_key)],
    rag_pipeline: Annotated[RagPipeline, Depends(get_agent_rag_pipeline)],
    workspace_repository: Annotated[
        WorkspaceRepository,
        Depends(get_workspace_repository),
    ],
) -> AgentTriageResponse:
    workspace_id = resolve_workspace_id(principal, ticket_request.workspace_id)
    await require_active_workspace(
        workspace_id=workspace_id,
        repository=workspace_repository,
    )
    normalized_request = ticket_request.model_copy(
        update={"workspace_id": workspace_id},
    )
    return await run_support_triage_skeleton(
        normalized_request,
        rag_pipeline=rag_pipeline,
        request_id=get_request_id(raw_request),
        trace_id=get_trace_id(),
    )
