from typing import Annotated

from fastapi import APIRouter, Depends, Request

from backend.app.agent.workflow import run_support_triage_skeleton
from backend.app.api.security import ApiPrincipal, require_api_key, resolve_workspace_id
from backend.app.core.request_id import get_request_id
from backend.app.core.tracing import get_trace_id
from backend.app.schemas.agent import AgentTriageResponse, SupportTicketRequest

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/support-triage", response_model=AgentTriageResponse)
async def support_triage(
    ticket_request: SupportTicketRequest,
    raw_request: Request,
    principal: Annotated[ApiPrincipal, Depends(require_api_key)],
) -> AgentTriageResponse:
    workspace_id = resolve_workspace_id(principal, ticket_request.workspace_id)
    normalized_request = ticket_request.model_copy(
        update={"workspace_id": workspace_id},
    )
    return run_support_triage_skeleton(
        normalized_request,
        request_id=get_request_id(raw_request),
        trace_id=get_trace_id(),
    )

