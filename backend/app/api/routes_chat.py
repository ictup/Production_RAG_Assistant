from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.security import require_api_key
from backend.app.core.request_id import get_request_id
from backend.app.db.session import get_db_session
from backend.app.rag.pipeline import RagPipeline
from backend.app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


async def get_rag_pipeline(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> RagPipeline:
    return RagPipeline(session=session)


def normalize_workspace_id(workspace_id: str | None) -> str:
    if workspace_id is None:
        return "public"

    normalized = workspace_id.strip()
    return normalized or "public"


@router.post("/chat", response_model=ChatResponse)
async def chat(
    http_request: Request,
    request: ChatRequest,
    _api_key: Annotated[str, Depends(require_api_key)],
    pipeline: Annotated[RagPipeline, Depends(get_rag_pipeline)],
    workspace_id: Annotated[str | None, Header(alias="X-Workspace-ID")] = None,
) -> ChatResponse:
    response = await pipeline.answer_question(
        request.to_pipeline_request(
            workspace_id=normalize_workspace_id(workspace_id),
        )
    )
    return ChatResponse.from_pipeline_response(
        response,
        request_id=get_request_id(http_request),
    )
