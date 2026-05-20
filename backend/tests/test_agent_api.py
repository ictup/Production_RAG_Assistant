from datetime import UTC, datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.app.api import routes_agent
from backend.app.core.config import Settings, get_settings
from backend.app.main import create_app
from backend.app.rag.citations import Source
from backend.app.rag.pipeline import RagRetrievalContext, RetrievalInfo

AUTH_HEADERS = {"Authorization": "Bearer dev-key"}


class FakeRagPipeline:
    def __init__(self) -> None:
        self.requests = []

    async def retrieve_context(self, request):
        self.requests.append(request)
        return RagRetrievalContext(
            sources=[
                Source(
                    source_id="1",
                    title="Citation Debugging",
                    section="Validation",
                    source_uri="docs/citations.md",
                    chunk_id="chunk-1",
                    score=0.92,
                )
            ],
            context="[1] Citation Debugging\nInspect retrieved chunks.",
            retrieval=RetrievalInfo(
                mode="hybrid_rrf_rerank",
                vector_top_k=5,
                sparse_top_k=5,
                fused_count=1,
                used_count=1,
                top_score=0.92,
            ),
        )


class FakeWorkspaceRepository:
    def __init__(
        self,
        workspace_ids: set[str] | None = None,
        archived_workspace_ids: set[str] | None = None,
    ) -> None:
        self.workspace_ids = workspace_ids or {"public", "tenant-a"}
        self.archived_workspace_ids = archived_workspace_ids or set()

    async def get_workspace(self, *, workspace_id: str):
        if workspace_id not in self.workspace_ids:
            return None
        archived_at = (
            datetime(2026, 5, 20, 8, 0, tzinfo=UTC)
            if workspace_id in self.archived_workspace_ids
            else None
        )
        return SimpleNamespace(id=workspace_id, archived_at=archived_at)


def build_client(
    settings: Settings | None = None,
    fake_pipeline: FakeRagPipeline | None = None,
    fake_workspace_repository: FakeWorkspaceRepository | None = None,
) -> TestClient:
    settings = settings or Settings(api_keys="dev-key")
    fake_pipeline = fake_pipeline or FakeRagPipeline()
    fake_workspace_repository = (
        fake_workspace_repository or FakeWorkspaceRepository()
    )
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[routes_agent.get_agent_rag_pipeline] = (
        lambda: fake_pipeline
    )
    app.dependency_overrides[routes_agent.get_workspace_repository] = (
        lambda: fake_workspace_repository
    )
    return TestClient(app)


def test_support_triage_route_returns_finalized_skeleton_response() -> None:
    fake_pipeline = FakeRagPipeline()
    client = build_client(fake_pipeline=fake_pipeline)

    response = client.post(
        "/agent/support-triage",
        headers={
            **AUTH_HEADERS,
            "X-Request-ID": "request-1",
            "X-Trace-ID": "trace-1",
        },
        json={
            "ticket_id": "TICKET-1",
            "customer_message": "How can I debug citation validation failures?",
            "workspace_id": " public ",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "agent_request-1"
    assert body["status"] == "finalized"
    assert body["category"] == "rag_failure"
    assert body["risk_level"] == "low"
    assert body["approval_required"] is False
    assert body["trace_id"] == "trace-1"
    assert body["sources"][0]["chunk_id"] == "chunk-1"
    assert body["retrieval_context"] == (
        "[1] Citation Debugging\nInspect retrieved chunks."
    )
    assert body["retrieval"]["top_score"] == 0.92
    assert body["metrics"]["tool_count"] == 3
    assert body["metrics"]["retrieved_source_count"] == 1
    assert [tool_call["tool_name"] for tool_call in body["tool_calls"]] == [
        "classify_ticket_tool",
        "risk_check_tool",
        "rag_search_tool",
    ]
    assert len(fake_pipeline.requests) == 1
    assert fake_pipeline.requests[0].workspace_id == "public"


def test_support_triage_route_returns_approval_required_for_high_risk_ticket() -> None:
    fake_pipeline = FakeRagPipeline()
    client = build_client(fake_pipeline=fake_pipeline)

    response = client.post(
        "/agent/support-triage",
        headers=AUTH_HEADERS,
        json={
            "ticket_id": "TICKET-2",
            "customer_message": (
                "Delete all logs that contain customer prompts from production."
            ),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "approval_required"
    assert body["category"] == "data_privacy"
    assert body["risk_level"] == "high"
    assert body["approval_required"] is True
    assert body["approval_id"] is None
    assert body["draft_answer"] is not None
    assert body["final_answer"] is None
    assert body["sources"] == []
    assert body["retrieval"] == {}
    assert fake_pipeline.requests == []


def test_support_triage_route_requires_api_key() -> None:
    client = build_client()

    response = client.post(
        "/agent/support-triage",
        json={
            "ticket_id": "TICKET-1",
            "customer_message": "How can I debug citations?",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "missing api key"


def test_support_triage_route_enforces_workspace_access() -> None:
    client = build_client(
        Settings(
            api_keys="tenant-key",
            api_key_workspace_access="tenant-key=tenant-a",
        )
    )

    response = client.post(
        "/agent/support-triage",
        headers={"Authorization": "Bearer tenant-key"},
        json={
            "ticket_id": "TICKET-1",
            "customer_message": "How can I debug citations?",
            "workspace_id": "tenant-b",
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "workspace access denied"


def test_openapi_exposes_support_triage_route() -> None:
    client = build_client()

    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/agent/support-triage" in response.json()["paths"]
