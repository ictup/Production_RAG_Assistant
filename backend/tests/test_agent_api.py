from fastapi.testclient import TestClient

from backend.app.core.config import Settings, get_settings
from backend.app.main import create_app

AUTH_HEADERS = {"Authorization": "Bearer dev-key"}


def build_client(settings: Settings | None = None) -> TestClient:
    settings = settings or Settings(api_keys="dev-key")
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_support_triage_route_returns_finalized_skeleton_response() -> None:
    client = build_client()

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
    assert body["metrics"]["tool_count"] == 2
    assert [tool_call["tool_name"] for tool_call in body["tool_calls"]] == [
        "classify_ticket_tool",
        "risk_check_tool",
    ]


def test_support_triage_route_returns_approval_required_for_high_risk_ticket() -> None:
    client = build_client()

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

