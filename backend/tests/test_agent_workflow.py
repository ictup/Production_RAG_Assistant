from backend.app.agent.workflow import (
    build_agent_run_id,
    run_support_triage_skeleton,
)
from backend.app.schemas.agent import SupportTicketRequest


def test_build_agent_run_id_uses_request_id() -> None:
    assert build_agent_run_id(" request-1 ") == "agent_request-1"


def test_support_triage_skeleton_finalizes_safe_ticket() -> None:
    response = run_support_triage_skeleton(
        SupportTicketRequest(
            ticket_id="TICKET-1",
            customer_message="How can I debug citation validation failures?",
        ),
        request_id="request-1",
        trace_id="trace-1",
    )

    assert response.run_id == "agent_request-1"
    assert response.status == "finalized"
    assert response.category == "rag_failure"
    assert response.risk_level == "low"
    assert response.approval_required is False
    assert response.final_answer is not None
    assert response.draft_answer is None
    assert response.trace_id == "trace-1"
    assert response.metrics["tool_count"] == 2
    assert [tool_call["tool_name"] for tool_call in response.tool_calls] == [
        "classify_ticket_tool",
        "risk_check_tool",
    ]


def test_support_triage_skeleton_routes_high_risk_ticket_to_approval() -> None:
    response = run_support_triage_skeleton(
        SupportTicketRequest(
            ticket_id="TICKET-2",
            customer_message="Delete all logs containing customer prompts.",
        ),
        request_id="request-2",
    )

    assert response.status == "approval_required"
    assert response.category == "data_privacy"
    assert response.risk_level == "high"
    assert response.approval_required is True
    assert response.approval_id is None
    assert response.final_answer is None
    assert response.draft_answer is not None
    assert response.reason is not None
    assert "customer prompt" in response.reason

