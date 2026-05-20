import pytest
from pydantic import ValidationError

from backend.app.schemas.agent import (
    AgentApprovalDecisionRequest,
    AgentTriageResponse,
    SupportTicketRequest,
)


def test_support_ticket_request_trims_required_strings() -> None:
    request = SupportTicketRequest(
        ticket_id=" TICKET-1 ",
        customer_message=" How do I debug citations? ",
        workspace_id=" public ",
        metadata=None,
    )

    assert request.ticket_id == "TICKET-1"
    assert request.customer_message == "How do I debug citations?"
    assert request.workspace_id == "public"
    assert request.metadata == {}


@pytest.mark.parametrize(
    "payload",
    [
        {"ticket_id": "", "customer_message": "message"},
        {"ticket_id": "TICKET-1", "customer_message": " "},
        {"ticket_id": "TICKET-1", "customer_message": "message", "metadata": []},
        {"ticket_id": "TICKET-1", "customer_message": "message", "priority": "p0"},
    ],
)
def test_support_ticket_request_rejects_invalid_payloads(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        SupportTicketRequest(**payload)


def test_agent_triage_response_defaults_to_empty_collections() -> None:
    response = AgentTriageResponse(run_id="run-1", status="finalized")

    assert response.approval_required is False
    assert response.sources == []
    assert response.tool_calls == []
    assert response.metrics == {}


def test_agent_approval_decision_rejects_blank_feedback() -> None:
    with pytest.raises(ValidationError):
        AgentApprovalDecisionRequest(decision="rejected", human_feedback=" ")

