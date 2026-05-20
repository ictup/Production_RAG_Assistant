from backend.app.agent.state import build_initial_agent_state
from backend.app.schemas.agent import SupportTicketRequest


def test_build_initial_agent_state_sets_safe_defaults() -> None:
    state = build_initial_agent_state(
        run_id=" run-1 ",
        ticket_id=" TICKET-1 ",
        customer_message=" The answer has wrong citations. ",
        metadata={"source": "test"},
    )

    assert state["run_id"] == "run-1"
    assert state["ticket_id"] == "TICKET-1"
    assert state["customer_message"] == "The answer has wrong citations."
    assert state["workspace_id"] == "public"
    assert state["category"] is None
    assert state["risk_level"] is None
    assert state["approval_required"] is False
    assert state["approval_id"] is None
    assert state["cited_source_ids"] == []
    assert state["cited_case_ids"] == []
    assert state["tool_calls"] == []
    assert state["node_runs"] == []
    assert state["errors"] == []
    assert state["metrics"] == {}
    assert state["metadata"] == {"source": "test"}


def test_support_ticket_request_builds_initial_state() -> None:
    request = SupportTicketRequest(
        ticket_id="TICKET-2",
        customer_message="Our gateway returns 429 errors.",
        priority="high",
        workspace_id="tenant-a",
        customer_tier="enterprise",
        metadata={"region": "eu"},
    )

    state = request.to_initial_state(run_id="run-2")

    assert state["run_id"] == "run-2"
    assert state["ticket_id"] == "TICKET-2"
    assert state["priority"] == "high"
    assert state["workspace_id"] == "tenant-a"
    assert state["customer_tier"] == "enterprise"
    assert state["metadata"] == {"region": "eu"}
