import time

from backend.app.agent.policies import check_support_risk, classify_ticket
from backend.app.agent.tools import ToolCallRecord
from backend.app.core.tracing import trace_span
from backend.app.schemas.agent import AgentTriageResponse, SupportTicketRequest


def build_agent_run_id(request_id: str) -> str:
    return f"agent_{request_id.strip()}"


def run_support_triage_skeleton(
    request: SupportTicketRequest,
    *,
    request_id: str,
    trace_id: str | None = None,
) -> AgentTriageResponse:
    started_at = time.perf_counter()
    run_id = build_agent_run_id(request_id)
    state = request.to_initial_state(run_id=run_id)
    tool_calls: list[ToolCallRecord] = []

    classification_started_at = time.perf_counter()
    with trace_span(
        "agent.classify_ticket",
        {
            "workspace_id": request.workspace_id,
            "priority": request.priority,
            "message_length": len(request.customer_message),
        },
    ):
        classification = classify_ticket(request.customer_message)
    tool_calls.append(
        ToolCallRecord(
            tool_name="classify_ticket_tool",
            input_summary={
                "message_length": len(request.customer_message),
                "priority": request.priority,
            },
            output_summary={
                "category": classification.category,
                "risk_level": classification.risk_level,
                "matched_terms": classification.matched_terms,
            },
            latency_ms=elapsed_ms(classification_started_at),
            success=True,
        )
    )

    risk_started_at = time.perf_counter()
    with trace_span(
        "agent.risk_check",
        {
            "workspace_id": request.workspace_id,
            "category": classification.category,
            "priority": request.priority,
        },
    ):
        risk = check_support_risk(
            customer_message=request.customer_message,
            priority=request.priority,
        )
    tool_calls.append(
        ToolCallRecord(
            tool_name="risk_check_tool",
            input_summary={
                "message_length": len(request.customer_message),
                "priority": request.priority,
                "category": classification.category,
            },
            output_summary={
                "risk_level": risk.risk_level,
                "approval_required": risk.approval_required,
            },
            latency_ms=elapsed_ms(risk_started_at),
            success=True,
        )
    )

    tool_call_payloads = [
        tool_call.model_dump(mode="json")
        for tool_call in tool_calls
    ]
    state["category"] = classification.category
    state["risk_level"] = risk.risk_level
    state["approval_required"] = risk.approval_required
    state["approval_status"] = "pending" if risk.approval_required else "not_required"
    state["final_action"] = (
        "approval_required" if risk.approval_required else "finalized"
    )
    state["tool_calls"] = tool_call_payloads
    state["metrics"] = {
        "latency_ms": elapsed_ms(started_at),
        "tool_count": len(tool_calls),
        "citation_valid": None,
    }

    if risk.approval_required:
        draft_answer = (
            "This support request requires human review before a response is "
            "finalized."
        )
        state["draft_answer"] = draft_answer
        return AgentTriageResponse(
            run_id=run_id,
            status="approval_required",
            category=classification.category,
            risk_level=risk.risk_level,
            approval_required=True,
            draft_answer=draft_answer,
            reason=risk.reason,
            tool_calls=tool_call_payloads,
            metrics=state["metrics"],
            trace_id=trace_id,
        )

    final_answer = (
        f"Ticket classified as {classification.category}. The next workflow "
        "steps will retrieve grounded knowledge, search historical cases, and "
        "draft a cited support response."
    )
    state["final_answer"] = final_answer
    return AgentTriageResponse(
        run_id=run_id,
        status="finalized",
        category=classification.category,
        risk_level=risk.risk_level,
        approval_required=False,
        final_answer=final_answer,
        reason=risk.reason,
        tool_calls=tool_call_payloads,
        metrics=state["metrics"],
        trace_id=trace_id,
    )


def elapsed_ms(started_at: float) -> int:
    return max(0, int((time.perf_counter() - started_at) * 1000))

