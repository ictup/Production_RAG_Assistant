from typing import Any, Literal, NotRequired, TypedDict

TicketCategory = Literal[
    "rag_failure",
    "serving_latency",
    "rate_limit",
    "deployment",
    "evaluation",
    "security",
    "data_privacy",
    "unknown",
]
RiskLevel = Literal["low", "medium", "high"]
ApprovalStatus = Literal["not_required", "pending", "approved", "rejected"]
FinalAction = Literal["finalized", "approval_required", "failed"]


class AgentState(TypedDict):
    run_id: str
    ticket_id: str
    customer_message: str
    priority: str
    workspace_id: str
    customer_tier: NotRequired[str | None]
    metadata: dict[str, Any]
    category: TicketCategory | None
    risk_level: RiskLevel | None
    retrieved_sources: list[dict[str, Any]]
    historical_cases: list[dict[str, Any]]
    draft_answer: str | None
    cited_source_ids: list[str]
    cited_case_ids: list[str]
    approval_required: bool
    approval_status: ApprovalStatus | None
    human_feedback: str | None
    final_answer: str | None
    final_action: FinalAction | None
    tool_calls: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    metrics: dict[str, Any]


def build_initial_agent_state(
    *,
    run_id: str,
    ticket_id: str,
    customer_message: str,
    priority: str = "normal",
    workspace_id: str = "public",
    customer_tier: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentState:
    return AgentState(
        run_id=run_id.strip(),
        ticket_id=ticket_id.strip(),
        customer_message=customer_message.strip(),
        priority=priority.strip(),
        workspace_id=workspace_id.strip(),
        customer_tier=customer_tier,
        metadata=dict(metadata or {}),
        category=None,
        risk_level=None,
        retrieved_sources=[],
        historical_cases=[],
        draft_answer=None,
        cited_source_ids=[],
        cited_case_ids=[],
        approval_required=False,
        approval_status=None,
        human_feedback=None,
        final_answer=None,
        final_action=None,
        tool_calls=[],
        errors=[],
        metrics={},
    )
