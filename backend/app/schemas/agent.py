from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from backend.app.agent.state import (
    AgentState,
    RiskLevel,
    TicketCategory,
    build_initial_agent_state,
)

TicketPriority = Literal["low", "normal", "high", "urgent"]
CustomerTier = Literal["free", "pro", "enterprise"]
AgentRunStatus = Literal["finalized", "approval_required", "failed"]
ApprovalDecision = Literal["approved", "rejected"]


class SupportTicketRequest(BaseModel):
    ticket_id: str
    customer_message: str
    priority: TicketPriority = "normal"
    workspace_id: str = "public"
    customer_tier: CustomerTier | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ticket_id", "customer_message", "workspace_id")
    @classmethod
    def value_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def metadata_must_be_object(cls, value: object) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("metadata must be an object")
        return dict(value)

    def to_initial_state(self, *, run_id: str) -> AgentState:
        return build_initial_agent_state(
            run_id=run_id,
            ticket_id=self.ticket_id,
            customer_message=self.customer_message,
            priority=self.priority,
            workspace_id=self.workspace_id,
            customer_tier=self.customer_tier,
            metadata=self.metadata,
        )


class AgentTriageResponse(BaseModel):
    run_id: str
    status: AgentRunStatus
    category: TicketCategory | None = None
    risk_level: RiskLevel | None = None
    approval_required: bool = False
    approval_id: str | None = None
    final_answer: str | None = None
    draft_answer: str | None = None
    reason: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_context: str | None = None
    retrieval: dict[str, Any] = Field(default_factory=dict)
    historical_cases: list[dict[str, Any]] = Field(default_factory=list)
    cited_source_ids: list[str] = Field(default_factory=list)
    cited_case_ids: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    node_runs: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None


class AgentApprovalDecisionRequest(BaseModel):
    decision: ApprovalDecision
    human_feedback: str | None = None

    @field_validator("human_feedback")
    @classmethod
    def human_feedback_must_not_be_blank(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("human_feedback must not be blank")
        return value
