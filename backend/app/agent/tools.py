from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ToolRiskLevel = Literal["low", "medium", "high"]


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    risk_level: ToolRiskLevel
    requires_approval: bool = False


class ToolCallRecord(BaseModel):
    tool_name: str
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = Field(ge=0)
    success: bool
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


RAG_SEARCH_TOOL_SPEC = ToolSpec(
    name="rag_search_tool",
    description=(
        "Search the internal knowledge base through the existing RAG pipeline "
        "and return cited context for support drafting."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "workspace_id": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1},
        },
        "required": ["query", "workspace_id"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "sources": {"type": "array"},
            "context": {"type": "string"},
            "top_score": {"type": ["number", "null"]},
            "refusal_recommended": {"type": "boolean"},
        },
    },
    risk_level="low",
)

CLASSIFY_TICKET_TOOL_SPEC = ToolSpec(
    name="classify_ticket_tool",
    description=(
        "Classify a support ticket into a controlled category before the "
        "workflow chooses retrieval, drafting, and approval steps."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "customer_message": {"type": "string"},
            "priority": {"type": "string"},
        },
        "required": ["customer_message"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "category": {"type": "string"},
            "risk_level": {"type": "string"},
            "matched_terms": {"type": "array"},
        },
    },
    risk_level="low",
)

TICKET_LOOKUP_TOOL_SPEC = ToolSpec(
    name="ticket_lookup_tool",
    description="Find similar historical support tickets in the same workspace.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "category": {"type": ["string", "null"]},
            "workspace_id": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1},
        },
        "required": ["query", "workspace_id"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "cases": {"type": "array"},
        },
    },
    risk_level="low",
)

DRAFT_RESPONSE_TOOL_SPEC = ToolSpec(
    name="draft_response_tool",
    description=(
        "Draft a cited support response from retrieved knowledge and "
        "historical cases without performing external actions."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "customer_message": {"type": "string"},
            "sources": {"type": "array"},
            "historical_cases": {"type": "array"},
        },
        "required": ["customer_message"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "draft": {"type": "string"},
            "cited_source_ids": {"type": "array"},
            "citation_valid": {"type": "boolean"},
        },
    },
    risk_level="medium",
)

RISK_CHECK_TOOL_SPEC = ToolSpec(
    name="risk_check_tool",
    description="Classify support draft risk and decide whether approval is required.",
    input_schema={
        "type": "object",
        "properties": {
            "customer_message": {"type": "string"},
            "draft_answer": {"type": "string"},
            "tool_calls": {"type": "array"},
        },
        "required": ["customer_message"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "risk_level": {"type": "string"},
            "approval_required": {"type": "boolean"},
            "reason": {"type": "string"},
        },
    },
    risk_level="low",
)

HUMAN_APPROVAL_TOOL_SPEC = ToolSpec(
    name="human_approval_tool",
    description=(
        "Create an internal pending approval request for a high-risk support "
        "draft. This tool does not send external messages."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "run_id": {"type": "string"},
            "ticket_id": {"type": "string"},
            "workspace_id": {"type": "string"},
            "draft_answer": {"type": "string"},
            "risk_level": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": [
            "run_id",
            "ticket_id",
            "workspace_id",
            "draft_answer",
            "risk_level",
            "reason",
        ],
    },
    output_schema={
        "type": "object",
        "properties": {
            "approval_id": {"type": "string"},
            "status": {"type": "string"},
        },
    },
    risk_level="high",
    requires_approval=True,
)

TOOL_REGISTRY: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        CLASSIFY_TICKET_TOOL_SPEC,
        RAG_SEARCH_TOOL_SPEC,
        TICKET_LOOKUP_TOOL_SPEC,
        DRAFT_RESPONSE_TOOL_SPEC,
        RISK_CHECK_TOOL_SPEC,
        HUMAN_APPROVAL_TOOL_SPEC,
    )
}


def get_tool_spec(name: str) -> ToolSpec:
    try:
        return TOOL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown agent tool: {name}") from exc
