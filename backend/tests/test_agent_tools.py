import pytest

from backend.app.agent.tools import TOOL_REGISTRY, ToolCallRecord, get_tool_spec


def test_tool_registry_contains_only_explicit_agent_tools() -> None:
    assert set(TOOL_REGISTRY) == {
        "rag_search_tool",
        "ticket_lookup_tool",
        "draft_response_tool",
        "risk_check_tool",
        "human_approval_tool",
    }


def test_human_approval_tool_is_marked_high_risk() -> None:
    spec = get_tool_spec("human_approval_tool")

    assert spec.risk_level == "high"
    assert spec.requires_approval is True
    assert "approval" in spec.description.lower()


def test_get_tool_spec_rejects_unknown_tool() -> None:
    with pytest.raises(ValueError, match="unknown agent tool"):
        get_tool_spec("shell_tool")


def test_tool_call_record_stores_sanitized_summaries() -> None:
    record = ToolCallRecord(
        tool_name="risk_check_tool",
        input_summary={"message_length": 42},
        output_summary={"risk_level": "low"},
        latency_ms=12,
        success=True,
    )

    assert record.tool_name == "risk_check_tool"
    assert record.input_summary == {"message_length": 42}
    assert record.output_summary == {"risk_level": "low"}
    assert record.error is None
    assert record.created_at.tzinfo is not None

