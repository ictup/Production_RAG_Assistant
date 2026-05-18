from datetime import UTC, datetime

import pytest

from backend.app.db.inspect_chat_logs import (
    ChatLogSnapshot,
    format_chat_log_snapshot,
    truncate_text,
    validate_chat_log_snapshot,
)
from backend.app.db.models import ChatLog


def make_chat_log() -> ChatLog:
    return ChatLog(
        request_id="request-1",
        workspace_id="public",
        question="What problem does FlashAttention solve?",
        answer="FlashAttention reduces memory traffic. [1]",
        sources=[{"source_id": "1"}],
        retrieval={"mode": "hybrid_rrf_rerank"},
        usage={"model": "fake-llm"},
        refusal=None,
        citation_valid=True,
        latency_ms=12,
        created_at=datetime(2026, 5, 18, 8, 0, tzinfo=UTC),
    )


def test_truncate_text_normalizes_whitespace_and_truncates() -> None:
    assert truncate_text("one\n two   three", max_length=20) == "one two three"
    assert truncate_text("abcdefghijklmnopqrstuvwxyz", max_length=10) == "abcdefg..."


def test_format_chat_log_snapshot_includes_recent_log_metadata() -> None:
    output = format_chat_log_snapshot(
        ChatLogSnapshot(
            workspace_id="public",
            logs_count=1,
            recent_logs=[make_chat_log()],
        )
    )

    assert "workspace: public" in output
    assert "chat logs: 1" in output
    assert "- request_id: request-1" in output
    assert "citation_valid: True" in output
    assert "refusal_reason: none" in output
    assert "question: What problem does FlashAttention solve?" in output


def test_format_chat_log_snapshot_handles_empty_logs() -> None:
    output = format_chat_log_snapshot(
        ChatLogSnapshot(workspace_id="public", logs_count=0, recent_logs=[])
    )

    assert "- none" in output


def test_validate_chat_log_snapshot_passes_when_count_meets_threshold() -> None:
    validate_chat_log_snapshot(
        ChatLogSnapshot(workspace_id="public", logs_count=2, recent_logs=[]),
        min_logs=2,
    )


def test_validate_chat_log_snapshot_fails_when_count_is_too_low() -> None:
    with pytest.raises(SystemExit, match="chat log check failed"):
        validate_chat_log_snapshot(
            ChatLogSnapshot(workspace_id="public", logs_count=0, recent_logs=[]),
            min_logs=1,
        )
