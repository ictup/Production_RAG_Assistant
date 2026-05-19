import json
import logging

import pytest
from fastapi.testclient import TestClient

from backend.app.core.tracing import (
    TRACE_ID_HEADER,
    TRACE_LOGGER_NAME,
    build_span_log_payload,
    get_trace_id,
    trace_context,
    trace_span,
)
from backend.app.main import create_app


def parse_trace_logs(caplog) -> list[dict]:
    return [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == TRACE_LOGGER_NAME
    ]


def test_build_span_log_payload_uses_safe_shape() -> None:
    payload = build_span_log_payload(
        trace_id="trace-1",
        span_id="span-1",
        parent_span_id=None,
        name="rag.embedding",
        status="ok",
        latency_ms=12,
        attributes={"provider": "fake", "top_k": 3},
    )

    assert payload == {
        "event": "trace_span",
        "trace_id": "trace-1",
        "span_id": "span-1",
        "parent_span_id": None,
        "name": "rag.embedding",
        "status": "ok",
        "latency_ms": 12,
        "attributes": {"provider": "fake", "top_k": 3},
    }


def test_trace_context_sets_and_resets_trace_id() -> None:
    assert get_trace_id() is None

    with trace_context("trace-1"):
        assert get_trace_id() == "trace-1"

    assert get_trace_id() is None


def test_trace_span_logs_nested_parent_relationship(caplog) -> None:
    with caplog.at_level(logging.INFO, logger=TRACE_LOGGER_NAME):
        with trace_context("trace-1"):
            with trace_span("outer"):
                with trace_span("inner"):
                    pass

    logs = parse_trace_logs(caplog)
    assert [log["name"] for log in logs] == ["inner", "outer"]
    inner_log, outer_log = logs
    assert inner_log["trace_id"] == "trace-1"
    assert outer_log["trace_id"] == "trace-1"
    assert inner_log["parent_span_id"] == outer_log["span_id"]
    assert outer_log["parent_span_id"] is None


def test_trace_span_logs_errors(caplog) -> None:
    with pytest.raises(ValueError, match="boom"):
        with caplog.at_level(logging.INFO, logger=TRACE_LOGGER_NAME):
            with trace_context("trace-1"):
                with trace_span("failing"):
                    raise ValueError("boom")

    logs = parse_trace_logs(caplog)
    assert len(logs) == 1
    assert logs[0]["name"] == "failing"
    assert logs[0]["status"] == "error"
    assert logs[0]["error_type"] == "ValueError"


def test_trace_context_middleware_adds_trace_header_and_logs_http_span(caplog) -> None:
    client = TestClient(create_app())

    with caplog.at_level(logging.INFO, logger=TRACE_LOGGER_NAME):
        response = client.get("/health", headers={TRACE_ID_HEADER: "client-trace"})

    assert response.status_code == 200
    assert response.headers[TRACE_ID_HEADER] == "client-trace"
    logs = parse_trace_logs(caplog)
    assert len(logs) == 1
    assert logs[0]["event"] == "trace_span"
    assert logs[0]["trace_id"] == "client-trace"
    assert logs[0]["name"] == "http.request"
    assert logs[0]["status"] == "ok"
    assert logs[0]["attributes"] == {
        "method": "GET",
        "path": "/health",
        "status_code": 200,
    }


def test_trace_logs_do_not_include_authorization_header(caplog) -> None:
    client = TestClient(create_app())

    with caplog.at_level(logging.INFO, logger=TRACE_LOGGER_NAME):
        response = client.post(
            "/chat",
            headers={
                "Authorization": "Bearer secret-key",
                TRACE_ID_HEADER: "client-trace",
            },
            json={"question": "What problem does FlashAttention solve?"},
        )

    assert response.status_code == 401
    assert "secret-key" not in caplog.text
