import json
import logging

from fastapi.testclient import TestClient

from backend.app.core.logging import (
    REQUEST_LOGGER_NAME,
    build_request_log_payload,
    serialize_log_payload,
)
from backend.app.core.request_id import REQUEST_ID_HEADER
from backend.app.main import create_app


def parse_request_logs(caplog) -> list[dict]:
    return [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == REQUEST_LOGGER_NAME
    ]


def test_build_request_log_payload_uses_safe_metadata_only() -> None:
    payload = build_request_log_payload(
        request_id="request-1",
        method="POST",
        path="/chat",
        status_code=200,
        latency_ms=12,
    )

    assert payload == {
        "event": "http_request",
        "request_id": "request-1",
        "method": "POST",
        "path": "/chat",
        "status_code": 200,
        "latency_ms": 12,
    }


def test_serialize_log_payload_returns_compact_json() -> None:
    assert serialize_log_payload({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_request_logging_middleware_logs_health_request(caplog) -> None:
    client = TestClient(create_app())

    with caplog.at_level(logging.INFO, logger=REQUEST_LOGGER_NAME):
        response = client.get(
            "/health",
            headers={REQUEST_ID_HEADER: "log-request-1"},
        )

    assert response.status_code == 200
    logs = parse_request_logs(caplog)
    assert len(logs) == 1
    assert logs[0]["event"] == "http_request"
    assert logs[0]["request_id"] == "log-request-1"
    assert logs[0]["method"] == "GET"
    assert logs[0]["path"] == "/health"
    assert logs[0]["status_code"] == 200
    assert isinstance(logs[0]["latency_ms"], int)


def test_request_logging_middleware_does_not_log_authorization(caplog) -> None:
    client = TestClient(create_app())

    with caplog.at_level(logging.INFO, logger=REQUEST_LOGGER_NAME):
        response = client.post(
            "/chat",
            headers={
                "Authorization": "Bearer secret-key",
                REQUEST_ID_HEADER: "log-request-2",
            },
            json={"question": "What problem does FlashAttention solve?"},
        )

    assert response.status_code == 401
    logs = parse_request_logs(caplog)
    assert len(logs) == 1
    assert logs[0]["request_id"] == "log-request-2"
    assert logs[0]["path"] == "/chat"
    assert logs[0]["status_code"] == 401
    assert "secret-key" not in json.dumps(logs[0])
