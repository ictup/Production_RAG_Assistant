import uuid

from fastapi.testclient import TestClient

from backend.app.core.request_id import (
    REQUEST_ID_HEADER,
    generate_request_id,
    normalize_request_id,
)
from backend.app.main import create_app


def test_generate_request_id_returns_uuid_string() -> None:
    request_id = generate_request_id()

    assert str(uuid.UUID(request_id)) == request_id


def test_normalize_request_id_uses_trimmed_client_value() -> None:
    assert normalize_request_id("  client-id  ") == "client-id"


def test_normalize_request_id_generates_when_missing_or_blank() -> None:
    assert uuid.UUID(normalize_request_id(None))
    assert uuid.UUID(normalize_request_id("   "))


def test_request_id_middleware_adds_response_header() -> None:
    client = TestClient(create_app())

    response = client.get("/health", headers={REQUEST_ID_HEADER: "client-id"})

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "client-id"


def test_request_id_middleware_generates_response_header() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert uuid.UUID(response.headers[REQUEST_ID_HEADER])
