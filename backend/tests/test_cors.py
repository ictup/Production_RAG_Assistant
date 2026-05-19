from fastapi.testclient import TestClient

from backend.app.core.config import Settings, get_settings
from backend.app.core.cors import parse_cors_allowed_origins
from backend.app.main import create_app


def test_parse_cors_allowed_origins_ignores_blanks() -> None:
    parsed_origins = parse_cors_allowed_origins(
        " http://localhost:5173, ,https://app.example.com "
    )

    assert parsed_origins == [
        "http://localhost:5173",
        "https://app.example.com",
    ]


def test_cors_headers_are_not_sent_when_not_configured() -> None:
    client = TestClient(create_app(Settings(cors_allowed_origins="")))

    response = client.get("/health", headers={"Origin": "https://app.example.com"})

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_cors_allows_configured_origin() -> None:
    client = TestClient(
        create_app(
            Settings(
                cors_allowed_origins=(
                    "http://localhost:5173,https://app.example.com"
                )
            )
        )
    )

    response = client.get("/health", headers={"Origin": "https://app.example.com"})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "https://app.example.com"
    )
    assert response.headers["access-control-expose-headers"] == "X-Request-ID"


def test_cors_rejects_unconfigured_origin() -> None:
    client = TestClient(
        create_app(Settings(cors_allowed_origins="https://app.example.com"))
    )

    response = client.get("/health", headers={"Origin": "https://evil.example.com"})

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_cors_preflight_allows_api_headers() -> None:
    client = TestClient(
        create_app(Settings(cors_allowed_origins="http://localhost:5173"))
    )

    response = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": (
                "Authorization,Content-Type,X-Workspace-ID,X-Request-ID"
            ),
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "http://localhost:5173"
    )
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "Authorization" in response.headers["access-control-allow-headers"]


def test_settings_load_cors_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://app.example.com")
    monkeypatch.setenv("CORS_ALLOWED_ORIGIN_REGEX", r"https://.*\.example\.com")
    monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "true")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.cors_allowed_origins == "https://app.example.com"
    assert settings.cors_allowed_origin_regex == r"https://.*\.example\.com"
    assert settings.cors_allow_credentials is True

    get_settings.cache_clear()
