from backend.app.core.config import Settings
from backend.app.core.config_check import check_settings, format_report


def make_settings(**overrides: object) -> Settings:
    values = {"api_keys": "dev-key", **overrides}
    return Settings(_env_file=None, **values)


def test_config_check_passes_local_default_settings() -> None:
    report = check_settings(make_settings())

    assert report.passed is True
    assert report.mode == "local"
    assert report.errors == []


def test_config_check_requires_openai_key_when_provider_enabled() -> None:
    report = check_settings(
        make_settings(
            embedding_provider="openai",
            openai_api_key=" ",
        )
    )

    assert report.passed is False
    assert [issue.variable for issue in report.errors] == ["OPENAI_API_KEY"]
    assert "OpenAI provider" in report.errors[0].message


def test_config_check_rejects_insecure_production_api_keys() -> None:
    report = check_settings(
        make_settings(
            env="production",
            api_keys="dev-key,short",
            api_key_workspace_access="dev-key=*",
        )
    )

    assert report.passed is False
    assert [issue.variable for issue in report.errors] == ["API_KEYS"]
    assert "short tokens" in report.errors[0].message


def test_config_check_accepts_stronger_production_api_key() -> None:
    report = check_settings(
        make_settings(
            env="production",
            api_keys="prod-token-1234567890",
            api_key_workspace_access="prod-token-1234567890=public",
            database_url="postgresql+asyncpg://u:p@postgres:5432/rag",
            sync_database_url="postgresql+psycopg://u:p@postgres:5432/rag",
            rate_limit_enabled=True,
        )
    )

    assert report.passed is True
    assert report.errors == []
    assert report.warnings == []


def test_config_check_reports_production_warnings() -> None:
    report = check_settings(
        make_settings(
            env="production",
            api_keys="prod-token-1234567890",
            database_url="postgresql+asyncpg://u:p@localhost:5432/rag",
            sync_database_url="postgresql+psycopg://u:p@127.0.0.1:5432/rag",
        )
    )

    assert report.passed is True
    assert {issue.variable for issue in report.warnings} == {
        "API_KEY_WORKSPACE_ACCESS",
        "DATABASE_URL",
        "RATE_LIMIT_ENABLED",
        "SYNC_DATABASE_URL",
    }


def test_config_check_rejects_wildcard_cors_credentials() -> None:
    report = check_settings(
        make_settings(
            cors_allowed_origins="https://app.example.com,*",
            cors_allow_credentials=True,
        )
    )

    assert report.passed is False
    assert [issue.variable for issue in report.errors] == ["CORS_ALLOW_CREDENTIALS"]


def test_config_check_can_force_production_mode() -> None:
    report = check_settings(
        make_settings(api_keys="dev-key"),
        strict_production=True,
    )

    assert report.mode == "production"
    assert report.passed is False
    assert [issue.variable for issue in report.errors] == ["API_KEYS"]


def test_format_report_does_not_include_secret_values() -> None:
    report = check_settings(
        make_settings(
            env="production",
            api_keys="prod-secret-token-1234567890",
            openai_api_key="secret-token-should-not-print",
            api_key_workspace_access="",
        )
    )

    formatted = format_report(report)

    assert "prod-secret-token-1234567890" not in formatted
    assert "secret-token-should-not-print" not in formatted
    assert "API_KEY_WORKSPACE_ACCESS" in formatted
    assert "configuration check mode: production" in formatted
