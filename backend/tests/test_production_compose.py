from pathlib import Path

import yaml

COMPOSE_PATH = Path("docker-compose.prod.yml")
MAKEFILE_PATH = Path("Makefile")


def load_production_compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


def test_production_compose_defines_api_worker_migrate_and_postgres() -> None:
    compose = load_production_compose()
    services = compose["services"]

    assert COMPOSE_PATH.exists()
    assert set(services) == {"api", "export-worker", "migrate", "postgres"}
    assert services["postgres"]["image"] == "pgvector/pgvector:pg16"
    assert services["api"]["image"] == "production-rag-assistant:${APP_VERSION:-local}"
    assert services["export-worker"]["image"] == (
        "production-rag-assistant:${APP_VERSION:-local}"
    )
    assert services["api"]["build"]["dockerfile"] == "Dockerfile"


def test_production_compose_uses_internal_database_urls() -> None:
    compose = load_production_compose()

    for service_name in ("api", "export-worker", "migrate"):
        environment = compose["services"][service_name]["environment"]
        assert "@postgres:5432" in environment["DATABASE_URL"]
        assert "@postgres:5432" in environment["SYNC_DATABASE_URL"]
        assert "localhost" not in environment["DATABASE_URL"]
        assert environment["ENV"] == "production"


def test_production_compose_orders_healthcheck_migration_and_api() -> None:
    compose = load_production_compose()
    services = compose["services"]

    assert services["postgres"]["healthcheck"]["test"][0] == "CMD-SHELL"
    assert services["migrate"]["command"] == ["alembic", "upgrade", "head"]
    assert services["migrate"]["depends_on"]["postgres"]["condition"] == (
        "service_healthy"
    )
    assert services["api"]["depends_on"]["migrate"]["condition"] == (
        "service_completed_successfully"
    )
    assert services["api"]["healthcheck"]["test"][0] == "CMD"
    assert services["export-worker"]["depends_on"]["migrate"]["condition"] == (
        "service_completed_successfully"
    )
    assert services["export-worker"]["restart"] == "unless-stopped"


def test_production_compose_runs_export_worker_with_shared_export_volume() -> None:
    compose = load_production_compose()
    services = compose["services"]

    assert services["export-worker"]["command"] == [
        "python",
        "-m",
        "backend.app.exporting.worker",
        "--loop",
    ]
    assert services["api"]["environment"]["EXPORT_STORAGE_DIR"] == (
        "${EXPORT_STORAGE_DIR:-/app/exports}"
    )
    assert services["export-worker"]["environment"]["EXPORT_STORAGE_DIR"] == (
        "${EXPORT_STORAGE_DIR:-/app/exports}"
    )
    assert "export_prod_data:/app/exports" in services["api"]["volumes"]
    assert "export_prod_data:/app/exports" in services["export-worker"]["volumes"]
    assert "export_prod_data" in compose["volumes"]


def test_production_postgres_enables_slow_query_observability() -> None:
    compose = load_production_compose()
    command = " ".join(compose["services"]["postgres"]["command"])

    assert "shared_preload_libraries=pg_stat_statements" in command
    assert "pg_stat_statements.track=all" in command
    assert "track_io_timing=on" in command
    assert "log_min_duration_statement=" in command


def test_production_makefile_targets_exist() -> None:
    content = MAKEFILE_PATH.read_text(encoding="utf-8")

    assert "prod-config:" in content
    assert "docker compose -f docker-compose.prod.yml config --quiet" in content
    assert "config-check:" in content
    assert "prod-config-check:" in content
    assert "python -m backend.app.core.config_check --production" in content
    assert "prod-build:" in content
    assert "prod-up:" in content
    assert "prod-down:" in content
    assert "prod-logs:" in content
    assert "prod-worker-logs:" in content
