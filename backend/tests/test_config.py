from backend.app.core.config import get_settings


def test_settings_load_database_urls_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/db")
    monkeypatch.setenv("SYNC_DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/db")
    monkeypatch.setenv("EXPORT_STORAGE_DIR", "var/exports")
    monkeypatch.setenv("EXPORT_WORKER_POLL_INTERVAL_SECONDS", "1.5")
    monkeypatch.setenv("EXPORT_JOB_RUNNING_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("EXPORT_FILE_RETENTION_SECONDS", "86400")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.database_url == "postgresql+asyncpg://u:p@localhost:5432/db"
    assert settings.sync_database_url == "postgresql+psycopg://u:p@localhost:5432/db"
    assert settings.export_storage_dir == "var/exports"
    assert settings.export_worker_poll_interval_seconds == 1.5
    assert settings.export_job_running_timeout_seconds == 120
    assert settings.export_file_retention_seconds == 86400

    get_settings.cache_clear()
