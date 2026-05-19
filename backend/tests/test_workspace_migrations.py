from pathlib import Path

MIGRATION_PATH = Path(
    "backend/app/db/migrations/versions/0007_add_workspace_foreign_keys.py"
)


def test_workspace_foreign_key_migration_backfills_existing_workspace_ids() -> None:
    migration = MIGRATION_PATH.read_text(encoding="utf-8")

    assert 'down_revision: str | None = "0006_create_workspaces"' in migration
    assert "SELECT DISTINCT workspace_id FROM documents" in migration
    assert "SELECT DISTINCT workspace_id FROM document_chunks" in migration
    assert "SELECT DISTINCT workspace_id FROM chat_sessions" in migration
    assert "SELECT DISTINCT workspace_id FROM chat_logs" in migration
    assert "ON CONFLICT (id) DO NOTHING" in migration


def test_workspace_foreign_key_migration_adds_all_workspace_constraints() -> None:
    migration = MIGRATION_PATH.read_text(encoding="utf-8")

    for constraint_name in (
        "documents_workspace_id_fkey",
        "document_chunks_workspace_id_fkey",
        "chat_sessions_workspace_id_fkey",
        "chat_logs_workspace_id_fkey",
    ):
        assert constraint_name in migration

    assert 'ondelete="RESTRICT"' in migration
    assert 'onupdate="CASCADE"' in migration
