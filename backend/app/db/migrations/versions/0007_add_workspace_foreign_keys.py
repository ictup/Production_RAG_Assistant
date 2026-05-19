"""Add workspace foreign keys.

Revision ID: 0007_add_workspace_foreign_keys
Revises: 0006_create_workspaces
Create Date: 2026-05-20
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0007_add_workspace_foreign_keys"
down_revision: str | None = "0006_create_workspaces"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


WORKSPACE_BACKFILL_SQL = """
INSERT INTO workspaces (id, name, description, metadata)
SELECT workspace_id, workspace_id, 'Backfilled from existing workspace_id references',
       '{}'::jsonb
FROM (
    SELECT DISTINCT workspace_id FROM documents
    UNION
    SELECT DISTINCT workspace_id FROM document_chunks
    UNION
    SELECT DISTINCT workspace_id FROM chat_sessions
    UNION
    SELECT DISTINCT workspace_id FROM chat_logs
) AS existing_workspaces
WHERE workspace_id IS NOT NULL
  AND btrim(workspace_id) <> ''
ON CONFLICT (id) DO NOTHING
"""


def upgrade() -> None:
    op.execute(WORKSPACE_BACKFILL_SQL)
    op.create_foreign_key(
        "documents_workspace_id_fkey",
        "documents",
        "workspaces",
        ["workspace_id"],
        ["id"],
        ondelete="RESTRICT",
        onupdate="CASCADE",
    )
    op.create_foreign_key(
        "document_chunks_workspace_id_fkey",
        "document_chunks",
        "workspaces",
        ["workspace_id"],
        ["id"],
        ondelete="RESTRICT",
        onupdate="CASCADE",
    )
    op.create_foreign_key(
        "chat_sessions_workspace_id_fkey",
        "chat_sessions",
        "workspaces",
        ["workspace_id"],
        ["id"],
        ondelete="RESTRICT",
        onupdate="CASCADE",
    )
    op.create_foreign_key(
        "chat_logs_workspace_id_fkey",
        "chat_logs",
        "workspaces",
        ["workspace_id"],
        ["id"],
        ondelete="RESTRICT",
        onupdate="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("chat_logs_workspace_id_fkey", "chat_logs", type_="foreignkey")
    op.drop_constraint(
        "chat_sessions_workspace_id_fkey",
        "chat_sessions",
        type_="foreignkey",
    )
    op.drop_constraint(
        "document_chunks_workspace_id_fkey",
        "document_chunks",
        type_="foreignkey",
    )
    op.drop_constraint("documents_workspace_id_fkey", "documents", type_="foreignkey")
