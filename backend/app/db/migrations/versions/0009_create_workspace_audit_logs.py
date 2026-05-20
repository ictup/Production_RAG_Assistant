"""Create workspace audit log table.

Revision ID: 0009_workspace_audit_logs
Revises: 0008_workspace_archive
Create Date: 2026-05-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0009_workspace_audit_logs"
down_revision: str | None = "0008_workspace_archive"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workspace_audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("request_id", sa.Text(), nullable=False),
        sa.Column("actor_hash", sa.String(length=64), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column(
            "workspace_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("workspace_count", sa.Integer(), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "workspace_audit_logs_created_at_idx",
        "workspace_audit_logs",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "workspace_audit_logs_request_id_idx",
        "workspace_audit_logs",
        ["request_id"],
        unique=False,
    )
    op.create_index(
        "workspace_audit_logs_workspace_ids_idx",
        "workspace_audit_logs",
        ["workspace_ids"],
        unique=False,
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index(
        "workspace_audit_logs_workspace_ids_idx",
        table_name="workspace_audit_logs",
    )
    op.drop_index(
        "workspace_audit_logs_request_id_idx",
        table_name="workspace_audit_logs",
    )
    op.drop_index(
        "workspace_audit_logs_created_at_idx",
        table_name="workspace_audit_logs",
    )
    op.drop_table("workspace_audit_logs")
