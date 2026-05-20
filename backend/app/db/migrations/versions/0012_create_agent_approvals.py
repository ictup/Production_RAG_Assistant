"""Create agent approvals table.

Revision ID: 0012_agent_approvals
Revises: 0011_support_tickets
Create Date: 2026-05-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0012_agent_approvals"
down_revision: str | None = "0011_support_tickets"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_approvals",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("ticket_id", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.Text(), nullable=False),
        sa.Column("request_id", sa.Text(), nullable=False),
        sa.Column("actor_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.Text(),
            server_default=sa.text("'pending'"),
            nullable=False,
        ),
        sa.Column("category", sa.Text(), nullable=True),
        sa.Column("risk_level", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("customer_message", sa.Text(), nullable=False),
        sa.Column("draft_answer", sa.Text(), nullable=False),
        sa.Column(
            "tool_calls",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "node_runs",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("human_feedback", sa.Text(), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'approved', 'rejected')",
            name="agent_approvals_status_check",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            ondelete="RESTRICT",
            onupdate="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", name="agent_approvals_run_id_key"),
    )
    op.create_index(
        "agent_approvals_workspace_created_at_idx",
        "agent_approvals",
        ["workspace_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "agent_approvals_status_created_at_idx",
        "agent_approvals",
        ["status", "created_at"],
        unique=False,
    )
    op.create_index(
        "agent_approvals_request_id_idx",
        "agent_approvals",
        ["request_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("agent_approvals_request_id_idx", table_name="agent_approvals")
    op.drop_index(
        "agent_approvals_status_created_at_idx",
        table_name="agent_approvals",
    )
    op.drop_index(
        "agent_approvals_workspace_created_at_idx",
        table_name="agent_approvals",
    )
    op.drop_table("agent_approvals")
