"""Create support tickets table.

Revision ID: 0011_support_tickets
Revises: 0010_export_jobs
Create Date: 2026-05-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0011_support_tickets"
down_revision: str | None = "0010_export_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "support_tickets",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ticket_id", sa.Text(), nullable=False),
        sa.Column(
            "workspace_id",
            sa.Text(),
            server_default=sa.text("'public'"),
            nullable=False,
        ),
        sa.Column("category", sa.Text(), nullable=True),
        sa.Column("customer_message", sa.Text(), nullable=False),
        sa.Column("resolution_summary", sa.Text(), nullable=True),
        sa.Column("final_response", sa.Text(), nullable=True),
        sa.Column(
            "tags",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'::text[]"),
            nullable=False,
        ),
        sa.Column("risk_level", sa.Text(), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            ondelete="RESTRICT",
            onupdate="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ticket_id", name="support_tickets_ticket_id_key"),
    )
    op.create_index(
        "support_tickets_workspace_created_at_idx",
        "support_tickets",
        ["workspace_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "support_tickets_category_idx",
        "support_tickets",
        ["category"],
        unique=False,
    )
    op.create_index(
        "support_tickets_tags_idx",
        "support_tickets",
        ["tags"],
        unique=False,
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("support_tickets_tags_idx", table_name="support_tickets")
    op.drop_index("support_tickets_category_idx", table_name="support_tickets")
    op.drop_index(
        "support_tickets_workspace_created_at_idx",
        table_name="support_tickets",
    )
    op.drop_table("support_tickets")
