"""Create workspace registry table.

Revision ID: 0006_create_workspaces
Revises: 0005_enable_pg_stat_statements
Create Date: 2026-05-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006_create_workspaces"
down_revision: str | None = "0005_enable_pg_stat_statements"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workspaces",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
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
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "workspaces_updated_at_idx",
        "workspaces",
        ["updated_at"],
        unique=False,
    )
    op.execute(
        """
        INSERT INTO workspaces (id, name, description, metadata)
        VALUES ('public', 'Public', 'Default local workspace', '{}'::jsonb)
        ON CONFLICT (id) DO NOTHING
        """
    )


def downgrade() -> None:
    op.drop_index("workspaces_updated_at_idx", table_name="workspaces")
    op.drop_table("workspaces")
