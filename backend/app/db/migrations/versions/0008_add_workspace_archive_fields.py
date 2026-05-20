"""Add workspace archive fields.

Revision ID: 0008_workspace_archive
Revises: 0007_add_workspace_foreign_keys
Create Date: 2026-05-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_workspace_archive"
down_revision: str | None = "0007_add_workspace_foreign_keys"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "workspaces",
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "workspaces",
        sa.Column("archived_reason", sa.Text(), nullable=True),
    )
    op.create_index("workspaces_archived_at_idx", "workspaces", ["archived_at"])


def downgrade() -> None:
    op.drop_index("workspaces_archived_at_idx", table_name="workspaces")
    op.drop_column("workspaces", "archived_reason")
    op.drop_column("workspaces", "archived_at")
