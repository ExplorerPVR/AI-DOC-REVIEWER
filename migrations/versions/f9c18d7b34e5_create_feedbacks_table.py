"""create feedbacks table

Revision ID: f9c18d7b34e5
Revises: e8a93bc712f4
Create Date: 2026-11-24 15:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f9c18d7b34e5"
down_revision: str | Sequence[str] | None = "e8a93bc712f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema: create feedbacks table."""
    op.create_table(
        "feedbacks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("trace_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_feedbacks_id"), "feedbacks", ["id"], unique=False)
    op.create_index(
        op.f("ix_feedbacks_trace_id"), "feedbacks", ["trace_id"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema: drop feedbacks table."""
    op.drop_index(op.f("ix_feedbacks_trace_id"), table_name="feedbacks")
    op.drop_index(op.f("ix_feedbacks_id"), table_name="feedbacks")
    op.drop_table("feedbacks")
