"""create llm_calls table

Revision ID: 43ce2c5fbd08
Revises: bb20b80bdc64
Create Date: 2026-11-04 12:15:01.771020

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "43ce2c5fbd08"
down_revision: str | Sequence[str] | None = "bb20b80bdc64"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "llm_calls",
        sa.Column("provider", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("model", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("prompt", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("time_taken_seconds", sa.Float(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_llm_calls_id"), "llm_calls", ["id"], unique=False)
    op.create_index(
        op.f("ix_llm_calls_provider"), "llm_calls", ["provider"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_llm_calls_provider"), table_name="llm_calls")
    op.drop_index(op.f("ix_llm_calls_id"), table_name="llm_calls")
    op.drop_table("llm_calls")
