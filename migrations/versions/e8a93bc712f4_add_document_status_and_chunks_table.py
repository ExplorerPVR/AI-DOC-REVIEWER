"""add document status and chunks table with pgvector

Revision ID: e8a93bc712f4
Revises: 43ce2c5fbd08
Create Date: 2026-11-15 18:25:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "e8a93bc712f4"
down_revision: str | Sequence[str] | None = "43ce2c5fbd08"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema: enable pgvector, add document status, and create chunks table."""
    # 1. Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 2. Add status, error_message, file_path to documents table
    op.add_column(
        "documents",
        sa.Column(
            "status",
            sqlmodel.sql.sqltypes.AutoString(),
            server_default="pending",
            nullable=False,
        ),
    )
    op.add_column(
        "documents",
        sa.Column("error_message", sa.Text(), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column(
            "file_path",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=True,
        ),
    )
    op.create_index(
        op.f("ix_documents_status"), "documents", ["status"], unique=False
    )

    # 3. Create chunks table
    op.create_table(
        "chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("filename", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column(
            "created_at",
            sqlmodel.sql.sqltypes.UTCDateTime(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_chunks_id"), "chunks", ["id"], unique=False)
    op.create_index(
        op.f("ix_chunks_document_id"), "chunks", ["document_id"], unique=False
    )
    op.create_index(
        op.f("ix_chunks_filename"), "chunks", ["filename"], unique=False
    )
    # 4. Create HNSW index for fast approximate nearest-neighbor vector search
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks")
    op.drop_index(op.f("ix_chunks_filename"), table_name="chunks")
    op.drop_index(op.f("ix_chunks_document_id"), table_name="chunks")
    op.drop_index(op.f("ix_chunks_id"), table_name="chunks")
    op.drop_table("chunks")

    op.drop_index(op.f("ix_documents_status"), table_name="documents")
    op.drop_column("documents", "file_path")
    op.drop_column("documents", "error_message")
    op.drop_column("documents", "status")

    op.execute("DROP EXTENSION IF EXISTS vector")
