"""SQLModel database models and Pydantic schemas for AI Document Assistant."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel
from pydantic import Field as PydanticField
from sqlmodel import Field as SQLModelField
from sqlmodel import SQLModel


class HealthResponse(BaseModel):
    """Response schema for the /health status check endpoint."""

    status: str = PydanticField(
        default="ok",
        description="Current health status of the service",
        examples=["ok"],
    )


class DocumentBase(SQLModel):
    """Base schema holding common document metadata fields."""

    filename: str = SQLModelField(
        description="Original name of the uploaded document file",
    )
    content_type: str = SQLModelField(
        default="application/octet-stream",
        description="MIME type of the uploaded document",
    )
    size_bytes: int = SQLModelField(
        description="Total size of the document in bytes",
        ge=1,
    )


class Document(DocumentBase, table=True):
    """SQLModel database table model for persisting document metadata."""

    __tablename__ = "documents"

    id: UUID = SQLModelField(
        default_factory=uuid4,
        primary_key=True,
        index=True,
        description="Unique UUID primary key for the document",
    )
    uploaded_at: datetime = SQLModelField(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp in UTC recording when the document was received",
    )


class DocumentResponse(DocumentBase):
    """Response schema returned upon document upload or retrieval."""

    id: UUID = SQLModelField(
        description="Unique UUID identifier for the document",
    )
    uploaded_at: datetime = SQLModelField(
        description="Timestamp in UTC recording when the document was received",
    )


class ErrorResponse(BaseModel):
    """Standard schema returned for client or validation errors."""

    detail: str = PydanticField(
        description="Human-readable explanation of the validation or operational error",
        examples=["Unsupported file type '.png'. Allowed types are: .pdf, .txt, .docx"],
    )
