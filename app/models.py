"""Request and response schemas for the AI Document Assistant API."""

from datetime import datetime, timezone
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Response schema for the /health status check endpoint."""

    status: str = Field(
        default="ok",
        description="Current health status of the service",
        examples=["ok"],
    )


class DocumentBase(BaseModel):
    """Base schema holding core document metadata."""

    filename: str = Field(
        ...,
        description="Original name of the uploaded document file",
        examples=["contract.pdf"],
    )
    size_bytes: int = Field(
        ...,
        description="Total size of the document in bytes",
        ge=1,
        examples=[1048576],
    )


class DocumentCreate(DocumentBase):
    """Schema representing document creation attributes."""

    pass


class DocumentResponse(DocumentBase):
    """Response schema returned upon document upload or retrieval."""

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique UUID identifier for the document",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp in UTC recording when the document was received",
    )

    model_config = ConfigDict(from_attributes=True)


class ErrorResponse(BaseModel):
    """Standard schema returned for client or validation errors."""

    detail: str = Field(
        ...,
        description="Human-readable explanation of the validation or operational error",
        examples=["Unsupported file type '.png'. Allowed types are: .pdf, .txt, .docx"],
    )
