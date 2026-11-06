"""SQLModel database models and Pydantic schemas for AI Document Assistant."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, field_validator
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


class LLMCallBase(SQLModel):
    """Base schema holding metadata for tracked LLM generations."""

    provider: str = SQLModelField(
        index=True,
        description="LLM provider name (e.g. gemini, groq, ollama)",
    )
    model: str = SQLModelField(
        description="Exact model identifier used for generation",
    )
    prompt: str | None = SQLModelField(
        default=None,
        description="Input prompt or question asked",
    )
    input_tokens: int = SQLModelField(
        default=0,
        description="Estimated or measured input/prompt token count",
    )
    output_tokens: int = SQLModelField(
        default=0,
        description="Estimated or measured output/completion token count",
    )
    time_taken_seconds: float = SQLModelField(
        description="Execution duration for the generation in seconds",
    )
    cost_usd: float = SQLModelField(
        default=0.0,
        description="Calculated cost in USD at published paid list price",
    )


class LLMCall(LLMCallBase, table=True):
    """SQLModel database table model for persisting LLM call metrics and costs."""

    __tablename__ = "llm_calls"

    id: UUID = SQLModelField(
        default_factory=uuid4,
        primary_key=True,
        index=True,
        description="Unique UUID primary key for the LLM call record",
    )
    created_at: datetime = SQLModelField(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp in UTC when the LLM call occurred",
    )


class LLMCallResponse(LLMCallBase):
    """Response schema returned for LLM call records."""

    id: UUID = SQLModelField(
        description="Unique UUID identifier for the call record",
    )
    created_at: datetime = SQLModelField(
        description="Timestamp in UTC recording when the call occurred",
    )


class AskRequest(BaseModel):
    """Request payload schema for the POST /ask endpoint."""

    question: str = PydanticField(
        min_length=1,
        description="Question or prompt to submit to the LLM",
        examples=["Explain what a database index does in 2 sentences."],
    )
    provider: str = PydanticField(
        default="gemini",
        description="Target LLM provider (gemini, groq, or ollama)",
        examples=["gemini"],
    )


class ErrorResponse(BaseModel):
    """Standard schema returned for client or validation errors."""

    detail: str = PydanticField(
        description="Human-readable explanation of the validation or operational error",
        examples=["Unsupported file type '.png'. Allowed types are: .pdf, .txt, .docx"],
    )


class ResumeData(BaseModel):
    """Structured resume information extracted by LLM."""

    name: str | None = PydanticField(
        default=None,
        description="Candidate's full legal name, or null if missing from resume",
    )
    email: str | None = PydanticField(
        default=None,
        description="Candidate's contact email address, or null if missing",
    )
    phone: str | None = PydanticField(
        default=None,
        description="Candidate's phone number, or null if missing",
    )
    skills: list[str] = PydanticField(
        default_factory=list,
        description="List of candidate's technical and professional skills",
    )

    @field_validator("name", "email", "phone", mode="before")
    @classmethod
    def normalize_optional_fields(cls, value: Any) -> str | None:
        """Ensure missing or placeholder values become null."""
        if value is None:
            return None
        cleaned = str(value).strip()
        if not cleaned or cleaned.lower() in {
            "null",
            "none",
            "n/a",
            "na",
            "not provided",
            "unknown",
        }:
            return None
        return cleaned

    @field_validator("skills", mode="before")
    @classmethod
    def normalize_skills(cls, value: Any) -> list[str]:
        """Normalize skills into a clean list of non-empty strings."""
        if value is None:
            return []
        if isinstance(value, list):
            result = []
            for item in value:
                if item is not None:
                    s = str(item).strip()
                    if s and s not in result:
                        result.append(s)
            return result
        if isinstance(value, str):
            parts = [s.strip() for s in value.split(",") if s.strip()]
            return parts
        return []


class ExtractRequest(BaseModel):
    """Request payload schema for the POST /extract endpoint."""

    text: str | None = PydanticField(
        default=None,
        description="Resume text content to extract information from",
    )
    resume_text: str | None = PydanticField(
        default=None,
        description="Alternative key for resume text content",
    )
    provider: str = PydanticField(
        default="gemini",
        description="Target LLM provider ('gemini', 'groq', 'ollama')",
        examples=["gemini"],
    )
