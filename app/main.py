"""FastAPI application entrypoint and routes for AI Document Assistant."""

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Annotated
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile, status

from app.models import (
    DocumentResponse,
    ErrorResponse,
    HealthResponse,
)

# Constants for document upload validation
MAX_FILE_SIZE_BYTES: int = int(
    os.getenv("MAX_FILE_SIZE_BYTES", 10 * 1024 * 1024)
)  # 10 MB limit
ALLOWED_EXTENSIONS: set[str] = {".pdf", ".txt", ".docx"}

# In-memory document storage (supports both DOCUMENTS_DB and documents_db naming)
DOCUMENTS_DB: list[dict] = []
documents_db = DOCUMENTS_DB

app = FastAPI(
    title="AI Document Assistant",
    description=(
        "FastAPI service for ingesting and cataloging documents (.pdf, .txt, .docx). "
        "Provides in-memory metadata storage and health monitoring."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Returns service availability and operational health status.",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Return health status probe."""
    return HealthResponse(status="ok")


@app.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a Document",
    description=(
        "Upload a single document (.pdf, .txt, or .docx) up to 10 MB. "
        "Validates file type and size, storing metadata in memory."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Empty file provided"},
        413: {"model": ErrorResponse, "description": "File exceeds 10 MB maximum size"},
        415: {"model": ErrorResponse, "description": "Unsupported file extension"},
    },
    tags=["Documents"],
)
async def upload_document(
    file: Annotated[UploadFile, File(description="Document file to upload (.pdf, .txt, .docx)")]
) -> DocumentResponse:
    """Validate uploaded document and record metadata in memory."""
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    # 1. Validate file extension (Allowed: .pdf, .txt, .docx)
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{extension}'. Allowed types are: "
                f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    # Read uploaded bytes to verify content size
    contents = await file.read()
    size_bytes = len(contents)

    # 2. Validate non-empty file
    if size_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file. Uploaded file contains no data.",
        )

    # 3. Validate maximum size limit (10 MB)
    if size_bytes > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"File too large ({size_bytes} bytes). "
                f"Maximum allowed size is 10 MB ({MAX_FILE_SIZE_BYTES} bytes)."
            ),
        )

    # Assign universally unique identifier (UUID v4) and current UTC timestamp
    doc_id = str(uuid.uuid4())
    uploaded_at = datetime.now(timezone.utc)

    doc_record = {
        "id": doc_id,
        "filename": filename,
        "size_bytes": size_bytes,
        "uploaded_at": uploaded_at,
    }

    # Save to in-memory store
    DOCUMENTS_DB.append(doc_record)

    return DocumentResponse(**doc_record)


@app.get(
    "/documents",
    response_model=list[DocumentResponse],
    status_code=status.HTTP_200_OK,
    summary="List Documents",
    description="Retrieve metadata for all uploaded documents stored in memory.",
    tags=["Documents"],
)
async def list_documents() -> list[DocumentResponse]:
    """Return all stored document metadata records."""
    return [DocumentResponse(**doc) for doc in DOCUMENTS_DB]
