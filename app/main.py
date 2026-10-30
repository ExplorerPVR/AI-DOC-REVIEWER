"""FastAPI application entrypoint and routes for AI Document Assistant."""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from sqlmodel import Session, select

from app.db import get_session, init_db
from app.models import (
    Document,
    DocumentResponse,
    ErrorResponse,
    HealthResponse,
)

# Constants for document upload validation
MAX_FILE_SIZE_BYTES: int = int(
    os.getenv("MAX_FILE_SIZE_BYTES", str(10 * 1024 * 1024))
)  # 10 MB limit
CHUNK_SIZE_BYTES: int = 1024 * 1024  # 1 MB read chunk size
ALLOWED_EXTENSIONS: set[str] = {".pdf", ".txt", ".docx"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Execute startup database initialization and handle clean shutdown."""
    init_db()
    yield


app = FastAPI(
    title="AI Document Assistant",
    description=(
        "FastAPI service for ingesting and cataloging documents (.pdf, .txt, .docx). "
        "Provides PostgreSQL-backed persistence and containerized deployment."
    ),
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
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
        "Validates file type and size, storing metadata in PostgreSQL."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Empty file provided"},
        413: {"model": ErrorResponse, "description": "File exceeds 10 MB maximum size"},
        415: {"model": ErrorResponse, "description": "Unsupported file extension"},
    },
    tags=["Documents"],
)
async def upload_document(
    file: Annotated[
        UploadFile,
        File(description="Document file to upload (.pdf, .txt, .docx)"),
    ],
    session: Annotated[Session, Depends(get_session)],
) -> Document:
    """Validate uploaded document and record metadata in PostgreSQL."""
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

    # 2. Fast-path check: inspect file.size if populated to avoid unnecessary I/O
    if file.size is not None and file.size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"File too large ({file.size} bytes). "
                f"Maximum allowed size is 10 MB ({MAX_FILE_SIZE_BYTES} bytes)."
            ),
        )

    # 3. Stream content in chunks to measure size without buffering into memory
    size_bytes = 0
    while chunk := await file.read(CHUNK_SIZE_BYTES):
        size_bytes += len(chunk)
        if size_bytes > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=(
                    f"File too large ({size_bytes} bytes). "
                    f"Maximum allowed size is 10 MB ({MAX_FILE_SIZE_BYTES} bytes)."
                ),
            )

    # 4. Validate non-empty file
    if size_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file. Uploaded file contains no data.",
        )

    # 5. Extract content type with fallback
    content_type = file.content_type or "application/octet-stream"

    # 6. Create and persist Document model to database
    doc_record = Document(
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
    )
    session.add(doc_record)
    session.commit()
    session.refresh(doc_record)

    return doc_record


@app.get(
    "/documents",
    response_model=list[DocumentResponse],
    status_code=status.HTTP_200_OK,
    summary="List Documents",
    description="Retrieve metadata for all uploaded documents stored in PostgreSQL.",
    tags=["Documents"],
)
async def list_documents(
    session: Annotated[Session, Depends(get_session)],
) -> list[Document]:
    """Return all stored document metadata records from PostgreSQL."""
    statement = select(Document).order_by(Document.uploaded_at)
    documents = session.exec(statement).all()
    return list(documents)
