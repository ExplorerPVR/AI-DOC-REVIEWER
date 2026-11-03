"""FastAPI application entrypoint and routes for AI Document Assistant."""

import io
import os
import zipfile
from pathlib import Path
from typing import Annotated

from docx import Document as DocxDocument
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sqlmodel import Session, select

from app.db import get_session
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

app = FastAPI(
    title="AI Document Assistant",
    description=(
        "FastAPI service for ingesting and cataloging documents (.pdf, .txt, .docx). "
        "Provides PostgreSQL-backed persistence and containerized deployment."
    ),
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


def validate_file_content(content: bytes, extension: str) -> None:
    """Validate content is not empty, whitespace-only, blank, or corrupted."""
    if not content or not content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty or contains only whitespace.",
        )

    # 1. Plain text validation (.txt)
    if extension == ".txt":
        try:
            text = content.decode("utf-8").strip()
            if not text:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The uploaded file is empty or contains only whitespace.",
                )
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Corrupted file: unable to decode text file contents.",
            )

    # 2. PDF validation via pypdf (.pdf)
    elif extension == ".pdf":
        try:
            pdf = PdfReader(io.BytesIO(content))
            if len(pdf.pages) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The uploaded PDF contains no extractable text.",
                )
            has_text = any((page.extract_text() or "").strip() for page in pdf.pages)
            has_images = any(len(page.images) > 0 for page in pdf.pages)
        except (PdfReadError, EOFError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded PDF is invalid or unreadable.",
            )
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded PDF is invalid or unreadable.",
            )

        if not has_text and not has_images:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded PDF contains no extractable text.",
            )

    # 3. Word document validation (.docx)
    elif extension == ".docx":
        try:
            doc_file = io.BytesIO(content)
            if not zipfile.is_zipfile(doc_file):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Corrupted file: invalid or unreadable DOCX structure.",
                )
            doc = DocxDocument(doc_file)
            paragraphs_text = "".join(p.text for p in doc.paragraphs).strip()
            tables_text = "".join(
                cell.text
                for table in doc.tables
                for row in table.rows
                for cell in row.cells
            ).strip()
            total_text = paragraphs_text + tables_text
            has_shapes = len(doc.inline_shapes) > 0

            if not total_text and not has_shapes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The uploaded DOCX contains no extractable text.",
                )
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Corrupted file: unable to read DOCX document.",
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
        400: {"model": ErrorResponse, "description": "Empty or corrupted file"},
        409: {"model": ErrorResponse, "description": "Duplicate filename"},
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

    # 3. Stream content in chunks to measure size without buffering huge payloads
    size_bytes = 0
    buffer = io.BytesIO()
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
        buffer.write(chunk)

    content = buffer.getvalue()

    # 4. Content validation (catches empty, whitespace, blank & corrupt docs)
    validate_file_content(content=content, extension=extension)

    # 5. Duplicate filename check (returns 409 Conflict if file already exists)
    existing_document = session.exec(
        select(Document).where(Document.filename == filename)
    ).first()
    if existing_document:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A document with this filename already exists.",
        )

    # 6. Extract content type with fallback
    content_type = file.content_type or "application/octet-stream"

    # 7. Create and persist Document model to database
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
