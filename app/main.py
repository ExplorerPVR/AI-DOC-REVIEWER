"""FastAPI application entrypoint and routes for AI Document Assistant."""

import io
import logging
import os
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated
from uuid import UUID

import pymupdf
from docx import Document as DocxDocument
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from app.db import get_session
from app.extractor import extract_resume_data
from app.ingestion import ingest_document_pdf
from app.llm import (
    LLMError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    generate,
)
from app.models import (
    AskRequest,
    Chunk,
    ChunkResponse,
    Document,
    DocumentResponse,
    DocumentStatus,
    ErrorResponse,
    ExtractRequest,
    HealthResponse,
    ResumeData,
)
from app.pricing import DEFAULT_MODELS

logger = logging.getLogger(__name__)

# Constants for document upload validation
MAX_FILE_SIZE_BYTES: int = int(
    os.getenv("MAX_FILE_SIZE_BYTES", str(10 * 1024 * 1024))
)  # 10 MB limit
CHUNK_SIZE_BYTES: int = 1024 * 1024  # 1 MB read chunk size
ALLOWED_EXTENSIONS: set[str] = {".pdf", ".txt", ".docx"}
UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "uploads"))

app = FastAPI(
    title="AI Document Assistant",
    description=(
        "FastAPI service for ingesting documents (.pdf, .txt, .docx), "
        "LLM streaming responses, and PostgreSQL persistence."
    ),
    version="0.3.0",
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

    # 2. PDF validation via PyMuPDF (.pdf)
    elif extension == ".pdf":
        try:
            doc = pymupdf.open(stream=content, filetype="pdf")
            if len(doc) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The uploaded PDF contains no extractable text.",
                )
            has_text = any(bool(page.get_text("text").strip()) for page in doc)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded PDF is invalid or unreadable.",
            )
        finally:
            if "doc" in locals() and not doc.is_closed:
                doc.close()

        if not has_text:
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
    background_tasks: BackgroundTasks,
) -> Document:
    """Validate uploaded document, save to disk, and trigger background ingestion."""
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

    # 7. Initial status: pending for PDF (background ingestion), ready for others
    initial_status = (
        DocumentStatus.PENDING if extension == ".pdf" else DocumentStatus.READY
    )

    # 8. Create and persist Document model to database
    doc_record = Document(
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
        status=initial_status,
    )
    session.add(doc_record)
    session.commit()
    session.refresh(doc_record)

    # 9. Save uploaded file to disk with path traversal sanitization
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    raw_name = Path(filename).name
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw_name)
    if not safe_name or safe_name.replace(".", "") == "":
        safe_name = f"document{extension}"

    destination = (UPLOAD_DIR / f"{doc_record.id}_{safe_name}").resolve()
    if not destination.is_relative_to(UPLOAD_DIR.resolve()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename path traversal attempt.",
        )
    destination.write_bytes(content)

    doc_record.file_path = str(destination)
    session.add(doc_record)
    session.commit()
    session.refresh(doc_record)

    # 10. Schedule background ingestion for PDFs
    if extension == ".pdf":
        background_tasks.add_task(
            ingest_document_pdf,
            document_id=doc_record.id,
            file_path=str(destination),
        )

    return doc_record


@app.get(
    "/documents",
    response_model=list[DocumentResponse],
    status_code=status.HTTP_200_OK,
    summary="List Documents",
    description="Retrieve metadata for all uploaded documents stored in PostgreSQL.",
    tags=["Documents"],
)
def list_documents(
    session: Annotated[Session, Depends(get_session)],
) -> list[Document]:
    """Return all stored document metadata records from PostgreSQL."""
    statement = select(Document).order_by(Document.uploaded_at)
    documents = session.exec(statement).all()
    return list(documents)


@app.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Document Details",
    description="Retrieve metadata and ingestion status for a single document.",
    tags=["Documents"],
)
def get_document(
    document_id: UUID,
    session: Annotated[Session, Depends(get_session)],
) -> Document:
    """Retrieve document metadata and current ingestion status by ID."""
    doc = session.get(Document, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    return doc


@app.get(
    "/documents/{document_id}/chunks",
    response_model=list[ChunkResponse],
    status_code=status.HTTP_200_OK,
    summary="Get Document Chunks",
    description="Retrieve all text chunks generated for an ingested document.",
    tags=["Documents"],
)
def get_document_chunks(
    document_id: UUID,
    session: Annotated[Session, Depends(get_session)],
) -> list[Chunk]:
    """Retrieve generated text chunks for an ingested document."""
    doc = session.get(Document, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    statement = (
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
    )
    chunks = session.exec(statement).all()
    return list(chunks)


@app.post(
    "/ask",
    summary="Ask an LLM",
    description=(
        "Stream completion answers from configured LLM providers "
        "(gemini, groq, ollama) with automatic retries and cost logging."
    ),
    responses={
        200: {
            "description": "Streamed text response chunks from the LLM provider",
            "content": {"text/plain": {}},
        },
        400: {"model": ErrorResponse, "description": "Invalid question or provider"},
        503: {
            "model": ErrorResponse,
            "description": "LLM provider unavailable or rate-limited",
        },
        504: {
            "model": ErrorResponse,
            "description": "LLM provider request timed out",
        },
    },
    tags=["LLM"],
)
def ask_llm(
    request: AskRequest,
    session: Annotated[Session, Depends(get_session)],
) -> StreamingResponse:
    """Stream response from selected LLM provider and persist call metrics."""
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'question' field cannot be empty.",
        )

    provider = (request.provider or "gemini").lower().strip()
    if provider not in DEFAULT_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported LLM provider '{provider}'. "
                f"Allowed providers: {', '.join(sorted(DEFAULT_MODELS.keys()))}"
            ),
        )

    try:
        gen = generate(prompt=question, provider=provider, session=session)
        # Trigger connection & retry loop before sending 200 header
        first_chunk = next(gen, None)
    except LLMTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"LLM provider request timed out: {exc.message}",
        ) from exc
    except (LLMRateLimitError, LLMServerError, LLMError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM provider service unavailable: {exc.message}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unexpected error communicating with LLM: {exc}",
        ) from exc

    def stream_content() -> Iterator[str]:
        if first_chunk:
            yield first_chunk
        try:
            for chunk in gen:
                if chunk:
                    yield chunk
        except Exception as err:
            logger.error("Error during streaming generation: %s", err)

    return StreamingResponse(stream_content(), media_type="text/plain")


@app.post(
    "/extract",
    response_model=ResumeData,
    status_code=status.HTTP_200_OK,
    summary="Extract Structured Resume Data",
    description=(
        "Extract candidate biographical details and skills into a structured JSON "
        "ResumeData schema using provider structured outputs, with automatic retry "
        "on schema validation failures and prompt injection defense."
    ),
    responses={
        200: {
            "model": ResumeData,
            "description": "Validated structured resume information",
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid input or empty resume text",
        },
        422: {
            "model": ErrorResponse,
            "description": "Schema validation failed after retry",
        },
        503: {
            "model": ErrorResponse,
            "description": "LLM provider unavailable or rate-limited",
        },
        504: {
            "model": ErrorResponse,
            "description": "LLM provider request timed out",
        },
    },
    tags=["Extraction"],
)
def extract_resume(
    request: ExtractRequest,
    session: Annotated[Session, Depends(get_session)],
) -> ResumeData:
    """Extract structured ResumeData from provided resume text."""
    content = request.text.strip()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The resume 'text' field cannot be empty.",
        )
    return extract_resume_data(
        resume_text=content,
        provider=request.provider,
        session=session,
    )
