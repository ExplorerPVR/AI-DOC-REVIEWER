"""FastAPI application entrypoint and routes for AI Document Assistant."""

import asyncio
import io
import json
import logging
import os
import re
import zipfile
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

import pymupdf
from docx import Document as DocxDocument
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from app.db import get_session
from app.extractor import extract_resume_data
from app.guardrails import (
    contains_system_prompt_leak,
    is_pii_harvesting_query,
    is_prompt_injection,
    redact_pii,
)
from app.ingestion import (
    ingest_document_pdf,
    sanitize_error_message,
    search_chunks,
)
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
    SearchResult,
)
from app.pricing import DEFAULT_MODELS
from app.rag import (
    NO_ANSWER_MESSAGE,
    RAG_SYSTEM_INSTRUCTIONS,
    build_rag_prompt,
    filter_relevant_chunks,
    format_sources_footer,
    is_refusal_answer,
    validate_inline_citations,
)

logger = logging.getLogger(__name__)

# Constants for document upload validation
MAX_FILE_SIZE_BYTES: int = int(
    os.getenv("MAX_FILE_SIZE_BYTES", str(10 * 1024 * 1024))
)  # 10 MB limit
CHUNK_SIZE_BYTES: int = 1024 * 1024  # 1 MB read chunk size
ALLOWED_EXTENSIONS: set[str] = {".pdf", ".txt", ".docx"}
UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "uploads"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup pre-warming and cleanup."""
    async def _prewarm_ollama():
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        model = os.getenv("OLLAMA_MODEL", "llama3")
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(
                    f"{base_url}/api/generate",
                    json={"model": model, "keep_alive": "24h"},
                )
                logger.info("Local Ollama model '%s' pre-warmed successfully.", model)
        except Exception as exc:
            logger.debug("Ollama pre-warm skipped: %s", exc)

    asyncio.create_task(_prewarm_ollama())
    yield


app = FastAPI(
    title="AI Document Assistant",
    description=(
        "FastAPI service for ingesting documents (.pdf, .txt, .docx), "
        "LLM streaming responses, and PostgreSQL persistence."
    ),
    version="0.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
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
        503: {"model": ErrorResponse, "description": "Database service unavailable"},
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

    # 7. Initial status: pending for background ingestion across all file types
    initial_status = DocumentStatus.PENDING

    # 8. Generate document UUID upfront to bind file path and database record
    doc_id = uuid4()

    # 9. Sanitize target file path and verify directory containment
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    raw_name = Path(filename).name
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw_name)
    if not safe_name or safe_name.replace(".", "") == "":
        safe_name = f"document{extension}"

    destination = (UPLOAD_DIR / f"{doc_id}_{safe_name}").resolve()
    if not destination.is_relative_to(UPLOAD_DIR.resolve()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename path traversal attempt.",
        )

    # 10. Write uploaded content atomically via temporary file
    temp_destination = (UPLOAD_DIR / f"tmp_{doc_id}_{safe_name}").resolve()
    try:
        temp_destination.write_bytes(content)
        temp_destination.replace(destination)
    except Exception as exc:
        if temp_destination.exists():
            temp_destination.unlink(missing_ok=True)
        if destination.exists():
            destination.unlink(missing_ok=True)
        logger.error("Failed saving uploaded file to disk: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed saving uploaded file: {exc}",
        ) from exc

    # 11. Persist Document to database with compensating cleanup on DB failure
    try:
        doc_record = Document(
            id=doc_id,
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            status=initial_status,
            file_path=str(destination),
        )
        session.add(doc_record)
        session.commit()
        session.refresh(doc_record)
    except Exception as exc:
        session.rollback()
        # Compensating cleanup: remove the file from filesystem if DB commit fails
        if destination.exists():
            destination.unlink(missing_ok=True)
        logger.error(
            "Database transaction failed for document %s, cleaned up file: %s",
            doc_id,
            exc,
        )
        raise

    # 12. Schedule background ingestion across all supported types (.pdf, .txt, .docx)
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
    responses={
        503: {"model": ErrorResponse, "description": "Database service unavailable"},
    },
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
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
        503: {"model": ErrorResponse, "description": "Database service unavailable"},
    },
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
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
        503: {"model": ErrorResponse, "description": "Database service unavailable"},
    },
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


@app.get(
    "/search",
    response_model=list[SearchResult],
    status_code=status.HTTP_200_OK,
    summary="Semantic Document Search",
    description=(
        "Search ingested document chunks by semantic cosine distance using "
        "pgvector's <=> operator, returning the top k most relevant matches."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Invalid search parameters"},
        503: {"model": ErrorResponse, "description": "Database service unavailable"},
    },
    tags=["Search"],
)
def search_documents(
    q: Annotated[
        str,
        Query(
            min_length=1,
            max_length=2000,
            description="Search query question to embed and search",
        ),
    ],
    session: Annotated[Session, Depends(get_session)],
    k: Annotated[
        int,
        Query(
            ge=1,
            le=100,
            description="Number of closest chunks to return (default 5)",
        ),
    ] = 5,
) -> list[SearchResult]:
    """Return the k closest document chunks to query q sorted by cosine distance."""
    if not q or not q.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Search query 'q' must not be empty or whitespace.",
        )

    return search_chunks(query=q.strip(), k=k, session=session)


@app.post(
    "/ask",
    summary="Ask Your Documents (RAG)",
    description=(
        "Answer questions strictly from ingested documents using semantic retrieval "
        "(pgvector) and grounded LLM generation with source file & page citations "
        "(page-level for PDF/paginated DOCX, document page 1 for unpaginated text). "
        "Supports chunked text/plain or Server-Sent Events (SSE) streaming with "
        "structured error communication. Returns \"I don't know\" when no relevant "
        "document context is found."
    ),
    responses={
        200: {
            "description": (
                "Chunked streaming response containing answer tokens and source "
                "citations, or SSE event stream with structured errors."
            ),
            "content": {
                "text/plain": {},
                "text/event-stream": {},
            },
        },
        400: {"model": ErrorResponse, "description": "Invalid question or provider"},
        503: {
            "model": ErrorResponse,
            "description": "Embedding or LLM provider unavailable or rate-limited",
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
    raw_request: Request,
) -> StreamingResponse:
    """Answer strictly from ingested documents with citations and 'I don't know'."""
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'question' field cannot be empty.",
        )

    provider = (
        request.provider or os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
    ).lower().strip()
    if provider not in DEFAULT_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported LLM provider '{provider}'. "
                f"Allowed providers: {', '.join(sorted(DEFAULT_MODELS.keys()))}"
            ),
        )

    # Determine whether client requested SSE or raw text streaming
    accept_header = raw_request.headers.get("accept", "").lower()
    use_sse = (
        request.stream_format == "sse"
        or "text/event-stream" in accept_header
    )
    validate_before_stream = (
        request.validate_before_stream
        or raw_request.headers.get("x-validate-before-stream", "").lower() == "true"
    )

    # 0. Security Guardrails: Prompt Injection & PII Harvesting Defense
    is_inj, inj_reason = is_prompt_injection(question)
    is_pii, pii_reason = is_pii_harvesting_query(question)
    if is_inj or is_pii:
        refusal_msg = f"I cannot answer this request. {inj_reason or pii_reason}"
        if use_sse:
            def sse_guardrail_refusal() -> Iterator[str]:
                payload = json.dumps({"text": refusal_msg, "guardrail_blocked": True})
                yield f"event: message\ndata: {payload}\n\n"
                yield "event: done\ndata: [DONE]\n\n"

            return StreamingResponse(
                sse_guardrail_refusal(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        return StreamingResponse(
            iter([refusal_msg]),
            media_type="text/plain",
        )

    # 1. Retrieve top-k relevant document chunks from pgvector
    try:
        retrieved_chunks = search_chunks(
            query=question,
            k=request.k,
            session=session,
        )
    except Exception as exc:
        logger.error("Failed to retrieve document chunks for /ask: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Document search service unavailable: {exc}",
        ) from exc

    # 2. Enforce relevance threshold: short-circuit 'I don't know' without calling LLM
    relevant_chunks = filter_relevant_chunks(retrieved_chunks)
    if not relevant_chunks:
        if use_sse:
            def sse_no_answer() -> Iterator[str]:
                payload = json.dumps({"text": NO_ANSWER_MESSAGE})
                yield f"event: message\ndata: {payload}\n\n"
                yield "event: done\ndata: [DONE]\n\n"

            return StreamingResponse(
                sse_no_answer(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        return StreamingResponse(
            iter([NO_ANSWER_MESSAGE]),
            media_type="text/plain",
        )

    # 3. Build strict grounded prompt with file & page metadata
    rag_user_prompt = build_rag_prompt(
        question=question,
        chunks=relevant_chunks,
        include_system_instructions=False,
    )

    try:
        gen = generate(
            prompt=rag_user_prompt,
            provider=provider,
            session=session,
            logged_prompt=question,
            system_instruction=RAG_SYSTEM_INSTRUCTIONS,
        )
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

    # Pre-validation mode: buffer & validate full answer before emitting any token
    if validate_before_stream:
        buffered_chunks: list[str] = []
        if first_chunk:
            buffered_chunks.append(first_chunk)
        try:
            for chunk in gen:
                if chunk:
                    buffered_chunks.append(chunk)
        except Exception as err:
            logger.error("Error during pre-validation generation: %s", err)
            safe_err = sanitize_error_message(err)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"LLM generation failed: {safe_err}",
            ) from err

        full_answer = "".join(buffered_chunks)

        # Refusal answers bypass citation check and return refusal text
        if not full_answer.strip() or is_refusal_answer(full_answer):
            def stream_prevalidated_refusal() -> Iterator[str]:
                out = full_answer if full_answer.strip() else NO_ANSWER_MESSAGE
                if use_sse:
                    yield f"event: message\ndata: {json.dumps({'text': out})}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                else:
                    yield out

            media_type = "text/event-stream" if use_sse else "text/plain"
            return StreamingResponse(
                stream_prevalidated_refusal(), media_type=media_type
            )

        # Check inline citations against retrieved document chunks
        if not validate_inline_citations(
            full_answer, relevant_chunks, require_citation=True
        ):
            logger.warning(
                "Suppressed ungrounded answer during pre-stream validation."
            )
            def stream_grounding_refusal() -> Iterator[str]:
                if use_sse:
                    err_payload = json.dumps({
                        "error": (
                            "Answer failed grounding validation: "
                            "citations could not be verified."
                        ),
                        "validation_failed": True,
                    })
                    yield f"event: error\ndata: {err_payload}\n\n"
                else:
                    yield NO_ANSWER_MESSAGE

            media_type = "text/event-stream" if use_sse else "text/plain"
            return StreamingResponse(
                stream_grounding_refusal(), media_type=media_type
            )

        # Check system prompt leakage
        if contains_system_prompt_leak(full_answer):
            logger.warning("Suppressed output containing leaked system prompt rules.")
            def stream_leak_refusal() -> Iterator[str]:
                if use_sse:
                    err_payload = json.dumps({
                        "error": "Response suppressed: security violation.",
                        "validation_failed": True,
                    })
                    yield f"event: error\ndata: {err_payload}\n\n"
                else:
                    yield NO_ANSWER_MESSAGE

            media_type = "text/event-stream" if use_sse else "text/plain"
            return StreamingResponse(stream_leak_refusal(), media_type=media_type)

        # Valid answer: stream verified chunks + sources footer
        sources_footer = format_sources_footer(relevant_chunks)
        def stream_prevalidated_success() -> Iterator[str]:
            for chunk in buffered_chunks:
                safe_chunk = redact_pii(chunk)
                if use_sse:
                    payload = json.dumps({"text": safe_chunk})
                    yield f"event: message\ndata: {payload}\n\n"
                else:
                    yield safe_chunk
            if sources_footer:
                if use_sse:
                    source_items = [
                        {"filename": c.filename, "page": c.page}
                        for c in relevant_chunks
                    ]
                    sources_payload = json.dumps(
                        {"sources": source_items, "footer": sources_footer}
                    )
                    yield f"event: sources\ndata: {sources_payload}\n\n"
                else:
                    yield sources_footer
            if use_sse:
                yield "event: done\ndata: [DONE]\n\n"

        media_type = "text/event-stream" if use_sse else "text/plain"
        headers = (
            {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
            if use_sse
            else None
        )
        return StreamingResponse(
            stream_prevalidated_success(),
            media_type=media_type,
            headers=headers,
        )

    # Real-time streaming mode: stream tokens as they arrive
    def stream_content() -> Iterator[str]:
        collected: list[str] = []
        stream_failed = False
        if first_chunk:
            safe_first = redact_pii(first_chunk)
            collected.append(first_chunk)
            if use_sse:
                yield f"event: message\ndata: {json.dumps({'text': safe_first})}\n\n"
            else:
                yield safe_first
        try:
            for chunk in gen:
                if chunk:
                    safe_chunk = redact_pii(chunk)
                    collected.append(chunk)
                    if use_sse:
                        payload = json.dumps({"text": safe_chunk})
                        yield f"event: message\ndata: {payload}\n\n"
                    else:
                        yield safe_chunk
        except Exception as err:
            logger.error("Error during streaming generation: %s", err)
            stream_failed = True
            safe_err = sanitize_error_message(err)
            if use_sse:
                yield (
                    f"event: error\n"
                    f"data: {json.dumps({'error': safe_err, 'partial': True})}\n\n"
                )
            else:
                yield f"\n\n[ERROR: Streaming generation interrupted: {safe_err}]\n"
            return

        if stream_failed:
            return

        full_answer = "".join(collected)
        if full_answer.strip() and not is_refusal_answer(full_answer):
            if contains_system_prompt_leak(full_answer):
                logger.warning(
                    "Suppressed output containing leaked system prompt rules."
                )
                if use_sse:
                    err_payload = json.dumps({
                        "error": "Response suppressed: security violation.",
                        "validation_failed": True,
                    })
                    yield f"event: error\ndata: {err_payload}\n\n"
                else:
                    yield (
                        "\n\n[VALIDATION_FAILED: "
                        "Response leaked system instructions.]\n"
                    )
                return

            if validate_inline_citations(
                full_answer, relevant_chunks, require_citation=True
            ):
                sources_footer = format_sources_footer(relevant_chunks)
                if sources_footer:
                    if use_sse:
                        source_items = [
                            {"filename": c.filename, "page": c.page}
                            for c in relevant_chunks
                        ]
                        sources_payload = json.dumps(
                            {"sources": source_items, "footer": sources_footer}
                        )
                        yield f"event: sources\ndata: {sources_payload}\n\n"
                    else:
                        yield sources_footer
            else:
                logger.warning(
                    "Suppressed sources footer: answer has ungrounded citations."
                )
                if use_sse:
                    err_payload = json.dumps({
                        "error": (
                            "Grounding validation failed: response contains "
                            "unverified citations or ungrounded claims."
                        ),
                        "validation_failed": True,
                    })
                    yield f"event: error\ndata: {err_payload}\n\n"
                else:
                    yield (
                        "\n\n[VALIDATION_FAILED: Grounding validation failed. "
                        "The response above contains unverified citations or "
                        "ungrounded claims.]\n"
                    )
                return

        if use_sse:
            yield "event: done\ndata: [DONE]\n\n"

    media_type = "text/event-stream" if use_sse else "text/plain"
    headers = (
        {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        if use_sse
        else None
    )
    return StreamingResponse(stream_content(), media_type=media_type, headers=headers)


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
    provider = (
        request.provider or os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
    ).lower().strip()
    return extract_resume_data(
        resume_text=content,
        provider=provider,
        session=session,
    )
