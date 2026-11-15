"""Background PDF ingestion, page extraction, chunking, and pgvector embeddings."""

import logging
import os
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

import httpx
import pymupdf
from sqlmodel import Session

import app.db as db_module
from app.models import Chunk, Document, DocumentStatus

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 768
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150


def read_pdf_pages(file_path: str | Path) -> list[tuple[int, str]]:
    """Extract 1-based page number and text from each page using PyMuPDF.

    Raises:
        ValueError: If file is missing, cannot be parsed, contains 0 pages,
                    or has no extractable text across all pages.
    """
    path = Path(file_path)
    if not path.is_file():
        raise ValueError(f"File not found on disk: {file_path}")

    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        raise ValueError(f"Corrupted or unreadable PDF: {exc}") from exc

    try:
        if len(doc) == 0:
            raise ValueError("The uploaded PDF contains 0 pages.")

        pages_content: list[tuple[int, str]] = []
        total_chars = 0

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_text = page.get_text("text").strip()
            page_num = page_idx + 1
            if page_text:
                pages_content.append((page_num, page_text))
                total_chars += len(page_text)

        if total_chars == 0:
            raise ValueError("The uploaded PDF contains no extractable text.")

        return pages_content
    finally:
        doc.close()


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks of approximately 500 to 1,000 characters."""
    cleaned = text.strip()
    if not cleaned:
        return []

    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    text_len = len(cleaned)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Look for a natural split boundary in the tail window of the chunk
        if end < text_len:
            search_start = max(start + chunk_size - 200, start)
            best_break = -1
            for sep in ["\n\n", "\n", ". ", "; ", ", ", " "]:
                pos = cleaned.rfind(sep, search_start, end)
                if pos != -1:
                    best_break = pos + len(sep)
                    break
            if best_break != -1 and best_break > start + 300:
                end = best_break

        chunk_str = cleaned[start:end].strip()
        if chunk_str:
            chunks.append(chunk_str)

        if end >= text_len:
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def generate_embedding(
    text: str,
    provider: str | None = None,
    timeout: float = 30.0,
) -> list[float]:
    """Generate a 768-dimensional dense vector embedding for the given text.

    Supports Gemini (text-embedding-004 / gemini-embedding-001) and Ollama
    (nomic-embed-text).
    """
    chosen_provider = (
        provider or os.getenv("EMBEDDING_PROVIDER") or "gemini"
    ).lower()

    if chosen_provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not configured.")

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        response = client.models.embed_content(
            model=model,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSION
            ),
        )
        if not response.embeddings or not response.embeddings[0].values:
            raise ValueError("Gemini returned empty embedding vector.")
        vector = list(response.embeddings[0].values)
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSION}-dim vector, got {len(vector)}."
            )
        return vector

    elif chosen_provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        url = f"{base_url}/api/embeddings"
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json={"model": model, "prompt": text})
            if resp.status_code != 200:
                raise ValueError(
                    f"Ollama embedding request failed (HTTP {resp.status_code}): "
                    f"{resp.text}"
                )
            data = resp.json()
            vector = data.get("embedding")
            if not vector or not isinstance(vector, list):
                raise ValueError("Ollama returned invalid embedding format.")
            if len(vector) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Expected {EMBEDDING_DIMENSION}-dim vector, got {len(vector)}."
                )
            return [float(x) for x in vector]

    else:
        raise ValueError(
            f"Unsupported embedding provider '{chosen_provider}'. "
            f"Allowed providers: 'gemini', 'ollama'."
        )


def ingest_document_pdf(
    document_id: UUID,
    file_path: str,
    session_factory: Callable[[], Session] | None = None,
    embedding_provider: str | None = None,
) -> None:
    """Background ingestion worker: extracts PDF pages, chunks, embeds, and saves."""
    get_db = session_factory or (lambda: Session(db_module.engine))

    with get_db() as session:
        doc = session.get(Document, document_id)
        if not doc:
            logger.warning(
                "Document %s not found in database during background ingestion.",
                document_id,
            )
            return

        doc.status = DocumentStatus.PROCESSING
        session.add(doc)
        session.commit()
        logger.info(
            "Started background ingestion for document %s (%s).",
            document_id,
            doc.filename,
        )

        try:
            # 1. Read PDF page by page
            pages = read_pdf_pages(file_path)

            # 2. Split each page's text into chunks of ~500-1000 characters
            chunk_records: list[Chunk] = []
            chunk_idx = 0

            for page_num, page_text in pages:
                page_chunks = chunk_text(page_text)
                for chunk_str in page_chunks:
                    # 3. Create 768-dim dense embedding
                    vector = generate_embedding(
                        text=chunk_str,
                        provider=embedding_provider,
                    )
                    chunk_obj = Chunk(
                        document_id=doc.id,
                        filename=doc.filename,
                        page_number=page_num,
                        chunk_index=chunk_idx,
                        text=chunk_str,
                        embedding=vector,
                    )
                    chunk_records.append(chunk_obj)
                    chunk_idx += 1

            if not chunk_records:
                raise ValueError("No extractable chunks found in PDF document.")

            # 4. Save all chunks to database
            for chk in chunk_records:
                session.add(chk)

            # 5. Mark document as ready
            doc.status = DocumentStatus.READY
            doc.error_message = None
            session.add(doc)
            session.commit()
            logger.info(
                "Successfully ingested document %s (%s): %d chunks stored in pgvector.",
                document_id,
                doc.filename,
                len(chunk_records),
            )

        except Exception as exc:
            logger.error(
                "Failed to ingest document %s (%s): %s",
                document_id,
                doc.filename,
                exc,
                exc_info=True,
            )
            session.rollback()
            refreshed_doc = session.get(Document, document_id)
            if refreshed_doc:
                refreshed_doc.status = DocumentStatus.FAILED
                refreshed_doc.error_message = str(exc)
                session.add(refreshed_doc)
                session.commit()
