"""Background PDF ingestion, page extraction, chunking, and pgvector embeddings."""

import logging
import math
import os
import re
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

import httpx
import pymupdf
from sqlmodel import Session, delete, select

import app.db as db_module
from app.llm import retry
from app.models import Chunk, Document, DocumentStatus, SearchResult

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 768
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_BATCH_SIZE = 16

# Reusable client instances to prevent socket exhaustion and per-chunk setup overhead
_gemini_client = None
_ollama_client: httpx.Client | None = None


def get_gemini_client(api_key: str):
    """Retrieve or initialize singleton Google GenAI client."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def get_ollama_client(timeout: float = 30.0) -> httpx.Client:
    """Retrieve or initialize singleton HTTP client for Ollama."""
    global _ollama_client
    if _ollama_client is None or _ollama_client.is_closed:
        _ollama_client = httpx.Client(timeout=timeout)
    return _ollama_client


def sanitize_error_message(exc: Exception) -> str:
    """Sanitize error message to prevent leaking internal URLs, keys, or details."""
    raw = str(exc)
    # Redact potential API keys or tokens (alphanumeric sequences >= 20 chars)
    sanitized = re.sub(
        r"(?:key|token|secret|api)[=:][\s]*([^\s&,]+)",
        r"[REDACTED]",
        raw,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(r"https?://[^\s]+", "[REDACTED_URL]", sanitized)
    lines = [line.strip() for line in sanitized.splitlines() if line.strip()]
    first_meaningful_line = lines[-1] if lines else "Ingestion processing error"
    if len(first_meaningful_line) > 150:
        first_meaningful_line = first_meaningful_line[:147] + "..."
    return first_meaningful_line


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


def read_txt_pages(file_path: str | Path) -> list[tuple[int, str]]:
    """Extract page number (1) and text from a plain text (.txt) file.

    Raises:
        ValueError: If file is missing, cannot be decoded, or has no text.
    """
    path = Path(file_path)
    if not path.is_file():
        raise ValueError(f"File not found on disk: {file_path}")

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="latin-1")
        except Exception as exc:
            raise ValueError(f"Corrupted or unreadable text file: {exc}") from exc

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("The uploaded text file contains no extractable text.")

    return [(1, cleaned)]


def read_docx_pages(file_path: str | Path) -> list[tuple[int, str]]:
    """Extract 1-based page numbers and text from a Word document (.docx).

    Preserves page boundaries by detecting explicit hard page breaks
    (<w:br w:type="page"/>) and soft break hints (<w:lastRenderedPageBreak/>)
    rendered by word processors. For reflowable documents without explicit
    page breaks, content is indexed under page 1.

    Raises:
        ValueError: If file is missing, corrupted, or has no extractable text.
    """
    from docx import Document as DocxDocument
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    path = Path(file_path)
    if not path.is_file():
        raise ValueError(f"File not found on disk: {file_path}")

    try:
        doc = DocxDocument(str(path))
    except Exception as exc:
        raise ValueError(f"Corrupted or unreadable DOCX: {exc}") from exc

    pages_dict: dict[int, list[str]] = {}
    current_page = 1

    try:
        for el in doc.element.body:
            tag = el.tag.split("}")[-1]
            if tag == "p":
                p = Paragraph(el, doc)
                p_text_parts: list[str] = []
                for run in p.runs:
                    run_breaks = run._r.xpath(
                        ".//w:br[@w:type='page'] | .//w:lastRenderedPageBreak"
                    )
                    if run_breaks:
                        if p_text_parts:
                            flush_text = "".join(p_text_parts).strip()
                            if flush_text:
                                pages_dict.setdefault(current_page, []).append(
                                    flush_text
                                )
                            p_text_parts = []
                        current_page += len(run_breaks)
                    p_text_parts.append(run.text)

                if not p.runs:
                    breaks = el.xpath(
                        ".//w:br[@w:type='page'] | .//w:lastRenderedPageBreak"
                    )
                    if breaks:
                        current_page += len(breaks)
                elif p_text_parts:
                    flush_text = "".join(p_text_parts).strip()
                    if flush_text:
                        pages_dict.setdefault(current_page, []).append(flush_text)

            elif tag == "tbl":
                tbl = Table(el, doc)
                for row in tbl.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        pages_dict.setdefault(current_page, []).append(
                            " | ".join(cells)
                        )

        result: list[tuple[int, str]] = []
        for page_num in sorted(pages_dict.keys()):
            page_text = "\n\n".join(pages_dict[page_num]).strip()
            if page_text:
                result.append((page_num, page_text))

        if not result:
            raise ValueError("The uploaded DOCX contains no extractable text.")

        return result
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Corrupted or unreadable DOCX: {exc}") from exc


def read_document_pages(file_path: str | Path) -> list[tuple[int, str]]:
    """Extract page numbers and text across supported formats (.pdf, .txt, .docx)."""
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf_pages(path)
    if ext == ".txt":
        return read_txt_pages(path)
    if ext == ".docx":
        return read_docx_pages(path)
    raise ValueError(f"Unsupported document format: '{ext}'")


def _dehyphenate_match(m: re.Match) -> str:
    """Smartly reassemble hyphenated words split across PDF line breaks."""
    w1, w2 = m.group(1), m.group(2)
    # Common compound prefixes that naturally retain their hyphen in English
    compound_prefixes = {
        "self",
        "multi",
        "cross",
        "semi",
        "pre",
        "post",
        "joint",
        "large",
        "hand",
        "zero",
        "few",
    }
    if w1.lower() in compound_prefixes:
        return f"{w1}-{w2}"
    return f"{w1}{w2}"


def clean_pdf_text(text: str) -> str:
    """Normalize PDF text by removing hyphenation and unwrapping soft linebreaks.

    - Resolves hyphenated words split across lines (e.g. 'joint-\\nembedding').
    - Converts citation markers (e.g. '\\n[14]') into distinct block boundaries.
    - Unwraps soft visual linebreaks within sentences into single spaces.
    - Preserves true structural double newlines ('\\n\\n').
    """
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    # Smartly rejoin hyphenated words broken across physical line breaks
    cleaned = re.sub(r"(\w+)-\s*\n\s*(\w+)", _dehyphenate_match, cleaned)
    # Ensure citation anchors like "\n[14]" act as block paragraphs
    cleaned = re.sub(r"(?<!\n)\n(\[\d+\])", r"\n\n\1", cleaned)
    # Replace soft single newlines within sentences with a space
    cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)
    # Collapse multiple consecutive horizontal whitespace characters
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks snapping cleanly to sentences or words.

    Ensures chunks are between ~500 to 1,000 characters and avoids cutting
    sentences, citations, or words in half.
    """
    cleaned = clean_pdf_text(text)
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
            search_start = max(start + int(chunk_size * 0.6), start)
            best_break = -1
            # Prioritize paragraph/citation breaks and sentences over newlines
            for sep in ["\n\n", ". ", "? ", "! ", "; ", "\n"]:
                pos = cleaned.rfind(sep, search_start, end)
                if pos != -1:
                    best_break = pos + len(sep)
                    break
            if best_break != -1 and best_break > start + int(chunk_size * 0.4):
                end = best_break
            else:
                # Snap backward to nearest whitespace so chunk never ends mid-word
                space_pos = cleaned.rfind(" ", search_start, end)
                if space_pos == -1:
                    space_pos = cleaned.rfind(" ", start + 1, end)
                if space_pos != -1 and space_pos > start:
                    end = space_pos

        chunk_str = cleaned[start:end].strip()
        if chunk_str:
            chunks.append(chunk_str)

        if end >= text_len:
            break

        # Calculate nominal overlap start and snap to sentence, citation, or word
        raw_start = max(end - chunk_overlap, start + 1)
        if raw_start < text_len:
            boundary_pos = -1
            # Search backward first (up to chunk_overlap chars) to capture complete unit
            for boundary in ["\n\n[", "\n\n", ". "]:
                p = cleaned.rfind(
                    boundary,
                    max(start + 1, raw_start - chunk_overlap),
                    min(raw_start + 60, end),
                )
                if p != -1:
                    boundary_pos = (
                        p + 2 if boundary.startswith("\n\n") else p + len(boundary)
                    )
                    break

            if boundary_pos != -1 and start < boundary_pos < end:
                start = boundary_pos
            else:
                # Fallback: snap cleanly to start of a whole word
                if raw_start > start and cleaned[raw_start - 1] not in " \n\t":
                    prev_space = cleaned.rfind(" ", start, raw_start)
                    if prev_space != -1:
                        start = prev_space + 1
                    else:
                        next_space = cleaned.find(" ", raw_start, end)
                        start = next_space + 1 if next_space != -1 else raw_start
                else:
                    start = raw_start
        else:
            start = raw_start

    return chunks


@retry(max_retries=3, delay=1.0, backoff=2.0)
def _embed_batch_with_retry(
    texts: list[str],
    provider: str,
    timeout: float = 30.0,
) -> list[list[float]]:
    """Execute batched embedding API call with exponential retry logic."""
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not configured.")

        from google.genai import types

        client = get_gemini_client(api_key)
        model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        response = client.models.embed_content(
            model=model,
            contents=texts,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSION
            ),
        )
        if not response.embeddings or len(response.embeddings) != len(texts):
            count = len(response.embeddings) if response.embeddings else 0
            raise ValueError(
                f"Gemini returned {count} embeddings for {len(texts)} texts."
            )
        results: list[list[float]] = []
        for emb in response.embeddings:
            vector = list(emb.values)
            if len(vector) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Expected {EMBEDDING_DIMENSION}-dim vector, got {len(vector)}."
                )
            results.append(vector)
        return results

    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        client = get_ollama_client(timeout=timeout)

        # Attempt batched embed API first
        embed_url = f"{base_url}/api/embed"
        try:
            resp = client.post(embed_url, json={"model": model, "input": texts})
            if resp.status_code == 200:
                data = resp.json()
                embeddings = data.get("embeddings", [])
                if len(embeddings) == len(texts):
                    return [[float(x) for x in emb] for emb in embeddings]
        except Exception:
            logger.debug(
                "Ollama /api/embed unavailable, falling back to /api/embeddings"
            )

        # Fallback to sequential /api/embeddings with reused client
        results = []
        url = f"{base_url}/api/embeddings"
        for text in texts:
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
            results.append([float(x) for x in vector])
        return results

    else:
        raise ValueError(
            f"Unsupported embedding provider '{provider}'. "
            f"Allowed providers: 'gemini', 'ollama'."
        )


def generate_embeddings_batch(
    texts: list[str],
    provider: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: float = 30.0,
) -> list[list[float]]:
    """Generate 768-dimensional dense vector embeddings in batches with retries."""
    if not texts:
        return []

    chosen_provider = (
        provider or os.getenv("EMBEDDING_PROVIDER") or "gemini"
    ).lower()

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = _embed_batch_with_retry(
            texts=batch,
            provider=chosen_provider,
            timeout=timeout,
        )
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def generate_embedding(
    text: str,
    provider: str | None = None,
    timeout: float = 30.0,
) -> list[float]:
    """Generate single 768-dimensional dense vector embedding (wrapper for batch)."""
    results = generate_embeddings_batch(
        texts=[text],
        provider=provider,
        batch_size=1,
        timeout=timeout,
    )
    return results[0]


def ingest_document_pdf(
    document_id: UUID,
    file_path: str,
    session_factory: Callable[[], Session] | None = None,
    embedding_provider: str | None = None,
) -> None:
    """Background ingestion worker.

    Separates database session ownership from external HTTP embedding calls to
    prevent database connection pool starvation.
    """
    get_db = session_factory or (lambda: Session(db_module.engine))

    # Phase 1: Mark status as PROCESSING in short DB tx & release connection
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
        filename = doc.filename

    logger.info(
        "Started background ingestion for document %s (%s).",
        document_id,
        filename,
    )

    # Phase 2: Compute & Network (Document parsing, chunking, external API embeddings)
    # NO DB CONNECTION IS HELD DURING THIS PHASE!
    error: Exception | None = None
    chunk_records_data: list[tuple[int, int, str, list[float]]] = []

    try:
        pages = read_document_pages(file_path)
        all_chunks: list[tuple[int, int, str]] = []
        chunk_idx = 0
        for page_num, page_text in pages:
            page_chunks = chunk_text(page_text)
            for chunk_str in page_chunks:
                all_chunks.append((page_num, chunk_idx, chunk_str))
                chunk_idx += 1

        if not all_chunks:
            raise ValueError("No extractable chunks found in document.")

        # Batch embed all chunks across external network
        chunk_texts = [item[2] for item in all_chunks]
        embeddings = generate_embeddings_batch(
            texts=chunk_texts,
            provider=embedding_provider,
        )

        for (page_num, chk_idx, chunk_str), vector in zip(all_chunks, embeddings):
            chunk_records_data.append((page_num, chk_idx, chunk_str, vector))

    except Exception as exc:
        logger.error(
            "Failed to ingest document %s (%s): %s",
            document_id,
            filename,
            exc,
            exc_info=True,
        )
        error = exc

    # Phase 3: Update DB in a second short transaction
    with get_db() as session:
        doc = session.get(Document, document_id)
        if not doc:
            return

        if error is not None:
            doc.status = DocumentStatus.FAILED
            doc.error_message = sanitize_error_message(error)
            session.add(doc)
            session.commit()
        else:
            session.exec(delete(Chunk).where(Chunk.document_id == doc.id))
            for page_num, chk_idx, chunk_str, vector in chunk_records_data:
                chk = Chunk(
                    document_id=doc.id,
                    filename=filename,
                    page_number=page_num,
                    chunk_index=chk_idx,
                    text=chunk_str,
                    embedding=vector,
                )
                session.add(chk)

            doc.status = DocumentStatus.READY
            doc.error_message = None
            session.add(doc)
            session.commit()
            logger.info(
                "Successfully ingested document %s (%s): %d chunks stored in pgvector.",
                document_id,
                filename,
                len(chunk_records_data),
            )


# Alias for generalized multi-format document ingestion (.pdf, .txt, .docx)
ingest_document = ingest_document_pdf


def search_chunks(
    query: str,
    k: int = 5,
    session: Session | None = None,
    embedding_provider: str | None = None,
) -> list[SearchResult]:
    """Search ingested document chunks by semantic cosine distance using pgvector.

    1. If no documents are in READY status, returns an empty list [] immediately
       without invoking the embedding model.
    2. Embeds query using the configured embedding model.
    3. Searches Chunk table using pgvector cosine distance (<=> operator),
       joined to Document where status is READY.
    4. Returns up to k closest chunks sorted by cosine distance ascending.
    """
    if not query or not query.strip():
        return []

    own_session = session is None

    # Short-circuit before calling embedding API if no READY documents exist
    ready_stmt = (
        select(Document.id)
        .where(Document.status == DocumentStatus.READY)
        .limit(1)
    )
    if own_session:
        with Session(db_module.engine) as check_session:
            has_ready_doc = check_session.exec(ready_stmt).first() is not None
    else:
        has_ready_doc = session.exec(ready_stmt).first() is not None

    if not has_ready_doc:
        return []

    query_vector = generate_embedding(query.strip(), provider=embedding_provider)

    s = Session(db_module.engine) if own_session else session

    try:
        dialect_name = s.get_bind().dialect.name
        if dialect_name == "postgresql":
            # Native pgvector cosine distance (<=> operator)
            dist_expr = Chunk.embedding.cosine_distance(query_vector)
            stmt = (
                select(
                    Chunk.text,
                    Chunk.filename,
                    Chunk.page_number,
                    dist_expr.label("distance"),
                )
                .join(Document, Chunk.document_id == Document.id)
                .where(Document.status == DocumentStatus.READY)
                .order_by(dist_expr)
                .limit(k)
            )
            rows = s.exec(stmt).all()
            return [
                SearchResult(
                    text=row[0],
                    filename=row[1],
                    page=row[2],
                    score=round(max(0.0, 1.0 - float(row[3])), 4),
                )
                for row in rows
            ]
        else:
            # Fallback for SQLite in unit tests
            stmt = (
                select(Chunk)
                .join(Document, Chunk.document_id == Document.id)
                .where(Document.status == DocumentStatus.READY)
            )
            chunks = s.exec(stmt).all()
            scored: list[tuple[Chunk, float, float]] = []
            for chk in chunks:
                dot = sum(a * b for a, b in zip(chk.embedding, query_vector))
                norm1 = math.sqrt(sum(a * a for a in chk.embedding))
                norm2 = math.sqrt(sum(b * b for b in query_vector))
                sim = (dot / (norm1 * norm2)) if (norm1 and norm2) else 0.0
                dist = max(0.0, min(2.0, 1.0 - sim))
                scored.append((chk, dist, max(0.0, sim)))

            scored.sort(key=lambda item: item[1])
            top_k = scored[:k]
            return [
                SearchResult(
                    text=chk.text,
                    filename=chk.filename,
                    page=chk.page_number,
                    score=round(sim_val, 4),
                )
                for chk, dist, sim_val in top_k
            ]
    finally:
        if own_session:
            s.close()
