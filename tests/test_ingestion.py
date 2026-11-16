"""Unit and integration tests for background PDF ingestion and pgvector chunking."""

import uuid
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine, select
from sqlmodel.pool import StaticPool

import app.db as db_module
import app.ingestion as ingestion_module
from app.db import get_session
from app.ingestion import (
    EMBEDDING_DIMENSION,
    chunk_text,
    generate_embedding,
    generate_embeddings_batch,
    ingest_document_pdf,
    read_pdf_pages,
    sanitize_error_message,
)
from app.main import app
from app.models import Chunk, Document, DocumentStatus

# In-memory SQLite engine for testing
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


def create_sample_multipage_pdf(file_path: Path) -> Path:
    """Create a 2-page PDF file using PyMuPDF with text on each page."""
    import pymupdf

    doc = pymupdf.open()
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text(
        (50, 72),
        "Page 1: Deep learning architectures rely on multi-head attention "
        "mechanisms and feed-forward layers to capture token interactions. "
        "PostgreSQL with pgvector allows scalable retrieval of dense embeddings.",
    )
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text(
        (50, 72),
        "Page 2: Vector embeddings encode semantic nuances in vector space. "
        "Cosine similarity provides fast nearest-neighbor indexing.",
    )
    doc.save(str(file_path))
    doc.close()
    return file_path


def create_blank_pdf_file(file_path: Path) -> Path:
    """Create a 1-page PDF file with zero extractable text."""
    import pymupdf

    doc = pymupdf.open()
    doc.new_page(width=595, height=842)
    doc.save(str(file_path))
    doc.close()
    return file_path


@pytest.fixture(name="session")
def session_fixture() -> Generator[Session, None, None]:
    """Provide a fresh isolated database session."""
    SQLModel.metadata.create_all(test_engine)
    with Session(test_engine) as session:
        yield session
    SQLModel.metadata.drop_all(test_engine)


@pytest.fixture(name="client")
def client_fixture(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient, None, None]:
    """Provide TestClient with database session dependency overridden."""

    def get_session_override() -> Generator[Session, None, None]:
        yield session

    app.dependency_overrides[get_session] = get_session_override
    orig_engine = db_module.engine
    db_module.engine = test_engine

    # Mock embedding calls to avoid external network dependencies during tests
    monkeypatch.setattr(
        ingestion_module,
        "generate_embedding",
        lambda *args, **kwargs: [0.05] * EMBEDDING_DIMENSION,
    )
    monkeypatch.setattr(
        ingestion_module,
        "generate_embeddings_batch",
        lambda texts, *args, **kwargs: [[0.05] * EMBEDDING_DIMENSION for _ in texts],
    )

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()
    db_module.engine = orig_engine


def test_chunk_text_small_and_empty():
    """Verify chunk_text handles small text and empty strings."""
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []

    short = "Hello world from document assistant."
    chunks = chunk_text(short, chunk_size=800, chunk_overlap=150)
    assert len(chunks) == 1
    assert chunks[0] == short


def test_chunk_text_splits_with_overlap():
    """Verify long text splits into chunks within character limits with overlap."""
    sentence = (
        "Distributed consensus algorithms like Raft and Paxos ensure state machine "
        "replication across heterogeneous nodes. "
    )
    long_text = sentence * 25  # ~2750 characters
    chunks = chunk_text(long_text, chunk_size=800, chunk_overlap=150)

    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk) <= 850
        assert len(chunk) >= 400


def test_chunk_text_overlap_snaps_to_words():
    """Verify chunk overlap does not break words in half."""
    words = [f"token{i}" for i in range(250)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=80)
    for chk in chunks:
        first_word = chk.split()[0]
        assert first_word in words


def test_sanitize_error_message():
    """Verify error sanitization redacts URLs and API tokens."""
    exc = RuntimeError(
        "Failed connecting to https://api.groq.com/v1 with "
        "api_key=sk-1234567890abcdef1234567890"
    )
    sanitized = sanitize_error_message(exc)
    assert "[REDACTED]" in sanitized
    assert "[REDACTED_URL]" in sanitized
    assert "sk-1234567890" not in sanitized


def test_read_pdf_pages_multipage(tmp_path: Path):
    """Verify read_pdf_pages extracts text page by page with 1-based indexing."""
    pdf_file = tmp_path / "sample.pdf"
    create_sample_multipage_pdf(pdf_file)

    pages = read_pdf_pages(pdf_file)
    assert len(pages) == 2
    assert pages[0][0] == 1
    assert "Page 1:" in pages[0][1]
    assert pages[1][0] == 2
    assert "Page 2:" in pages[1][1]


def test_read_pdf_pages_missing_file():
    """Verify read_pdf_pages raises ValueError for missing file."""
    with pytest.raises(ValueError, match="File not found"):
        read_pdf_pages("non_existent_document.pdf")


def test_read_pdf_pages_corrupted_file(tmp_path: Path):
    """Verify read_pdf_pages raises ValueError for corrupted file."""
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_bytes(b"%PDF-1.4 garbage unparseable content")

    with pytest.raises(ValueError, match="Corrupted or unreadable PDF"):
        read_pdf_pages(corrupt_file)


def test_read_pdf_pages_empty_content(tmp_path: Path):
    """Verify read_pdf_pages raises ValueError when PDF has no text."""
    blank_file = tmp_path / "blank.pdf"
    create_blank_pdf_file(blank_file)

    with pytest.raises(ValueError, match="no extractable text"):
        read_pdf_pages(blank_file)


def test_generate_embedding_gemini(monkeypatch: pytest.MonkeyPatch):
    """Verify generate_embedding calls Gemini client and returns 768-dim list."""
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")

    mock_client = MagicMock()
    mock_values = [0.123] * 768
    mock_embedding_obj = MagicMock(values=mock_values)
    mock_resp = MagicMock(embeddings=[mock_embedding_obj])
    mock_client.models.embed_content.return_value = mock_resp

    monkeypatch.setattr(
        ingestion_module, "get_gemini_client", lambda api_key: mock_client
    )

    result = generate_embedding("sample text", provider="gemini")
    assert len(result) == 768
    assert result[0] == pytest.approx(0.123)


def test_generate_embedding_ollama(monkeypatch: pytest.MonkeyPatch):
    """Verify generate_embedding calls Ollama endpoint and returns 768-dim list."""

    class MockClient:
        is_closed = False

        def post(self, url, *args, **kwargs):
            class MockResponse:
                status_code = 200

                def json(self):
                    return {"embeddings": [[0.456] * 768]}

            return MockResponse()

    monkeypatch.setattr(
        ingestion_module, "get_ollama_client", lambda *args, **kwargs: MockClient()
    )

    result = generate_embeddings_batch(["sample text"], provider="ollama")
    assert len(result) == 1
    assert len(result[0]) == 768
    assert result[0][0] == pytest.approx(0.456)


def test_generate_embedding_unsupported_provider():
    """Verify generate_embedding raises ValueError for invalid provider."""
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        generate_embedding("text", provider="unknown_ai")


def test_background_ingestion_success(
    tmp_path: Path, session: Session, monkeypatch: pytest.MonkeyPatch
):
    """Verify background ingestion parses PDF, chunks, embeds, and marks ready."""
    pdf_file = tmp_path / "research_paper.pdf"
    create_sample_multipage_pdf(pdf_file)

    doc = Document(
        filename="research_paper.pdf",
        content_type="application/pdf",
        size_bytes=len(pdf_file.read_bytes()),
        status=DocumentStatus.PENDING,
        file_path=str(pdf_file),
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    monkeypatch.setattr(
        ingestion_module,
        "generate_embeddings_batch",
        lambda texts, *args, **kwargs: [[0.01] * EMBEDDING_DIMENSION for _ in texts],
    )

    ingest_document_pdf(
        document_id=doc.id,
        file_path=str(pdf_file),
        session_factory=lambda: Session(test_engine),
    )

    # Verify document status updated to READY
    with Session(test_engine) as check_session:
        updated_doc = check_session.get(Document, doc.id)
        assert updated_doc is not None
        assert updated_doc.status == DocumentStatus.READY
        assert updated_doc.error_message is None

        # Verify chunks created in DB
        chunks = check_session.exec(
            select(Chunk)
            .where(Chunk.document_id == doc.id)
            .order_by(Chunk.chunk_index)
        ).all()
        assert len(chunks) >= 2
        for chk in chunks:
            assert chk.filename == "research_paper.pdf"
            assert chk.page_number in (1, 2)
            assert len(chk.text) > 0
            assert len(chk.embedding) == 768


def test_background_ingestion_broken_pdf_ends_as_failed(
    tmp_path: Path, session: Session
):
    """Verify corrupted PDF leads to status failed without crashing the API."""
    broken_file = tmp_path / "corrupted.pdf"
    broken_file.write_bytes(b"%PDF-1.4 unreadable stream without catalog")

    doc = Document(
        filename="corrupted.pdf",
        content_type="application/pdf",
        size_bytes=len(broken_file.read_bytes()),
        status=DocumentStatus.PENDING,
        file_path=str(broken_file),
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    ingest_document_pdf(
        document_id=doc.id,
        file_path=str(broken_file),
        session_factory=lambda: Session(test_engine),
    )

    with Session(test_engine) as check_session:
        updated_doc = check_session.get(Document, doc.id)
        assert updated_doc is not None
        assert updated_doc.status == DocumentStatus.FAILED
        assert updated_doc.error_message is not None
        assert "Corrupted or unreadable" in updated_doc.error_message


def test_background_ingestion_empty_pdf_ends_as_failed(
    tmp_path: Path, session: Session
):
    """Verify PDF with 0 extractable text leads to status failed."""
    blank_file = tmp_path / "empty.pdf"
    create_blank_pdf_file(blank_file)

    doc = Document(
        filename="empty.pdf",
        content_type="application/pdf",
        size_bytes=len(blank_file.read_bytes()),
        status=DocumentStatus.PENDING,
        file_path=str(blank_file),
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    ingest_document_pdf(
        document_id=doc.id,
        file_path=str(blank_file),
        session_factory=lambda: Session(test_engine),
    )

    with Session(test_engine) as check_session:
        updated_doc = check_session.get(Document, doc.id)
        assert updated_doc is not None
        assert updated_doc.status == DocumentStatus.FAILED
        assert "no extractable text" in updated_doc.error_message


def test_background_ingestion_embedding_error_ends_as_failed(
    tmp_path: Path, session: Session, monkeypatch: pytest.MonkeyPatch
):
    """Verify embedding API failure transitions status to failed gracefully."""
    pdf_file = tmp_path / "valid_doc.pdf"
    create_sample_multipage_pdf(pdf_file)

    doc = Document(
        filename="valid_doc.pdf",
        content_type="application/pdf",
        size_bytes=len(pdf_file.read_bytes()),
        status=DocumentStatus.PENDING,
        file_path=str(pdf_file),
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    def failing_embed(*args, **kwargs):
        raise RuntimeError("Quota exhausted on embedding provider.")

    monkeypatch.setattr(
        ingestion_module,
        "generate_embeddings_batch",
        failing_embed,
    )

    ingest_document_pdf(
        document_id=doc.id,
        file_path=str(pdf_file),
        session_factory=lambda: Session(test_engine),
    )

    with Session(test_engine) as check_session:
        updated_doc = check_session.get(Document, doc.id)
        assert updated_doc is not None
        assert updated_doc.status == DocumentStatus.FAILED
        assert "Quota exhausted" in updated_doc.error_message


def test_upload_pdf_returns_201_and_saves_file(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify uploading a PDF returns 201 immediately and saves file to uploads/."""
    pdf_path = tmp_path / "architecture.pdf"
    create_sample_multipage_pdf(pdf_path)
    pdf_bytes = pdf_path.read_bytes()

    uploads_dir = tmp_path / "test_uploads"
    monkeypatch.setattr("app.main.UPLOAD_DIR", uploads_dir)

    files = {"file": ("architecture.pdf", pdf_bytes, "application/pdf")}
    response = client.post("/documents", files=files)
    assert response.status_code == 201

    data = response.json()
    assert data["filename"] == "architecture.pdf"
    assert data["status"] in (DocumentStatus.PENDING, DocumentStatus.READY)
    doc_id = data["id"]

    # Verify file was written to disk
    saved_files = list(uploads_dir.glob(f"{doc_id}_*"))
    assert len(saved_files) == 1
    assert saved_files[0].read_bytes() == pdf_bytes

    # Verify GET /documents/{id}
    detail_res = client.get(f"/documents/{doc_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["id"] == doc_id


def test_upload_path_traversal_sanitized(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify filename with path traversal characters is safely sanitized."""
    pdf_path = tmp_path / "valid.pdf"
    create_sample_multipage_pdf(pdf_path)
    uploads_dir = tmp_path / "test_uploads"
    monkeypatch.setattr("app.main.UPLOAD_DIR", uploads_dir)

    files = {
        "file": ("../../etc/passwd.pdf", pdf_path.read_bytes(), "application/pdf")
    }
    response = client.post("/documents", files=files)
    assert response.status_code == 201

    doc_id = response.json()["id"]
    saved = list(uploads_dir.glob(f"{doc_id}_*"))
    assert len(saved) == 1
    assert saved[0].is_relative_to(uploads_dir)
    assert ".." not in saved[0].name


def test_get_document_invalid_uuid_returns_422(client: TestClient):
    """Verify FastAPI auto-validates UUID path parameter and returns 422."""
    resp = client.get("/documents/not-a-valid-uuid")
    assert resp.status_code == 422


def test_get_document_chunks_endpoint(client: TestClient, session: Session):
    """Verify GET /documents/{id}/chunks returns ordered chunk metadata."""
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        filename="manual.pdf",
        content_type="application/pdf",
        size_bytes=2048,
        status=DocumentStatus.READY,
    )
    session.add(doc)

    chk1 = Chunk(
        document_id=doc_id,
        filename="manual.pdf",
        page_number=1,
        chunk_index=0,
        text="Chapter 1 Introduction to vectors",
        embedding=[0.1] * 768,
    )
    chk2 = Chunk(
        document_id=doc_id,
        filename="manual.pdf",
        page_number=1,
        chunk_index=1,
        text="Chapter 2 Ingestion pipelines",
        embedding=[0.2] * 768,
    )
    session.add(chk1)
    session.add(chk2)
    session.commit()

    resp = client.get(f"/documents/{doc_id}/chunks")
    assert resp.status_code == 200
    chunks_data = resp.json()
    assert len(chunks_data) == 2
    assert chunks_data[0]["chunk_index"] == 0
    assert chunks_data[0]["text"] == "Chapter 1 Introduction to vectors"
    assert chunks_data[1]["chunk_index"] == 1
