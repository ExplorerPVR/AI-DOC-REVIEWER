"""Unit and integration tests for FastAPI document assistant endpoints."""

import io
import uuid
from collections.abc import Generator

import pytest
from docx import Document as DocxDocument
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from sqlmodel import Session, SQLModel, create_engine, select
from sqlmodel.pool import StaticPool

from app.db import get_session
from app.main import app
from app.models import Document

# Test in-memory SQLite database engine
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


def create_valid_pdf_bytes() -> bytes:
    """Generate minimal valid PDF bytes containing text."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        b"4 0 obj << /Length 44 >> stream\n"
        b"BT /F1 12 Tf 100 700 Td (Valid PDF Content) Tj ET\n"
        b"endstream endobj\n"
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000255 00000 n \n"
        b"0000000357 00000 n \n"
        b"trailer << /Size 6 /Root 1 0 R >>\n"
        b"startxref\n436\n%%EOF"
    )


def create_blank_pdf_bytes() -> bytes:
    """Generate a blank PDF with 1 page and no text or images."""
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def create_valid_docx_bytes() -> bytes:
    """Generate a valid DOCX document containing text."""
    doc = DocxDocument()
    doc.add_paragraph("Valid DOCX document paragraph text.")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def create_blank_docx_bytes() -> bytes:
    """Generate a blank DOCX document with no text or shapes."""
    doc = DocxDocument()
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture(name="session")
def session_fixture() -> Generator[Session, None, None]:
    """Provide a fresh, isolated database session for each test."""
    SQLModel.metadata.create_all(test_engine)
    with Session(test_engine) as session:
        yield session
    SQLModel.metadata.drop_all(test_engine)


@pytest.fixture(name="client")
def client_fixture(session: Session) -> Generator[TestClient, None, None]:
    """Provide a TestClient with the database session dependency overridden."""

    def get_session_override() -> Generator[Session, None, None]:
        yield session

    app.dependency_overrides[get_session] = get_session_override
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_health_check(client: TestClient):
    """Test 1: Health check endpoint returns status 200 and {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_document_success(client: TestClient, session: Session):
    """Test 2: Successful upload returns 201 with metadata and stores in DB."""
    file_content = b"This is a sample document for testing database persistence."
    files = {"file": ("sample.txt", file_content, "text/plain")}

    response = client.post("/documents", files=files)
    assert response.status_code == 201

    data = response.json()
    assert uuid.UUID(data["id"], version=4)
    assert data["filename"] == "sample.txt"
    assert data["content_type"] == "text/plain"
    assert data["size_bytes"] == len(file_content)
    assert "uploaded_at" in data

    db_doc = session.get(Document, uuid.UUID(data["id"]))
    assert db_doc is not None
    assert db_doc.filename == "sample.txt"
    assert db_doc.content_type == "text/plain"
    assert db_doc.size_bytes == len(file_content)


def test_upload_document_empty_file(client: TestClient, session: Session):
    """Test 3: Empty file upload returns status 400 and does not write to DB."""
    files = {"file": ("empty.txt", b"", "text/plain")}

    response = client.post("/documents", files=files)
    assert response.status_code == 400

    data = response.json()
    assert "empty" in data["detail"].lower() or "whitespace" in data["detail"].lower()

    docs = session.exec(select(Document)).all()
    assert len(docs) == 0


def test_upload_document_wrong_file_type(client: TestClient, session: Session):
    """Test 4: Disallowed file extension returns status 415 and does not write to DB."""
    files = {"file": ("script.py", b"print('hello world')", "text/x-python")}

    response = client.post("/documents", files=files)
    assert response.status_code == 415
    assert "Unsupported file type '.py'" in response.json()["detail"]

    docs = session.exec(select(Document)).all()
    assert len(docs) == 0


def test_upload_document_file_too_large(client: TestClient, session: Session):
    """Test 5: File exceeding 10 MB limit returns 413 and does not write to DB."""
    large_size = 10 * 1024 * 1024 + 1
    large_content = b"x" * large_size
    files = {"file": ("oversized.txt", large_content, "text/plain")}

    response = client.post("/documents", files=files)
    assert response.status_code == 413

    data = response.json()
    assert "File too large" in data["detail"]
    assert "10 MB" in data["detail"]

    docs = session.exec(select(Document)).all()
    assert len(docs) == 0


def test_list_documents(client: TestClient):
    """Test 6: Listing documents returns status 200 and previously stored records."""
    initial_res = client.get("/documents")
    assert initial_res.status_code == 200
    assert initial_res.json() == []

    file1 = {"file": ("doc1.txt", b"First doc content", "text/plain")}
    file2 = {"file": ("doc2.pdf", create_valid_pdf_bytes(), "application/pdf")}

    res1 = client.post("/documents", files=file1)
    res2 = client.post("/documents", files=file2)
    assert res1.status_code == 201
    assert res2.status_code == 201

    list_res = client.get("/documents")
    assert list_res.status_code == 200

    docs = list_res.json()
    assert len(docs) == 2
    assert docs[0]["id"] != docs[1]["id"]
    assert uuid.UUID(docs[0]["id"], version=4)
    assert uuid.UUID(docs[1]["id"], version=4)
    assert docs[0]["filename"] == "doc1.txt"
    assert docs[1]["filename"] == "doc2.pdf"


def test_upload_docx_and_case_insensitive_extension(client: TestClient):
    """Test 7: Support for .docx files and uppercase extensions like .DOCX."""
    files_docx = {
        "file": (
            "agreement.DOCX",
            create_valid_docx_bytes(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    }
    response = client.post("/documents", files=files_docx)
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "agreement.DOCX"
    expected_type = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert data["content_type"] == expected_type
    assert uuid.UUID(data["id"], version=4)


def test_database_persistence_across_sessions(client: TestClient):
    """Test 8: Verify data persists across separate database operations."""
    file_data = {
        "file": ("persisted.pdf", create_valid_pdf_bytes(), "application/pdf")
    }
    res = client.post("/documents", files=file_data)
    assert res.status_code == 201
    doc_id = res.json()["id"]

    with Session(test_engine) as new_session:
        retrieved = new_session.get(Document, uuid.UUID(doc_id))
        assert retrieved is not None
        assert retrieved.filename == "persisted.pdf"


def test_upload_blank_pdf_rejected(client: TestClient):
    """Test 9: Uploading a blank PDF (>0 bytes with no text) returns status 400."""
    blank_pdf = create_blank_pdf_bytes()
    assert len(blank_pdf) > 0  # Confirms it is > 0 bytes

    files = {"file": ("blank_paper.pdf", blank_pdf, "application/pdf")}
    response = client.post("/documents", files=files)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "no extractable text" in detail.lower()


def test_upload_blank_docx_rejected(client: TestClient):
    """Test 10: Uploading a blank DOCX (>0 bytes with no text) returns status 400."""
    blank_docx = create_blank_docx_bytes()
    assert len(blank_docx) > 0  # Confirms it is > 0 bytes

    files = {
        "file": (
            "blank_doc.docx",
            blank_docx,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    }
    response = client.post("/documents", files=files)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert (
        "no extractable text" in detail.lower()
        or "no readable text" in detail.lower()
    )


def test_upload_whitespace_txt_rejected(client: TestClient):
    """Test 11: Uploading a whitespace-only TXT file returns status 400."""
    whitespace_bytes = b"   \r\n\t  \n  "
    assert len(whitespace_bytes) > 0

    files = {"file": ("spaces.txt", whitespace_bytes, "text/plain")}
    response = client.post("/documents", files=files)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "whitespace" in detail.lower()


def test_upload_corrupted_pdf_rejected(client: TestClient):
    """Test 12: Uploading a corrupted PDF file returns status 400."""
    corrupted_bytes = b"%PDF-1.4 garbage binary random unreadable stream"
    files = {"file": ("corrupt.pdf", corrupted_bytes, "application/pdf")}

    response = client.post("/documents", files=files)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "invalid or unreadable" in detail.lower()


def test_upload_corrupted_docx_rejected(client: TestClient):
    """Test 13: Uploading a corrupted DOCX file returns status 400."""
    corrupted_bytes = b"PK\x03\x04 not a valid zip archive for docx"
    files = {
        "file": (
            "corrupt.docx",
            corrupted_bytes,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    }

    response = client.post("/documents", files=files)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "corrupted" in detail.lower() or "invalid" in detail.lower()


def test_upload_duplicate_filename_rejected(client: TestClient):
    """Test 14: Uploading duplicate filename returns status 409 Conflict."""
    files = {"file": ("unique_doc.txt", b"Initial content here", "text/plain")}
    res1 = client.post("/documents", files=files)
    assert res1.status_code == 201

    files_dup = {
        "file": ("unique_doc.txt", b"Different content but same name", "text/plain")
    }
    res2 = client.post("/documents", files=files_dup)
    assert res2.status_code == 409
    assert res2.json()["detail"] == "A document with this filename already exists."
