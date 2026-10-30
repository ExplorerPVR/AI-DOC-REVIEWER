"""Unit and integration tests for FastAPI document assistant endpoints."""

import uuid
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
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

    # Verify document exists in database
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
    assert "Empty file" in data["detail"]

    # Verify no documents inserted
    docs = session.exec(select(Document)).all()
    assert len(docs) == 0


def test_upload_document_wrong_file_type(client: TestClient, session: Session):
    """Test 4: Uploading unsupported file type returns status 415."""
    files = {"file": ("photo.png", b"\x89PNG\r\n\x1a\n", "image/png")}

    response = client.post("/documents", files=files)
    assert response.status_code == 415

    data = response.json()
    assert "Unsupported file type" in data["detail"]
    assert ".pdf" in data["detail"]
    assert ".txt" in data["detail"]
    assert ".docx" in data["detail"]

    docs = session.exec(select(Document)).all()
    assert len(docs) == 0


def test_upload_document_file_too_large(client: TestClient, session: Session):
    """Test 5: Uploading file larger than 10 MB returns status 413."""
    oversized_bytes = 10 * 1024 * 1024 + 1  # 10 MB + 1 byte
    oversized_content = b"x" * oversized_bytes
    files = {"file": ("large_file.pdf", oversized_content, "application/pdf")}

    response = client.post("/documents", files=files)
    assert response.status_code == 413

    data = response.json()
    assert "File too large" in data["detail"]

    docs = session.exec(select(Document)).all()
    assert len(docs) == 0


def test_list_documents(client: TestClient):
    """Test 6: Listing documents returns status 200 and previously stored records."""
    # Verify initially empty
    initial_res = client.get("/documents")
    assert initial_res.status_code == 200
    assert initial_res.json() == []

    # Upload two documents
    file1 = {"file": ("doc1.txt", b"First doc content", "text/plain")}
    file2 = {"file": ("doc2.pdf", b"%PDF-1.4 mock content", "application/pdf")}

    res1 = client.post("/documents", files=file1)
    res2 = client.post("/documents", files=file2)
    assert res1.status_code == 201
    assert res2.status_code == 201

    # List documents
    list_res = client.get("/documents")
    assert list_res.status_code == 200

    docs = list_res.json()
    assert len(docs) == 2
    assert docs[0]["id"] != docs[1]["id"]
    assert uuid.UUID(docs[0]["id"], version=4)
    assert uuid.UUID(docs[1]["id"], version=4)
    assert docs[0]["filename"] == "doc1.txt"
    assert docs[0]["content_type"] == "text/plain"
    assert docs[1]["filename"] == "doc2.pdf"
    assert docs[1]["content_type"] == "application/pdf"


def test_upload_docx_and_case_insensitive_extension(client: TestClient):
    """Test 7: Support for .docx files and uppercase extensions like .DOCX and .TXT."""
    files_docx = {
        "file": (
            "agreement.DOCX",
            b"Mock docx binary payload",
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
    file_data = {"file": ("persisted.pdf", b"%PDF-persisted", "application/pdf")}
    res = client.post("/documents", files=file_data)
    assert res.status_code == 201
    doc_id = res.json()["id"]

    # Verify query in a brand new isolated session against the same engine
    with Session(test_engine) as new_session:
        retrieved = new_session.get(Document, uuid.UUID(doc_id))
        assert retrieved is not None
        assert retrieved.filename == "persisted.pdf"
        assert retrieved.content_type == "application/pdf"
