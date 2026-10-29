"""Unit and integration tests for FastAPI document assistant endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app, documents_db


@pytest.fixture(autouse=True)
def reset_in_memory_db():
    """Ensure in-memory documents list is reset before and after every test."""
    documents_db.clear()
    yield
    documents_db.clear()


@pytest.fixture
def client() -> TestClient:
    """Fixture providing a FastAPI TestClient instance."""
    return TestClient(app)


def test_health_check(client: TestClient):
    """Test 1: Health check endpoint returns status 200 and {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_document_success(client: TestClient):
    """Test 2: Successful upload returns status 201 with metadata."""
    file_content = b"This is a sample document for testing."
    files = {"file": ("sample.txt", file_content, "text/plain")}

    response = client.post("/documents", files=files)
    assert response.status_code == 201

    data = response.json()
    assert data["id"] == 1
    assert data["filename"] == "sample.txt"
    assert data["size_bytes"] == len(file_content)
    assert "uploaded_at" in data


def test_upload_document_empty_file(client: TestClient):
    """Test 3: Empty file upload returns status 400 with a clear error message."""
    files = {"file": ("empty.txt", b"", "text/plain")}

    response = client.post("/documents", files=files)
    assert response.status_code == 400

    data = response.json()
    assert "Empty file" in data["detail"]


def test_upload_document_wrong_file_type(client: TestClient):
    """Test 4: Uploading unsupported file type returns status 415."""
    files = {"file": ("photo.png", b"\x89PNG\r\n\x1a\n", "image/png")}

    response = client.post("/documents", files=files)
    assert response.status_code == 415

    data = response.json()
    assert "Unsupported file type" in data["detail"]
    assert ".pdf" in data["detail"]
    assert ".txt" in data["detail"]
    assert ".docx" in data["detail"]


def test_upload_document_file_too_large(client: TestClient):
    """Test 5: Uploading file larger than 10 MB returns status 413."""
    oversized_bytes = 10 * 1024 * 1024 + 1  # 10 MB + 1 byte
    oversized_content = b"x" * oversized_bytes
    files = {"file": ("large_file.pdf", oversized_content, "application/pdf")}

    response = client.post("/documents", files=files)
    assert response.status_code == 413

    data = response.json()
    assert "File too large" in data["detail"]


def test_list_documents(client: TestClient):
    """Test 6: Listing documents returns status 200 and previously uploaded documents."""
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
    assert docs[0]["id"] == 1
    assert docs[0]["filename"] == "doc1.txt"
    assert docs[0]["size_bytes"] == len(b"First doc content")
    assert docs[1]["id"] == 2
    assert docs[1]["filename"] == "doc2.pdf"


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
    assert data["id"] == 1
