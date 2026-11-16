"""Unit and integration tests for GET /search semantic document retrieval."""

import uuid
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

import app.db as db_module
import app.ingestion as ingestion_module
from app.db import get_session
from app.ingestion import EMBEDDING_DIMENSION
from app.main import app
from app.models import Chunk, Document, DocumentStatus

test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


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
    """Provide TestClient with database session and embedding mocked."""

    def get_session_override() -> Generator[Session, None, None]:
        yield session

    app.dependency_overrides[get_session] = get_session_override
    orig_engine = db_module.engine
    db_module.engine = test_engine

    # Default query mock: unit vector pointing along first dimension
    mock_query_vector = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)
    monkeypatch.setattr(
        ingestion_module,
        "generate_embedding",
        lambda *args, **kwargs: mock_query_vector,
    )

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()
    db_module.engine = orig_engine


def test_search_missing_query_returns_422(client: TestClient):
    """Verify GET /search without ?q parameter returns 422 Unprocessable Content."""
    resp = client.get("/search")
    assert resp.status_code == 422


def test_search_empty_query_returns_422(client: TestClient):
    """Verify GET /search with empty or whitespace-only q returns 422."""
    resp_empty = client.get("/search?q=")
    assert resp_empty.status_code == 422

    resp_ws = client.get("/search?q=%20%20%20")
    assert resp_ws.status_code == 422
    assert "empty or whitespace" in resp_ws.json()["detail"].lower()


def test_search_no_documents_ready_returns_empty_list(
    client: TestClient, session: Session
):
    """Verify GET /search returns empty list [] when no documents are READY."""
    # 1. Zero documents in DB
    resp = client.get("/search?q=attention%20mechanisms")
    assert resp.status_code == 200
    assert resp.json() == []

    # 2. Document exists but status is pending / processing
    doc = Document(
        id=uuid.uuid4(),
        filename="draft.pdf",
        content_type="application/pdf",
        size_bytes=1024,
        status=DocumentStatus.PENDING,
    )
    session.add(doc)
    session.commit()

    resp_pending = client.get("/search?q=attention%20mechanisms")
    assert resp_pending.status_code == 200
    assert resp_pending.json() == []


def test_search_returns_sorted_closest_first_with_metadata(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
):
    """Verify /search returns closest chunks sorted by cosine distance with metadata."""
    # Mock query vector: along dimension 0
    query_vec = [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)
    monkeypatch.setattr(
        ingestion_module,
        "generate_embedding",
        lambda *args, **kwargs: query_vec,
    )

    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        filename="deep_learning_guide.pdf",
        content_type="application/pdf",
        size_bytes=4096,
        status=DocumentStatus.READY,
    )
    session.add(doc)

    # Chunk A: Identical vector (cosine similarity = 1.0, distance = 0.0)
    chk_a = Chunk(
        document_id=doc_id,
        filename="deep_learning_guide.pdf",
        page_number=3,
        chunk_index=0,
        text="Self-attention allows tokens to attend across sequences.",
        embedding=[1.0] + [0.0] * (EMBEDDING_DIMENSION - 1),
    )
    # Chunk B: Partially aligned vector (cosine similarity ~ 0.707)
    chk_b = Chunk(
        document_id=doc_id,
        filename="deep_learning_guide.pdf",
        page_number=5,
        chunk_index=1,
        text="Residual connections prevent vanishing gradients in deep nets.",
        embedding=[0.7071, 0.7071] + [0.0] * (EMBEDDING_DIMENSION - 2),
    )
    # Chunk C: Orthogonal vector (cosine similarity = 0.0, distance = 1.0)
    chk_c = Chunk(
        document_id=doc_id,
        filename="deep_learning_guide.pdf",
        page_number=8,
        chunk_index=2,
        text="Relational databases use B-trees for primary key indexes.",
        embedding=[0.0, 1.0] + [0.0] * (EMBEDDING_DIMENSION - 2),
    )
    session.add(chk_a)
    session.add(chk_b)
    session.add(chk_c)
    session.commit()

    resp = client.get("/search?q=what%20is%20attention&k=5")
    assert resp.status_code == 200
    results = resp.json()

    assert len(results) == 3

    # Verify required keys in each result: text, filename, page, score
    for item in results:
        assert "text" in item
        assert "filename" in item
        assert "page" in item
        assert "score" in item
        assert item["filename"] == "deep_learning_guide.pdf"

    # Verify sorted closest first (highest similarity / closest distance first)
    assert results[0]["page"] == 3
    assert "Self-attention" in results[0]["text"]
    assert results[0]["score"] == 1.0

    assert results[1]["page"] == 5
    assert results[1]["score"] < results[0]["score"]
    assert results[1]["score"] > results[2]["score"]

    assert results[2]["page"] == 8
    assert results[2]["score"] == 0.0


def test_search_respects_k_limit(client: TestClient, session: Session):
    """Verify /search limits results to the specified k parameter."""
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        filename="textbook.pdf",
        content_type="application/pdf",
        size_bytes=2048,
        status=DocumentStatus.READY,
    )
    session.add(doc)

    for i in range(10):
        chk = Chunk(
            document_id=doc_id,
            filename="textbook.pdf",
            page_number=i + 1,
            chunk_index=i,
            text=f"Chapter {i + 1} content on machine learning foundations.",
            embedding=[0.1 * (i + 1)] * EMBEDDING_DIMENSION,
        )
        session.add(chk)
    session.commit()

    # Request top 3
    resp_k3 = client.get("/search?q=machine%20learning&k=3")
    assert resp_k3.status_code == 200
    assert len(resp_k3.json()) == 3

    # Request default (k=5)
    resp_def = client.get("/search?q=machine%20learning")
    assert resp_def.status_code == 200
    assert len(resp_def.json()) == 5


def test_search_invalid_k_rejected(client: TestClient):
    """Verify /search with k < 1 or k > 100 returns 422."""
    resp_zero = client.get("/search?q=valid&k=0")
    assert resp_zero.status_code == 422

    resp_too_large = client.get("/search?q=valid&k=150")
    assert resp_too_large.status_code == 422
