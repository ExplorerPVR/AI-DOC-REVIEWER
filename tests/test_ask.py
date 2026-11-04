"""Unit and integration tests for POST /ask endpoint and LLM generation."""

import uuid
from collections.abc import Generator, Iterator

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine, select
from sqlmodel.pool import StaticPool

import app.llm as llm_module
from app.db import get_session
from app.llm import (
    LLMRateLimitError,
    LLMTimeoutError,
    estimate_tokens,
)
from app.main import app
from app.models import LLMCall
from app.pricing import calculate_cost, get_model_price

# Isolated in-memory SQLite engine for LLM tests
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


@pytest.fixture(name="session")
def session_fixture() -> Generator[Session, None, None]:
    """Provide a fresh, isolated database session with all tables created."""
    SQLModel.metadata.create_all(test_engine)
    with Session(test_engine) as session:
        yield session
    SQLModel.metadata.drop_all(test_engine)


@pytest.fixture(name="client")
def client_fixture(session: Session) -> Generator[TestClient, None, None]:
    """Provide TestClient with database session dependency overridden."""

    def get_session_override() -> Generator[Session, None, None]:
        yield session

    app.dependency_overrides[get_session] = get_session_override
    original_engine = llm_module.engine
    llm_module.engine = test_engine

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()
    llm_module.engine = original_engine


def test_pricing_calculation():
    """Verify pricing lookup and accurate cost calculations."""
    gemini_price = get_model_price("gemini-1.5-flash")
    assert gemini_price.input_price_per_million == 0.075
    assert gemini_price.output_price_per_million == 0.30

    # 1M input + 1M output tokens = $0.075 + $0.30 = $0.375
    cost = calculate_cost("gemini-1.5-flash", 1_000_000, 1_000_000)
    assert cost == 0.375

    # Small call calculation (1000 input, 500 output)
    small_cost = calculate_cost("gemini-1.5-flash", 1000, 500)
    assert small_cost > 0.0

    # Groq pricing calculation
    groq_cost = calculate_cost("llama-3.3-70b-versatile", 1_000_000, 1_000_000)
    assert groq_cost == 1.38

    # Free/local model cost
    free_cost = calculate_cost("llama3", 500, 500)
    assert free_cost == 0.0


def test_estimate_tokens():
    """Verify token estimation heuristic."""
    assert estimate_tokens("") == 0
    text = "Explain the difference between SQL and NoSQL in two paragraphs."
    tokens = estimate_tokens(text)
    assert tokens > 0


def test_ask_gemini_streaming_success(
    client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test 1: POST /ask streams answers from Gemini and records LLMCall in DB."""
    mock_chunks = ["SQL is relational", " while NoSQL", " is non-relational."]

    def mock_stream_gemini(
        prompt: str,
        model: str,
        timeout: float = 30.0,
    ) -> Iterator[str]:
        yield from mock_chunks

    monkeypatch.setattr(llm_module, "_stream_gemini", mock_stream_gemini)

    payload = {"question": "What is SQL vs NoSQL?", "provider": "gemini"}
    response = client.post("/ask", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "".join(mock_chunks)

    # Verify LLMCall row recorded in database
    calls = session.exec(select(LLMCall)).all()
    assert len(calls) == 1
    call = calls[0]
    assert uuid.UUID(str(call.id), version=4)
    assert call.provider == "gemini"
    assert call.model == "gemini-1.5-flash"
    assert call.input_tokens > 0
    assert call.output_tokens > 0
    assert call.cost_usd > 0.0
    assert call.time_taken_seconds >= 0.0


def test_ask_groq_streaming_success(
    client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test 2: POST /ask streams answers from Groq and records LLMCall in DB."""
    mock_chunks = ["Groq delivers", " ultra-fast", " inference speed."]

    def mock_stream_groq(
        prompt: str,
        model: str,
        timeout: float = 30.0,
    ) -> Iterator[str]:
        yield from mock_chunks

    monkeypatch.setattr(llm_module, "_stream_groq", mock_stream_groq)

    payload = {"question": "Why use Groq?", "provider": "groq"}
    response = client.post("/ask", json=payload)

    assert response.status_code == 200
    assert response.text == "".join(mock_chunks)

    calls = session.exec(select(LLMCall).where(LLMCall.provider == "groq")).all()
    assert len(calls) == 1
    call = calls[0]
    assert call.model == "llama-3.3-70b-versatile"
    assert call.input_tokens > 0
    assert call.output_tokens > 0
    assert call.cost_usd > 0.0


def test_ask_ollama_streaming_success(
    client: TestClient,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test 3: POST /ask streams answers from local Ollama."""
    mock_chunks = ["Ollama runs", " models locally", " on your machine."]

    def mock_stream_ollama(
        prompt: str,
        model: str,
        timeout: float = 30.0,
    ) -> Iterator[str]:
        yield from mock_chunks

    monkeypatch.setattr(llm_module, "_stream_ollama", mock_stream_ollama)

    payload = {"question": "How does Ollama work?", "provider": "ollama"}
    response = client.post("/ask", json=payload)

    assert response.status_code == 200
    assert response.text == "".join(mock_chunks)

    calls = session.exec(select(LLMCall).where(LLMCall.provider == "ollama")).all()
    assert len(calls) == 1
    call = calls[0]
    assert call.model == "llama3"
    assert call.cost_usd == 0.0


def test_ask_rate_limit_429_retried_and_fails_with_503(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test 4: Rate limit (429) is retried, then returns 503 Service Unavailable."""
    attempts = 0

    def mock_stream_failing(
        prompt: str,
        model: str,
        timeout: float = 30.0,
    ) -> Iterator[str]:
        nonlocal attempts
        attempts += 1
        raise LLMRateLimitError("Gemini quota exhausted (HTTP 429)")

    monkeypatch.setattr(llm_module, "_stream_gemini", mock_stream_failing)

    payload = {"question": "Trigger rate limit", "provider": "gemini"}
    response = client.post("/ask", json=payload)

    # Must retry up to configured retries (3) and return HTTP 503
    assert attempts == 3
    assert response.status_code == 503
    assert "LLM provider service unavailable" in response.json()["detail"]
    assert "429" in response.json()["detail"]


def test_ask_timeout_retried_and_fails_with_504(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test 5: Timeout is retried, then returns clear 504 Gateway Timeout."""
    attempts = 0

    def mock_stream_timeout(
        prompt: str,
        model: str,
        timeout: float = 30.0,
    ) -> Iterator[str]:
        nonlocal attempts
        attempts += 1
        raise LLMTimeoutError("Groq request timed out after 30s (504)")

    monkeypatch.setattr(llm_module, "_stream_groq", mock_stream_timeout)

    payload = {"question": "Trigger timeout", "provider": "groq"}
    response = client.post("/ask", json=payload)

    # Must retry and return HTTP 504
    assert attempts == 3
    assert response.status_code == 504
    assert "LLM provider request timed out" in response.json()["detail"]


def test_ask_empty_question_rejected(client: TestClient):
    """Test 6: Empty or whitespace question returns status 400 Bad Request."""
    response = client.post("/ask", json={"question": "   ", "provider": "gemini"})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_ask_unsupported_provider_rejected(client: TestClient):
    """Test 7: Unsupported provider name returns status 400 Bad Request."""
    payload = {"question": "Hello", "provider": "non_existent_provider"}
    response = client.post("/ask", json=payload)
    assert response.status_code == 400
    assert "Unsupported LLM provider" in response.json()["detail"]
