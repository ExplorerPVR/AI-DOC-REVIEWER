# AI Document Assistant (`ai-doc-assistant`)
[![CI](https://github.com/ExplorerPVR/AI-DOC-REVIEWER/actions/workflows/ci.yml/badge.svg)](https://github.com/ExplorerPVR/AI-DOC-REVIEWER/actions/workflows/ci.yml)

A production-ready asynchronous FastAPI service for ingesting, validating, and cataloging documents (`.pdf`, `.txt`, `.docx`). Features **PostgreSQL persistence via SQLModel**, full **Docker container orchestration**, **Automated CI with GitHub Actions**, and **Deep Document Content Validation**.

---

## 📋 Features & Specifications

- **Automated CI with GitHub Actions**:
  - Automatically runs on every push and pull request.
  - Enforces code quality and formatting via `ruff check .`.
  - Runs the full test suite via `pytest`.
- **Deep Document Content Validation**:
  - Validates document integrity beyond file size and extension.
  - Rejects blank PDFs (>0 bytes with no text or images), blank DOCX files (>0 bytes with no text or shapes), and whitespace-only text files.
  - Rejects malformed or corrupted files with clear `400 Bad Request` responses before saving.
- **Production LLM Streaming & Cost Tracking (`POST /ask`)**:
  - Multi-provider support: Google Gemini, Groq Cloud, and local Ollama.
  - Asynchronous chunk-by-chunk response streaming (`StreamingResponse`).
  - Automatic retries with exponential backoff on rate limits (429) and server errors (5xx).
  - 30-second call timeouts with clear `504 Gateway Timeout` and `503 Service Unavailable` error responses.
  - Automatically records tokens, execution duration, and USD cost at published list price into the `llm_calls` PostgreSQL table.
- **Database Schema Migrations via Alembic**:
  - Full version-controlled database schema management using Alembic.
  - Automatically runs `alembic upgrade head` on container startup before FastAPI initializes.
  - Zero `create_all()` in application code; schema changes are managed via reversible migration files.
- **PostgreSQL Database with SQLModel**:
  - `Document` database table storing:
    - `id`: Unique UUID primary key.
    - `filename`: Original name of the uploaded document.
    - `content_type`: MIME type (e.g. `application/pdf`, `text/plain`).
    - `size_bytes`: File size in bytes.
    - `uploaded_at`: UTC timestamp.
  - Dynamic connection resolution reading `DATABASE_URL` from `.env`.
- **Full Docker Containerization**:
  - **`app` Service**: FastAPI service running in a minimal Python 3.12 container with `uv`.
  - **`db` Service**: PostgreSQL 16 container with health check (`pg_isready`) and automated connection retries.
  - **Volume Persistence**: Named volume `postgres_data` ensures uploaded documents and metadata survive container restarts (`docker compose down` and `docker compose up`).
- **Zero-Secret Hardcoding**: All credentials and database URLs are loaded from `.env` (template in `.env.example`).
- **Standalone Test Suite**: Automated tests use an in-memory SQLite engine via FastAPI dependency overrides (`get_session`), passing completely standalone without requiring PostgreSQL running.

---

## 📁 Repository Structure

```text
ai-doc-assistant/
├── .github/
│   └── workflows/
│       └── ci.yml           # Automated CI pipeline (Ruff lint check & Pytest suite)
├── app/
│   ├── __init__.py
│   ├── db.py                # Database engine & session generator
│   ├── llm.py               # Multi-provider streaming, retries, and token tracking
│   ├── main.py              # FastAPI app, route handlers & validation
│   ├── models.py            # SQLModel table models (Document, LLMCall) & schemas
│   └── pricing.py           # Model pricing per million tokens & cost calculator
├── migrations/              # Alembic database migration scripts & versions
│   ├── versions/
│   │   ├── bb20b80bdc64_create_document_table.py
│   │   └── 43ce2c5fbd08_create_llm_calls_table.py
│   ├── env.py               # Alembic environment config (reads DATABASE_URL)
│   ├── README
│   └── script.py.mako
├── tests/
│   ├── __init__.py
│   ├── test_ask.py          # 9 automated tests for LLM streaming, retries, and costs
│   └── test_documents.py    # 14 automated tests with in-memory SQLite fixture
├── alembic.ini              # Alembic CLI configuration file
├── .dockerignore            # Excludes caches, .venv, .git, and .env from Docker context
├── .env                     # Local environment & DB credentials (gitignored)
├── .env.example             # Template environment variables (committed)
├── .gitignore               # Excludes caches, .venv, .env, and generated data
├── docker-compose.yml       # Two-service stack (app + postgres:16 with persistent volume)
├── Dockerfile               # Multi-stage container build with Astral uv
├── pyproject.toml           # Project dependencies managed via uv
├── uv.lock                  # Deterministic dependency lockfile
└── README.md                # Comprehensive documentation and run guide
```

---

## 🛠️ Prerequisites

- **Docker Desktop** (version 24+ with Docker Compose v2)
- **Python** (optional for local development: `>= 3.12` with `uv`)

---

## 🚀 Quickstart: Run with Docker Compose

Run the entire application stack (PostgreSQL + FastAPI) with a single command:

```bash
# 1. Copy sample environment file
cp .env.example .env

# 2. Build and start services in detached mode
docker compose up -d --build
```

Once started:
- **Interactive Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc UI**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Service Health Probe**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 🔄 Verifying Database Persistence Across Restarts

```bash
# 1. Upload a document while services are running
curl -X POST "http://127.0.0.1:8000/documents" \
  -F "file=@sample.txt"

# 2. Stop and remove containers
docker compose down

# 3. Start services again
docker compose up -d

# 4. Verify the document is still present in PostgreSQL
curl -X GET "http://127.0.0.1:8000/documents"
```
The uploaded document metadata remains intact because the `postgres_data` volume preserves the PostgreSQL database across container lifecycles.

---

## 🗄️ Database Migrations with Alembic

Database schema evolution is managed via **Alembic**. Table creation is handled exclusively via migration scripts rather than `create_all()`.

```bash
# Apply all pending migrations to head
uv run alembic upgrade head

# Roll back the most recent migration
uv run alembic downgrade -1

# Inspect the current database schema revision
uv run alembic current

# Generate a new migration after updating SQLModel models
uv run alembic revision --autogenerate -m "describe schema change"
```

When running under Docker Compose, the container automatically applies `alembic upgrade head` upon startup before launching Uvicorn.

---

## 🧪 API Endpoints & `curl` Examples

### 1. Health Check
- **URL**: `GET /health`
- **Status**: `200 OK`

```bash
curl -X GET "http://127.0.0.1:8000/health"
```
```json
{
  "status": "ok"
}
```

---

### 2. Upload Document
Uploads a document (`.pdf`, `.txt`, `.docx`), validates content integrity and size, and persists metadata in PostgreSQL.

- **URL**: `POST /documents`
- **Content-Type**: `multipart/form-data`
- **Success Status**: `201 Created`

```bash
curl -X POST "http://127.0.0.1:8000/documents" \
  -F "file=@agreement.pdf"
```

**Response (`201 Created`)**:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "agreement.pdf",
  "content_type": "application/pdf",
  "size_bytes": 1048576,
  "uploaded_at": "2026-10-29T14:30:00.123456Z"
}
```

#### Error Responses:
- **Empty File (`400 Bad Request`)**:
  ```json
  {
    "detail": "Empty file. Uploaded file contains no data."
  }
  ```
- **Blank Document (`400 Bad Request`)**:
  ```json
  {
    "detail": "Blank document. PDF has no readable text or visual content."
  }
  ```
  or
  ```json
  {
    "detail": "Blank document. File contains only whitespace characters."
  }
  ```
- **Corrupted Document (`400 Bad Request`)**:
  ```json
  {
    "detail": "Corrupted or invalid file. Unable to parse PDF document."
  }
  ```
- **Wrong File Type (`415 Unsupported Media Type`)**:
  ```json
  {
    "detail": "Unsupported file type '.png'. Allowed types are: .docx, .pdf, .txt"
  }
  ```
- **File Too Large (`413 Content Too Large`)**:
  ```json
  {
    "detail": "File too large (10485761 bytes). Maximum allowed size is 10 MB (10485760 bytes)."
  }
  ```

---

### 3. List Documents
Fetches all stored document records from PostgreSQL.

- **URL**: `GET /documents`
- **Status**: `200 OK`

```bash
curl -X GET "http://127.0.0.1:8000/documents"
```

**Response (`200 OK`)**:
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "filename": "agreement.pdf",
    "content_type": "application/pdf",
    "size_bytes": 1048576,
    "uploaded_at": "2026-10-29T14:30:00.123456Z"
  }
]
```

---

---

### 4. Ask an LLM (Streaming Completion)
Streams answers from configured LLM providers (Gemini, Groq, Ollama) with automatic retries, timeouts, and call cost logging.

- **URL**: `POST /ask`
- **Content-Type**: `application/json`
- **Transfer-Encoding**: `chunked` (text/plain streaming)

```bash
# Ask Gemini (default)
curl -N -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the difference between SQL and NoSQL in two sentences?", "provider": "gemini"}'

# Ask Groq
curl -N -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain Docker containers briefly.", "provider": "groq"}'

# Ask Local Ollama
curl -N -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Hello from local LLM!", "provider": "ollama"}'
```

#### Error Handling:
- **Rate Limit Exceeded (`503 Service Unavailable`)**: Returned if the LLM provider rate limits (429) persist after 3 retries with exponential backoff.
- **Gateway Timeout (`504 Gateway Timeout`)**: Returned if generation exceeds the 30-second timeout threshold across retries.
- **Unsupported Provider (`400 Bad Request`)**: Returned if a non-supported provider name is passed.

---

## 🤖 Continuous Integration (CI) Checks

The CI pipeline runs automatically on all pushes and pull requests via GitHub Actions (`.github/workflows/ci.yml`).

### Workflow Steps:
1. **Lint & Code Style**: Runs `uv run ruff check .` to check for style violations, unformatted imports, and common programming errors.
2. **Automated Testing**: Runs `uv run pytest` to execute the full unit and integration test suite (23 automated tests) against the in-memory SQLite database.

### Running CI Checks Locally:
```bash
# Run Ruff linting check
uv run ruff check .

# Run Pytest suite
uv run pytest -v
```
