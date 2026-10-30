# AI Document Assistant (`ai-doc-assistant`)

A production-ready asynchronous FastAPI service for ingesting, validating, and cataloging documents (`.pdf`, `.txt`, `.docx`). This stage introduces **PostgreSQL persistence via SQLModel** and full **Docker container orchestration via Docker Compose**.

---

## 📋 Features & Specifications

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
├── app/
│   ├── __init__.py
│   ├── db.py                # Database engine, session generator & retry initialization
│   ├── main.py              # FastAPI app, lifespan, route handlers & validation
│   └── models.py            # SQLModel table model & Pydantic response schemas
├── tests/
│   ├── __init__.py
│   └── test_documents.py    # 8 automated tests with in-memory SQLite fixture
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
- **Python** (optional for local non-Docker development: `>= 3.12` with `uv`)

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

A key requirement of Assignment 4 is that data survives container teardown:

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
Uploads a document (`.pdf`, `.txt`, `.docx`), validates type and size, and persists metadata in PostgreSQL.

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

## 🧪 Running Automated Tests

Run the test suite using `pytest`. The tests use an in-memory SQLite engine via FastAPI dependency overrides and do not require PostgreSQL or Docker:

```bash
uv run pytest -v
```

### Test Coverage:
1. `test_health_check`: Verifies `GET /health` returns `200 OK`.
2. `test_upload_document_success`: Verifies valid upload returns `201 Created`, correct metadata, and stores into the database.
3. `test_upload_document_empty_file`: Verifies 0-byte upload returns `400 Bad Request` and does not insert rows into the database.
4. `test_upload_document_wrong_file_type`: Verifies disallowed extensions return `415 Unsupported Media Type`.
5. `test_upload_document_file_too_large`: Verifies files exceeding 10 MB return `413 Content Too Large`.
6. `test_list_documents`: Verifies `GET /documents` reads all records from the database table.
7. `test_upload_docx_and_case_insensitive_extension`: Verifies `.docx` and uppercase extensions persist with appropriate MIME types.
8. `test_database_persistence_across_sessions`: Verifies document records persist across distinct database sessions.
