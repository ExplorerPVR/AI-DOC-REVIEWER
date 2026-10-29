# AI Document Assistant (`ai-doc-assistant`)
> **Assignment 3: Practice Project – FastAPI Basics**

A clean, asynchronous FastAPI service for ingesting and cataloging documents (`.pdf`, `.txt`, `.docx`). This project represents the foundational stage of an AI Assistant that will later answer questions over uploaded documents using LLMs and RAG (Retrieval-Augmented Generation).

---

## 📋 Features & Specifications

- **Health Check Probe (`GET /health`)**: Returns `200 OK` with `{"status": "ok"}`.
- **Document Upload (`POST /documents`)**:
  - Accepts single file uploads via `multipart/form-data`.
  - **Allowed Extensions**: `.pdf`, `.txt`, `.docx`.
  - **Maximum File Size**: 10 MB (10,485,760 bytes).
  - **In-Memory Catalog**: Metadata stored in memory (`id` as UUID, `filename`, `size_bytes`, `uploaded_at`).
  - **Strict Error Handling**:
    - `400 Bad Request`: Empty file payload.
    - `415 Unsupported Media Type`: Disallowed file extension.
    - `413 Content Too Large`: Payload exceeds 10 MB limit.
    - `201 Created`: Successful upload returning file metadata.
- **List Documents (`GET /documents`)**: Returns `200 OK` and a JSON array of all stored documents.
- **Interactive Documentation**: Available out-of-the-box via Swagger UI (`/docs`) and ReDoc (`/redoc`).

---

## 📁 Repository Structure

```text
ai-doc-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app definition, routes, validation & in-memory store
│   └── models.py            # Pydantic request, response, and error schemas
├── tests/
│   ├── __init__.py
│   └── test_documents.py    # TestClient test suite (7 comprehensive test cases)
├── .env                     # Local environment variables (gitignored)
├── .env.example             # Template environment variables (committed)
├── .gitignore               # Excludes caches, .venv, .env, and generated data
├── pyproject.toml           # Project dependencies managed via uv
├── uv.lock                  # Deterministic dependency lockfile
└── README.md                # Project documentation and run guide
```

---

## 🛠️ Prerequisites

- **Python**: `>= 3.12`
- **uv**: Fast Python package manager ([Astral uv](https://docs.astral.sh/uv/))

To install `uv` (if not already installed):
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 🚀 Setup & Installation

1. **Clone the repository and enter the directory**:
   ```bash
   git clone <repo-url>
   cd "Assignment 2 Agix"
   ```

2. **Configure environment variables**:
   Copy the example environment template to `.env`:
   ```bash
   cp .env.example .env
   ```

3. **Install dependencies and create virtual environment**:
   Using `uv`, synchronize project dependencies from `uv.lock`:
   ```bash
   uv sync
   ```

---

## ▶️ Running the Application

Start the local development server with auto-reload:

```bash
uv run uvicorn app.main:app --reload --port 8000
```

The service will start at `http://127.0.0.1:8000`.

- **Swagger UI (Interactive API Docs)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc UI**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

All 3 endpoints can be tested directly from the Swagger UI at `/docs`.

---

## 🧪 API Endpoints & `curl` Examples

### 1. Health Check
Checks if the API service is up and operational.

- **URL**: `GET /health`
- **Status**: `200 OK`

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

**Response**:
```json
{
  "status": "ok"
}
```

---

### 2. Upload Document
Uploads a document (`.pdf`, `.txt`, `.docx`), validates its extension and byte size, and stores its metadata in memory with a unique UUID.

- **URL**: `POST /documents`
- **Content-Type**: `multipart/form-data`
- **Success Status**: `201 Created`

```bash
curl -X POST "http://127.0.0.1:8000/documents" \
  -F "file=@sample.txt"
```

**Response (`201 Created`)**:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "sample.txt",
  "size_bytes": 38,
  "uploaded_at": "2026-10-28T13:00:00.123456Z"
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
Returns an array of all uploaded documents recorded in memory.

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
    "filename": "sample.txt",
    "size_bytes": 38,
    "uploaded_at": "2026-10-28T13:00:00.123456Z"
  }
]
```

---

## 🧪 Running Automated Tests

Run the test suite using `pytest` inside the virtual environment:

```bash
uv run pytest -v
```

### Test Coverage Breakdown:
1. `test_health_check`: Verifies `GET /health` returns `200 OK` and `{"status": "ok"}`.
2. `test_upload_document_success`: Verifies valid `.txt` upload returns `201 Created` with expected fields and a valid UUID (`id`, `filename`, `size_bytes`, `uploaded_at`).
3. `test_upload_document_empty_file`: Verifies 0-byte file upload returns `400 Bad Request` with an informative error message.
4. `test_upload_document_wrong_file_type`: Verifies unsupported extensions (e.g., `.png`) return `415 Unsupported Media Type`.
5. `test_upload_document_file_too_large`: Verifies files exceeding 10 MB return `413 Content Too Large`.
6. `test_list_documents`: Verifies `GET /documents` initially returns `[]` and returns all uploaded document metadata with unique UUIDs.
7. `test_upload_docx_and_case_insensitive_extension`: Verifies uppercase extensions and Word `.docx` documents upload properly.

---

## 🔍 Code Explanation & Technical Design

### Pydantic Models (`app/models.py`)
- `HealthResponse`: Schema guaranteeing `{ "status": "ok" }`.
- `DocumentBase`: Defines core fields shared between models (`filename: str`, `size_bytes: int`).
- `DocumentCreate`: Schema representing document creation payload attributes.
- `DocumentResponse`: Inherits from `DocumentBase`, adding `id: str` (UUID v4) and `uploaded_at: datetime` (formatted in ISO 8601).
- `ErrorResponse`: Standardized schema for client error responses.

### Request Flow & Validation (`app/main.py`)
1. **Extension Checking**: `Path(file.filename).suffix.lower()` ensures case-insensitive extension checking against `ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}` before reading file data. Returns `415` if invalid.
2. **Size & Content Validation**: `await file.read()` reads file bytes. If `len(contents) == 0`, raises `400`. If `len(contents) > 10 * 1024 * 1024`, raises `413`.
3. **In-Memory Store**: `DOCUMENTS_DB: list[dict] = []` stores document metadata dictionaries. Each new entry receives a unique `str(uuid.uuid4())` identifier and `datetime.now(timezone.utc)`.
4. **Endpoint Modularity**: All 3 routes include OpenAPI summary, description, and status codes for interactive exploration in `/docs`.
