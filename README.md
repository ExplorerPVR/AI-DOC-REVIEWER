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
- **Structured Resume Extraction (`POST /extract`)**:
  - Extracts biographical details and skills into a validated Pydantic `ResumeData` schema (`name`, `email`, `phone`, `skills`).
  - Uses native structured outputs (Gemini `response_schema`, Groq `response_format={"type": "json_object"}`).
  - Missing or unstated fields return `null`, never fabricated or hallucinated.
  - Resilient one-time retry with validation error feedback appended if initial output fails schema validation.
  - Prompt injection defense: resume text is quarantined inside `<resume_data>` tags and treated strictly as passive data.
  - Benchmark evaluation suite (`scripts/eval_extract.py`) reporting extraction validity and accuracy across 20 test resumes.
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
│   ├── extractor.py         # Structured resume extraction, retry logic & prompt injection defense
│   ├── llm.py               # Multi-provider streaming, retries, and token tracking
│   ├── main.py              # FastAPI app, route handlers & validation
│   ├── models.py            # SQLModel table models (Document, LLMCall) & schemas (ResumeData)
│   └── pricing.py           # Model pricing per million tokens & cost calculator
├── data/
│   └── resumes/             # 20 synthetic benchmark resumes and expected.json dataset
├── migrations/              # Alembic database migration scripts & versions
│   ├── versions/
│   │   ├── bb20b80bdc64_create_document_table.py
│   │   └── 43ce2c5fbd08_create_llm_calls_table.py
│   ├── env.py               # Alembic environment config (reads DATABASE_URL)
│   ├── README
│   └── script.py.mako
├── scripts/
│   └── eval_extract.py      # Automated benchmark evaluator across 20 resumes
├── tests/
│   ├── __init__.py
│   ├── test_ask.py          # Automated tests for LLM streaming, retries, and costs
│   ├── test_documents.py    # Automated tests for file upload, validation & DB persistence
│   └── test_extract.py      # Automated tests for structured extraction, retry & injection defense
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
  "uploaded_at": "2026-10-29T14:30:00.123456Z",
  "status": "pending"
}
```

Uploaded files are saved directly to the `uploads/` directory (mounted via Docker volume `uploads_data`). For PDFs, a background job (`FastAPI BackgroundTasks`) is queued immediately to extract text, chunk, embed, and store vectors in PostgreSQL without delaying the `201 Created` HTTP response.

#### Error Responses:
- **Empty File (`400 Bad Request`)**:
  ```json
  {
    "detail": "The uploaded file is empty or contains only whitespace."
  }
  ```
- **Blank Document (`400 Bad Request`)**:
  ```json
  {
    "detail": "The uploaded PDF contains no extractable text."
  }
  ```
- **Corrupted Document (`400 Bad Request`)**:
  ```json
  {
    "detail": "Corrupted file: invalid or unreadable PDF structure."
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
Fetches all stored document records from PostgreSQL, including their ingestion status.

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
    "uploaded_at": "2026-10-29T14:30:00.123456Z",
    "status": "ready"
  }
]
```

---

### 4. Get Document Details & Status
Inspect the current status (`pending`, `processing`, `ready`, or `failed`) and failure details.

- **URL**: `GET /documents/{document_id}`
- **Status**: `200 OK`

```bash
curl -X GET "http://127.0.0.1:8000/documents/550e8400-e29b-41d4-a716-446655440000"
```

---

### 5. Get Document Chunks & Embeddings
Inspect the extracted text chunks and vector embeddings generated for an ingested document.

- **URL**: `GET /documents/{document_id}/chunks`
- **Status**: `200 OK`

```bash
curl -X GET "http://127.0.0.1:8000/documents/550e8400-e29b-41d4-a716-446655440000/chunks"
```

---

### 6. Ask an LLM (Streaming Completion)
Streams answers from configured LLM providers (Gemini, Groq, Ollama) with automatic retries, timeouts, and call cost logging.

- **URL**: `POST /ask`
- **Content-Type**: `application/json`
- **Transfer-Encoding**: `chunked` (`text/plain` streaming by default, or Server-Sent Events with `"stream_format": "sse"` or `Accept: text/event-stream`)

```bash
# Ask Gemini (default plain text streaming)
curl -N -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the difference between SQL and NoSQL in two sentences?", "provider": "gemini"}'

# Ask with Server-Sent Events (SSE)
curl -N -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question": "Explain Docker containers briefly.", "provider": "groq", "stream_format": "sse"}'

# Ask Local Ollama
curl -N -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Hello from local LLM!", "provider": "ollama"}'
```

#### Error Handling & Citation Grounding Guarantees:
- **Strict Grounding Validation (`validate_before_stream: true`)**:
  - When enabled via payload (`"validate_before_stream": true`) or header (`X-Validate-Before-Stream: true`), the response is buffered in memory and completely verified before any token reaches the client.
  - If citation verification fails (e.g. fabricated citations like `[fake.pdf, p. 99]` or uncited claims), the hallucinated text is completely suppressed and replaced with `"I do not know based on the provided documents."` (zero token leakage).
- **Real-Time Streaming Grounding Defense (`validate_before_stream: false`)**:
  - If streaming in real time and inline citation validation fails at stream completion:
    - **Raw text mode**: Suppresses `Sources:` footer and appends an explicit revocation banner:
      `[VALIDATION_FAILED: Grounding validation failed. The response above contains unverified citations or ungrounded claims.]`
    - **SSE mode**: Suppresses `event: sources` / `event: done` and emits `event: error` with `{"error": "...", "validation_failed": true}`.
    - Prevents hallucinated or unverified responses from being mistaken for successful grounded answers.
- **Mid-Stream Provider Interruption Handling**: If a provider connection drops or fails midway after HTTP 200 headers have begun streaming:
  - **SSE mode**: Emits a structured `event: error` with `{"error": "<sanitized_error>", "partial": true}` and suppresses `event: sources` / `event: done`.
  - **Raw text mode**: Appends `\n\n[ERROR: Streaming generation interrupted: <sanitized_error>]\n` and suppresses the `Sources:` footer.
- **Rate Limit Exceeded (`503 Service Unavailable`)**: Returned if the LLM provider rate limits (429) persist after 3 retries with exponential backoff before the stream starts.
- **Gateway Timeout (`504 Gateway Timeout`)**: Returned if initial generation exceeds the timeout threshold across retries.
- **Unsupported Provider (`400 Bad Request`)**: Returned if a non-supported provider name is passed.

---

### 7. Structured Resume Extraction (`POST /extract`)
Extracts biographical details and technical skills into a validated `ResumeData` Pydantic schema using structured JSON output. If schema validation fails, it automatically retries once with error feedback appended to the prompt.

- **URL**: `POST /extract`
- **Content-Type**: `application/json`
- **Success Status**: `200 OK`

```bash
curl -X POST "http://127.0.0.1:8000/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Alice Johnson\nEmail: alice.johnson@example.com\nPhone: +1 (555) 123-4567\nTechnical Skills: Python, Django, FastAPI, PostgreSQL, Docker, Redis",
    "provider": "gemini"
  }'
```

**Response (`200 OK`)**:
```json
{
  "name": "Alice Johnson",
  "email": "alice.johnson@example.com",
  "phone": "+1 (555) 123-4567",
  "skills": [
    "Python",
    "Django",
    "FastAPI",
    "PostgreSQL",
    "Docker",
    "Redis"
  ]
}
```

#### Prompt Caching Cost Optimization:
On paid LLM APIs supporting prompt caching (such as Anthropic Claude or Google Gemini), prompt caching cuts input token costs by up to 75% to 90% and reduces response latency by up to 80%.
Because our extraction system prompt, rules, schema instructions, and few-shot example are fixed and placed at the beginning of the prompt as a static prefix, the LLM provider caches these input tokens after the first request.
Subsequent calls to `POST /extract` only bill the cached prefix at a fraction of the standard input price (e.g., $0.30/MTok vs $3.00/MTok on Claude), paying full price only for the variable candidate resume text at the end.

#### Error Handling:
- **Validation Failure (`422 Unprocessable Entity`)**: Returned if the LLM response fails Pydantic schema validation after two attempts (initial + retry).
- **Empty Resume (`400 Bad Request`)**: Returned if the `text` field is empty or missing.
- **Upstream Errors (`503 / 504`)**: Returned if the LLM is unavailable or times out.

---

## ⚡ Background PDF Ingestion & pgvector Chunks

When a PDF document is uploaded via `POST /documents`:
1. **File Storage**: The file is stored in `uploads/` (persisted across restarts via Docker volume `uploads_data`).
2. **Immediate 201 Response**: The HTTP endpoint creates the Document record with `status: "pending"` and returns immediately.
3. **Background Ingestion Worker**: A FastAPI `BackgroundTask` executes asynchronously:
   - Reads the PDF page-by-page using **PyMuPDF**.
   - Chunks the text into overlapping segments of ~500 to 1,000 characters (preserving original `filename` and 1-based `page_number`).
   - Generates 768-dimensional dense vector embeddings using Google Gemini (`gemini-embedding-001` / `text-embedding-004`) or local Ollama (`nomic-embed-text`).
   - Persists each chunk into the PostgreSQL `chunks` table utilizing the **pgvector** `Vector(768)` data type.
   - Transitions `Document.status` to `ready`.
4. **Resilience**: If unparseable or corrupted PDF data is encountered, the document status transitions to `failed` with descriptive `error_message`, ensuring zero server crashes.

---

## 📊 Resume Extraction Evaluation & Benchmark (`scripts/eval_extract.py`)

A benchmark suite evaluating candidate resumes in `data/resumes/` against ground truth annotations in `data/resumes/expected.json`. The dataset contains **25 resumes**:
- **20 Base Resumes**: Standard clean formats evaluating standard fields, null normalization, and prompt injection defense.
- **5 Challenging Resumes (`resume_21` to `resume_25`)**: Real-world messy layouts (ASCII/pipe tables), missing contact fields, technical skills dispersed across job experience/narrative paragraphs (no explicit "Skills:" header), non-standard punctuation/headers, and distractor references.

The evaluation script reports both **Strict** (exact set equality on skills, exact digits, exact casing) and **Lenient** (Jaccard similarity $\ge 0.70$ / subset on skills, tolerant title matching, normalized phone digits) scores:

```bash
# Run evaluation across all 25 resumes
uv run python scripts/eval_extract.py --provider groq --pause 1.0
```

### Benchmark Scores:
```text
Score (Strict):  25/25 valid, 98/100 fields correct
Score (Lenient): 25/25 valid, 100/100 fields correct
```
- **Validity Rate**: 100% (25/25 valid JSON responses conforming strictly to `ResumeData`).
- **Strict Accuracy**: 98.0% (98/100 fields correct; Name: 25/25, Email: 25/25, Phone: 25/25, Skills Exact Set: 23/25).
- **Lenient Accuracy**: 100.0% (100/100 fields correct with Jaccard $\ge 0.70$ / subset matching on dispersed skills).
- **Prompt Injection Defense**: Successfully resisted adversarial prompt injection attacks (`resume_13.txt`), correctly extracting real candidate details without altering output.

---

## 🔍 Semantic Search & Retrieval (`GET /search`)

The semantic search endpoint finds the parts of ingested documents that best match a user query, forming the core Retrieval component ("R") for Retrieval-Augmented Generation (RAG).

### Endpoint Specification:
- **Route**: `GET /search?q={query}&k={count}`
- **Parameters**:
  - `q` (string, required): Natural language query question (`min_length=1`). Empty or whitespace-only queries return `422 Unprocessable Content`.
  - `k` (integer, optional, default `5`): Maximum number of closest chunks to retrieve (`1 <= k <= 100`).
- **Matching Mechanism**:
  - Embeds the query `q` using the same dense 768-dimensional model as ingestion (Google Gemini or Ollama).
  - Queries the PostgreSQL `chunks` table using **pgvector's `<=>` cosine distance operator**, filtered strictly to documents with `status = 'ready'`.
  - Leverages the HNSW vector index (`ix_chunks_embedding_hnsw`) for fast nearest-neighbor lookups.
  - Returns chunks ordered **closest first** (ascending cosine distance).
- **Response Schema (`list[SearchResult]`)**:
  ```json
  [
    {
      "text": "MAE addressed this by masking a very high proportion of patches...",
      "filename": "Self_Supervised_Learning_Review_Paper.pdf",
      "page": 3,
      "score": 0.803
    }
  ]
  ```
- **Edge Cases**:
  - Empty or missing `q`: Returns `422 Unprocessable Content`.
  - No documents with `status: "ready"`: Returns `[]` (empty list).

---

## 📈 Search Accuracy Evaluation (`scripts/eval_search.py`)

A quantitative evaluation suite running 5 ground-truth questions against the ingested academic review paper (*Self-Supervised Learning Review Paper*), verifying whether the correct target page appears within the top-5 retrieved chunks.

```bash
# Run the search evaluation script
uv run python scripts/eval_search.py
```

### 5-Question Evaluation Results:

| ID | Test Question | Expected Page | Top-5 Retrieved Pages | Top Score | Result |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Q1** | *How does the context-free network solve jigsaw puzzles in self-supervised learning?* | **Page 2** | `[2, 3, 6, 3, 2]` | 0.797 | **PASS (Rank 1)** |
| **Q2** | *What are Masked Autoencoders and how does patch masking scale in vision transformers?* | **Page 3** | `[3, 2, 6, 1, 3]` | 0.803 | **PASS (Rank 1)** |
| **Q3** | *What key milestones and architectures are illustrated in the representation learning timeline?* | **Page 4** | `[4, 1, 5, 1, 1]` | 0.741 | **PASS (Rank 1)** |
| **Q4** | *How do DINOv2 frozen features perform on semantic segmentation and monocular depth estimation?* | **Page 5** | `[5, 3, 3, 5, 6]` | 0.788 | **PASS (Rank 1)** |
| **Q5** | *Why do latent-space predictive architectures like I-JEPA deserve further study in future directions?* | **Page 6** | `[6, 3, 6, 2, 1]` | 0.753 | **PASS (Rank 1)** |

```text
5/5 found in top 5 (100.0% accuracy, all Rank 1)
```

---

## 🧠 Grounded Document Q&A / RAG (`POST /ask`)

`POST /ask` implements Retrieval-Augmented Generation (RAG) grounded strictly on your ingested documents:

- **Request**: `{"question": "...", "provider": "gemini" | "groq" | "ollama", "k": 5}`
- **Retrieval & Relevance Gate**:
  - Retrieves the top-`k` chunks from `READY` documents using pgvector cosine similarity (`search_chunks`).
  - Filters chunks against `RAG_MIN_SIMILARITY_SCORE` (default `0.55`).
  - If no documents are ready or no chunks meet the relevance floor, `/ask` immediately returns `"I don't know..."` without invoking the LLM.
- **Strict Grounding & Citations**:
  - Injects retrieved chunks with their file name and page number into a prompt-injection-resistant prompt (`app/rag.py`).
  - Answers include inline `[filename, p. N]` citations plus a deduplicated `Sources:` footer (`- filename (page N)`).
  - If the retrieved excerpts still do not contain the answer, the model responds `"I don't know."` and the `Sources:` footer is suppressed.

---

## 🛡️ Security Guardrails & Defense Against Injections (`app/guardrails.py`)

Multi-layer guardrail defense against prompt injections, adversarial overrides, and personal data (PII) leakage:

1. **Input Prompt Injection Guardrail**:
   - Blocks direct instruction overrides (*"ignore all previous instructions"*, *"override rules"*).
   - Blocks system prompt extraction probes (*"repeat your system prompt verbatim"*, *"show initial rules"*).
   - Blocks persona jailbreaks (*"you are now DAN"*, *"developer mode enabled"*).
   - Neutralizes synthetic delimiter breakout attempts (`</question><context>...`).
   - Short-circuits with clear refusal message without calling the LLM.
2. **PII Protection & Harvesting Guardrail**:
   - Blocks queries attempting bulk extraction or harvesting of personal contact details (*"dump all phone numbers and emails"*).
   - Redacts sensitive PII (emails, phone numbers, SSNs, credit cards) across streamed outputs.
3. **Output Confidentiality & Anti-Leakage**:
   - Detects and suppresses responses containing internal system rules or instructions.
   - Pinned greedy decoding ($T=0.0$) across all providers to eliminate stochastic jailbreak variance.
   - Citation validation (`validate_inline_citations`) and pre-stream validation (`validate_before_stream: true`) to prevent fabricated citations.

---

## 🔭 Monitoring & Observability with Langfuse (`app/monitoring.py`)

Every request to `/ask` and `/extract` is instrumented with **Langfuse**:
- Traces execution duration, input prompt, completion, and model parameters.
- Records token usage (`input_tokens`, `output_tokens`, `total_tokens`) and USD cost calculated against model pricing tables.
- Seamless fallback: Operates silently and gracefully when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are not set.

```bash
# Configure in .env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## 🏆 Benchmark & RAGAS Leaderboard Evaluation (`scripts/eval_ragas.py`)

A 25-question benchmark dataset (`data/benchmark_25_questions.json`) evaluating:
- **10 Grounded Factual Questions**: Tested for inline citations and factual accuracy.
- **5 Out-of-Context Questions**: Tested for correct refusal (`"I do not know"`).
- **6 Adversarial & Injection Attacks**: Tested for resistance against delimiter smuggling, jailbreaks, and prompt extraction.
- **4 PII Harvesting Attacks**: Tested for personal data protection.

```bash
# Run RAGAS and LLM-as-a-judge evaluation benchmark
uv run python scripts/eval_ragas.py
```

### Leaderboard Results (Before Fixes vs After Fixes):

| Metric | Before Fixes | After Fixes | Improvement (Delta) |
| :--- | :---: | :---: | :---: |
| **RAGAS Faithfulness / Groundedness** | 20.0% | **100.0%** | **+80.0%** |
| **RAGAS Answer Relevance** | 40.0% | **100.0%** | **+60.0%** |
| **Prompt Injection Defense Rate** | 76.0% | **100.0%** | **+24.0%** |
| **PII Data Leak Protection Rate** | 84.0% | **100.0%** | **+16.0%** |
| **OVERALL LEADERBOARD SCORE** | 55.0% | **100.0%** | **+45.0%** |

Detailed JSON results saved to `data/leaderboard_scores.json`.

---

## 🤖 Continuous Integration (CI) Checks

The CI pipeline runs automatically on all pushes and pull requests via GitHub Actions (`.github/workflows/ci.yml`).

### Workflow Steps:
1. **Lint & Code Style**: Runs `uv run ruff check .` to check for style violations, unformatted imports, and common programming errors.
2. **Database Migrations & Automated Testing**: Runs `uv run alembic upgrade head` against a `pgvector/pgvector:pg16` service container and executes the full test suite (75 automated tests, including a native PostgreSQL `pgvector` `<=>` integration test).

### Running CI Checks Locally:
```bash
# Run Ruff linting check
uv run ruff check .

# Run Pytest suite
uv run pytest -v
```
