FROM python:3.12-slim

# Install uv binary from the official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory inside container
WORKDIR /app

# Enable bytecode compilation and use installed virtual environment
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PATH="/app/.venv/bin:$PATH"

# Copy dependency specifications first for layer caching
COPY pyproject.toml uv.lock ./

# Install production dependencies without dev tooling
RUN uv sync --frozen --no-dev

# Copy database migration files
COPY alembic.ini ./
COPY migrations ./migrations

# Copy application source code
COPY app ./app

# Expose port for FastAPI application
EXPOSE 8000

# Apply migrations on container startup then start Uvicorn server
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
