"""Database engine and session management for AI Document Assistant."""

import logging
import os
from collections.abc import Generator

from dotenv import load_dotenv
from fastapi import HTTPException, status
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, create_engine

logger = logging.getLogger(__name__)

# Load environment variables from .env file if available
load_dotenv()

# Read the database address from DATABASE_URL in .env
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@127.0.0.1:5432/doc_assistant",
)

# SQLite requires check_same_thread=False; PostgreSQL uses fast connect timeout
is_sqlite = "sqlite" in DATABASE_URL
connect_args: dict = (
    {"check_same_thread": False}
    if is_sqlite
    else {"connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "3"))}
)

engine_kwargs: dict = {
    "connect_args": connect_args,
    "echo": False,
    "pool_pre_ping": True,
}

if not is_sqlite:
    engine_kwargs.update(
        {
            "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
            "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "20")),
            "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", "5")),
            "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", "1800")),
        }
    )

engine = create_engine(DATABASE_URL, **engine_kwargs)


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency yielding an active SQLModel database session."""
    try:
        with Session(engine) as session:
            yield session
    except OperationalError as exc:
        logger.error("Database connection failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable. Please ensure PostgreSQL is running.",
        ) from exc
