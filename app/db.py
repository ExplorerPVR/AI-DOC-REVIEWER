"""Database engine and session management for AI Document Assistant."""

import os
from collections.abc import Generator

from dotenv import load_dotenv
from sqlmodel import Session, create_engine

# Load environment variables from .env file if available
load_dotenv()

# Read the database address from DATABASE_URL in .env
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/doc_assistant",
)

# SQLite requires check_same_thread=False; PostgreSQL does not
connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False,
    pool_pre_ping=True,
)


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency yielding an active SQLModel database session."""
    with Session(engine) as session:
        yield session
