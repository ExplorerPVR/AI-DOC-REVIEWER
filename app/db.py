"""Database engine, session management, and table initialization."""

from collections.abc import Generator
import os
import time
from dotenv import load_dotenv
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, SQLModel, create_engine

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


def init_db(engine_override=None, retries: int = 5, delay: float = 2.0) -> None:
    """Initialize database tables with automatic connection retry logic for Docker startup."""
    target_engine = engine_override or engine
    for attempt in range(1, retries + 1):
        try:
            SQLModel.metadata.create_all(target_engine)
            return
        except OperationalError:
            if attempt == retries:
                raise
            time.sleep(delay)


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency yielding an active SQLModel database session."""
    with Session(engine) as session:
        yield session
