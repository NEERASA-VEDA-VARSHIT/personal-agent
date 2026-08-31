"""Database session management and initialization."""

from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

# Database URL - SQLite for local dev, easily swappable
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./memory.db")

# Create engine with SQLite-specific settings
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=os.getenv("SQL_ECHO", "false").lower() == "true",
    )
else:
    # For production PostgreSQL
    engine = create_engine(
        DATABASE_URL,
        echo=os.getenv("SQL_ECHO", "false").lower() == "true",
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initialize the database (create all tables)."""
    Base.metadata.create_all(bind=engine)


def get_session() -> Generator[Session, None, None]:
    """Dependency for FastAPI to inject a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
