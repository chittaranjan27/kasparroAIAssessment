"""SQLAlchemy engine and session factory.

DATABASE_URL is read from the environment (required). No default is provided so
that a missing variable fails fast rather than silently using sqlite or similar.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()


@lru_cache(maxsize=1)
def _engine():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        echo=False,
    )


def get_engine():
    return _engine()


SessionLocal: sessionmaker[Session] = sessionmaker(
    autocommit=False, autoflush=False, bind=None
)


def get_session() -> Session:
    """Return a new session bound to the engine. Caller must close it."""
    engine = get_engine()
    SessionLocal.configure(bind=engine)
    return SessionLocal()


def create_tables() -> None:
    """Create all ORM tables if they do not exist (idempotent)."""
    from app.db.models import Base  # local import avoids circular deps at import time

    engine = get_engine()
    Base.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS "
                    "langfuse_trace_id VARCHAR(64)"
                )
            )
            connection.execute(
                text(
                    "ALTER TABLE answer_results ADD COLUMN IF NOT EXISTS "
                    "langfuse_trace_id VARCHAR(64)"
                )
            )
            connection.execute(
                text(
                    "ALTER TABLE answer_results ADD COLUMN IF NOT EXISTS "
                    "langfuse_observation_id VARCHAR(64)"
                )
            )
            connection.execute(
                text(
                    "ALTER TABLE answer_results ADD COLUMN IF NOT EXISTS "
                    "tool_call_details JSON NOT NULL DEFAULT '[]'::json"
                )
            )


def ping() -> bool:
    """Return True if the database is reachable."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
