"""Persistence layer. SQLite locally for this submission (zero external
services to install); swap DATABASE_URL to a postgresql:// URL for the
proposal's production PostgreSQL target — SQLAlchemy makes that a one-line
change, no code changes needed elsewhere.
"""
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "helpdesk.db"
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    subject = Column(String(500))
    description = Column(Text, nullable=False)
    predicted_category = Column(String(100))
    predicted_priority = Column(String(50))
    category_confidence = Column(Float)
    priority_confidence = Column(Float)
    suggested_resolution = Column(Text)
    rag_backend = Column(String(50))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id = Column(Integer, nullable=False)
    agent_accepted = Column(String(10))  # "yes" / "no" / "edited"
    corrected_category = Column(String(100))
    corrected_priority = Column(String(50))
    corrected_resolution = Column(Text)
    notes = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db():
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
