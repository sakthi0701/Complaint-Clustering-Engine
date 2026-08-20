"""
database.py — SQLAlchemy SQLite persistence layer for CCE.

Tables:
  - complaints      Raw ingested texts with cluster assignment and sandbox flag.
  - clusters        Active cluster registry with theme labels and metadata.
  - pipeline_runs   Audit log of each pipeline execution.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ---------------------------------------------------------------------------
# Database path — stored alongside the api package for portability
# ---------------------------------------------------------------------------
_DB_PATH = os.environ.get(
    "CCE_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "cce_data.db"),
)
DATABASE_URL = f"sqlite:///{os.path.abspath(_DB_PATH)}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite + FastAPI threading
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class Complaint(Base):
    __tablename__ = "complaints"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    text = Column(Text, nullable=False)
    cluster_id = Column(Integer, nullable=True)       # NULL if not yet processed
    is_sandbox = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Cluster(Base):
    __tablename__ = "clusters"

    cluster_id = Column(Integer, primary_key=True)
    theme_label = Column(String(120), nullable=False)
    sample_count = Column(Integer, default=0)
    last_updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    total_ingested = Column(Integer, default=0)
    classified_count = Column(Integer, default=0)
    sandbox_count = Column(Integer, default=0)
    new_clusters_formed = Column(Integer, default=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_tables() -> None:
    """Create all tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a DB session and ensures it is closed."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def persist_pipeline_result(db: Session, result: dict) -> None:
    """
    Persists a pipeline result dict (as returned by core.pipeline.run_pipeline)
    into the database.  Updates cluster records and inserts complaint rows.
    """
    now = datetime.now(timezone.utc)

    # Upsert cluster records
    for summary in result.get("cluster_summaries", []):
        existing = db.get(Cluster, summary["cluster_id"])
        if existing:
            existing.sample_count += summary["count"]
            existing.theme_label = summary["theme_label"]
            existing.last_updated_at = now
        else:
            db.add(
                Cluster(
                    cluster_id=summary["cluster_id"],
                    theme_label=summary["theme_label"],
                    sample_count=summary["count"],
                    last_updated_at=now,
                )
            )

    # Insert production complaints
    for record in result.get("production", []):
        db.add(
            Complaint(
                text=record["text"],
                cluster_id=record["cluster_id"],
                is_sandbox=False,
                created_at=now,
            )
        )

    # Insert sandbox complaints
    for record in result.get("sandbox", []):
        db.add(
            Complaint(
                text=record["text"],
                cluster_id=-1,
                is_sandbox=True,
                created_at=now,
            )
        )

    # Log the run
    stats = result.get("stats", {})
    db.add(
        PipelineRun(
            run_at=now,
            total_ingested=stats.get("total", 0),
            classified_count=stats.get("classified", 0),
            sandbox_count=stats.get("sandbox", 0),
            new_clusters_formed=len(result.get("cluster_summaries", [])),
        )
    )

    db.commit()
