"""
schemas.py — Pydantic request/response models for the CCE FastAPI server.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        description="One or more raw complaint strings to ingest and cluster.",
        examples=[["App keeps crashing", "Unauthorized charge on my account"]],
    )


class ReclusterRequest(BaseModel):
    max_sandbox_items: Optional[int] = Field(
        default=None,
        description="Cap the number of sandbox items to use. Defaults to all.",
    )


class RetrainRequest(BaseModel):
    min_cluster_size: int = Field(
        default=5,
        ge=2,
        description="Minimum complaints to form a cluster. Lower = more clusters.",
    )
    min_samples: int = Field(
        default=1,
        ge=1,
        description="HDBSCAN conservatism. 1 = most permissive (recommended for text).",
    )
    cluster_selection_method: Literal["leaf", "eom"] = Field(
        default="leaf",
        description="'leaf' = fine-grained clusters. 'eom' = fewer, merged clusters.",
    )


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class ClusterSummary(BaseModel):
    cluster_id: int
    theme_label: str
    sample_count: int
    volume_pct: float = Field(..., description="Percentage of total classified volume.")
    samples: list[str]


class SandboxItem(BaseModel):
    id: int
    text: str
    created_at: datetime


class IngestResponse(BaseModel):
    run_id: int
    total_ingested: int
    classified_count: int
    sandbox_count: int
    cluster_summaries: list[ClusterSummary]


class ClustersResponse(BaseModel):
    total_clusters: int
    clusters: list[ClusterSummary]


class SandboxResponse(BaseModel):
    total_sandbox: int
    page: int
    page_size: int
    items: list[SandboxItem]


class ComplaintItem(BaseModel):
    id: int
    text: str
    created_at: datetime


class ClusterComplaintsResponse(BaseModel):
    cluster_id: int
    theme_label: str
    total: int
    page: int
    page_size: int
    complaints: list[ComplaintItem]


class ReclusterResponse(BaseModel):
    new_clusters_formed: int
    sandbox_remaining: int
    cluster_summaries: list[ClusterSummary]


class RetrainResponse(BaseModel):
    status: str
    total_retrained: int
    clusters_formed: int
    sandbox_count: int
    cluster_summaries: list[ClusterSummary]
