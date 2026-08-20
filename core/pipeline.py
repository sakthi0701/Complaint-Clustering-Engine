"""
pipeline.py — Orchestrator for the CCE ML pipeline.

Responsibilities:
  1. Vectorize raw complaint texts via LM Studio.
  2. Cluster embeddings with HDBSCAN (fit or approximate-predict).
  3. Isolate sandbox (cluster -1) from production clusters.
  4. Label each cluster via Groq LLM.
  5. Return a structured PipelineResult dict ready for API persistence.

Model state is held in module-level singletons so the HDBSCAN model
is fitted once per server process and reused for all subsequent ingestions.
Call `reset_model()` to force a full retrain on the next ingest.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from core.vectorizer import LMStudioVectorizer
from core.clustering_engine import ClusteringEngine
from core.labeler import ThemeLabeler

# ---------------------------------------------------------------------------
# Module-level singleton state (survives across API requests)
# ---------------------------------------------------------------------------
_vectorizer: LMStudioVectorizer | None = None
_engine: ClusteringEngine | None = None
_labeler: ThemeLabeler | None = None


def _get_components() -> tuple[LMStudioVectorizer, ClusteringEngine, ThemeLabeler]:
    """Lazy-initialise the three pipeline components with env-configurable HDBSCAN params."""
    global _vectorizer, _engine, _labeler
    if _vectorizer is None:
        _vectorizer = LMStudioVectorizer()
    if _engine is None:
        min_size = int(os.environ.get("MIN_CLUSTER_SIZE", 5))
        min_samples = int(os.environ.get("MIN_SAMPLES", 1))
        method = os.environ.get("CLUSTER_SELECTION_METHOD", "leaf")
        _engine = ClusteringEngine(
            min_cluster_size=min_size,
            min_samples=min_samples,
            cluster_selection_method=method,
        )
    if _labeler is None:
        _labeler = ThemeLabeler()
    return _vectorizer, _engine, _labeler


def reset_model() -> None:
    """Force a full HDBSCAN retrain on the next pipeline run."""
    global _engine
    _engine = None


def set_engine_params(
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
) -> None:
    """
    Hot-swap HDBSCAN parameters without a server restart.
    Replaces the in-memory engine singleton with a fresh unfitted instance.
    Call this before a retrain to apply new tuning settings.
    """
    global _engine
    _engine = ClusteringEngine(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
    )



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(texts: list[str]) -> dict:
    """
    End-to-end pipeline run on a list of raw complaint strings.

    Returns a dict with:
      {
        "production": [{"text": str, "cluster_id": int}, ...],
        "sandbox":    [{"text": str}, ...],
        "cluster_summaries": [
          {"cluster_id": int, "theme_label": str, "count": int, "samples": [str, ...]},
          ...
        ],
        "stats": {"total": int, "classified": int, "sandbox": int},
      }
    """
    vectorizer, engine, labeler = _get_components()

    # 1. Build a DataFrame and clean it
    raw_df = pd.DataFrame({"complaint_text": texts})
    valid_df = vectorizer.clean_and_filter(raw_df, text_column="complaint_text")

    if valid_df.empty:
        return {
            "production": [],
            "sandbox": [],
            "cluster_summaries": [],
            "stats": {"total": len(texts), "classified": 0, "sandbox": 0},
        }

    # 2. Embed
    embeddings = vectorizer.vectorize(valid_df["complaint_text"])

    # 3. Cluster — fit if first run, approximate predict otherwise
    if engine.is_fitted:
        labels = engine.predict_new(embeddings)
    else:
        labels = engine.fit_predict(embeddings)

    # 4. Isolate sandbox
    production_df, sandbox_df = engine.isolate_sandbox(valid_df, labels)

    # 5. Build production records
    production_records = production_df[["complaint_text", "Cluster_ID"]].rename(
        columns={"complaint_text": "text", "Cluster_ID": "cluster_id"}
    ).to_dict(orient="records")

    sandbox_records = [{"text": t} for t in sandbox_df["complaint_text"].tolist()]

    # 6. Label each unique cluster
    cluster_summaries = []
    for cluster_id, group in production_df.groupby("Cluster_ID"):
        samples = labeler.get_cluster_samples(group, "complaint_text", top_n=3)
        theme = labeler.generate_label(int(cluster_id), samples)
        cluster_summaries.append(
            {
                "cluster_id": int(cluster_id),
                "theme_label": theme,
                "count": len(group),
                "samples": samples,
            }
        )

    total = len(valid_df)
    return {
        "production": production_records,
        "sandbox": sandbox_records,
        "cluster_summaries": cluster_summaries,
        "stats": {
            "total": total,
            "classified": len(production_df),
            "sandbox": len(sandbox_df),
        },
    }


def run_sandbox_recluster(sandbox_texts: list[str]) -> dict:
    """
    Sub-clustering pass over accumulated sandbox records.
    Always forces a fresh HDBSCAN fit on the sandbox subset.

    Returns same structure as run_pipeline().
    """
    if not sandbox_texts:
        return {
            "production": [],
            "sandbox": [],
            "cluster_summaries": [],
            "stats": {"total": 0, "classified": 0, "sandbox": 0},
        }

    vectorizer, _, labeler = _get_components()

    # Fresh mini-engine just for the sandbox subset
    min_size = max(2, len(sandbox_texts) // 5)
    sandbox_engine = ClusteringEngine(min_cluster_size=min_size)

    raw_df = pd.DataFrame({"complaint_text": sandbox_texts})
    valid_df = vectorizer.clean_and_filter(raw_df, text_column="complaint_text")
    embeddings = vectorizer.vectorize(valid_df["complaint_text"])
    labels = sandbox_engine.fit_predict(embeddings)

    production_df, new_sandbox_df = sandbox_engine.isolate_sandbox(valid_df, labels)

    production_records = production_df[["complaint_text", "Cluster_ID"]].rename(
        columns={"complaint_text": "text", "Cluster_ID": "cluster_id"}
    ).to_dict(orient="records")

    sandbox_records = [{"text": t} for t in new_sandbox_df["complaint_text"].tolist()]

    cluster_summaries = []
    for cluster_id, group in production_df.groupby("Cluster_ID"):
        samples = labeler.get_cluster_samples(group, "complaint_text", top_n=3)
        theme = labeler.generate_label(int(cluster_id), samples)
        cluster_summaries.append(
            {
                "cluster_id": int(cluster_id),
                "theme_label": theme,
                "count": len(group),
                "samples": samples,
            }
        )

    return {
        "production": production_records,
        "sandbox": sandbox_records,
        "cluster_summaries": cluster_summaries,
        "stats": {
            "total": len(valid_df),
            "classified": len(production_df),
            "sandbox": len(new_sandbox_df),
        },
    }
