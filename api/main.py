"""
main.py — FastAPI server for the Complaint Clustering Engine (CCE).

Endpoints:
  POST /complaints/ingest           Bulk ingest texts -> cluster -> persist
  GET  /clusters                    List all active cluster summaries
  GET  /clusters/{id}/complaints    Drill into tickets for a specific cluster
  GET  /sandbox                     Paginated feed of unclassified complaints
  POST /sandbox/recluster           Sub-cluster sandbox records
  POST /model/retrain               Re-cluster all DB records with tuned HDBSCAN params
  POST /model/reset                 Reset in-memory model (retrain on next ingest)
"""
from __future__ import annotations

import sys
import os

# Ensure the project root is on the path so `core.*` resolves correctly
# when the server is launched from the api/ directory or from the root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()  # Load GROQ_API_KEY and other env vars from .env

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from api.database import (
    Cluster,
    Complaint,
    PipelineRun,
    create_tables,
    get_db,
    persist_pipeline_result,
)
from api.schemas import (
    ClustersResponse,
    ClusterSummary,
    ComplaintItem,
    ClusterComplaintsResponse,
    IngestRequest,
    IngestResponse,
    ReclusterRequest,
    ReclusterResponse,
    RetrainRequest,
    RetrainResponse,
    SandboxItem,
    SandboxResponse,
)
from core.pipeline import reset_model, run_pipeline, run_sandbox_recluster, set_engine_params


# ---------------------------------------------------------------------------
# Lifespan: initialise DB tables on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    yield


app = FastAPI(
    title="Complaint Clustering Engine API",
    description=(
        "Ingest unstructured customer complaints, cluster them semantically with HDBSCAN, "
        "label each cluster via Groq LLM, and isolate emerging trends in the Sandbox."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# POST /complaints/ingest
# ---------------------------------------------------------------------------

@app.post("/complaints/ingest", response_model=IngestResponse, tags=["Complaints"])
async def ingest_complaints(request: IngestRequest, db: Session = Depends(get_db)):
    """
    Accepts a list of raw complaint strings, runs the full ML pipeline
    (embed → cluster → label), persists the results, and returns a
    structured summary of cluster assignments.
    """
    try:
        result = run_pipeline(request.texts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    try:
        persist_pipeline_result(db, result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc

    # Fetch the run_id of the record just inserted
    run = db.query(PipelineRun).order_by(PipelineRun.id.desc()).first()

    total_classified = result["stats"]["classified"]

    summaries = [
        ClusterSummary(
            cluster_id=s["cluster_id"],
            theme_label=s["theme_label"],
            sample_count=s["count"],
            volume_pct=round((s["count"] / total_classified * 100) if total_classified else 0, 1),
            samples=s["samples"],
        )
        for s in result["cluster_summaries"]
    ]

    return IngestResponse(
        run_id=run.id if run else -1,
        total_ingested=result["stats"]["total"],
        classified_count=result["stats"]["classified"],
        sandbox_count=result["stats"]["sandbox"],
        cluster_summaries=summaries,
    )


# ---------------------------------------------------------------------------
# GET /clusters
# ---------------------------------------------------------------------------

@app.get("/clusters", response_model=ClustersResponse, tags=["Clusters"])
async def list_clusters(db: Session = Depends(get_db)):
    """Returns all active clusters with their theme labels and volume statistics."""
    clusters = db.query(Cluster).all()

    if not clusters:
        return ClustersResponse(total_clusters=0, clusters=[])

    total_complaints = (
        db.query(Complaint).filter(Complaint.is_sandbox == False).count()  # noqa: E712
    )

    summaries = []
    for c in clusters:
        # Fetch 3 sample texts from this cluster
        samples = (
            db.query(Complaint.text)
            .filter(Complaint.cluster_id == c.cluster_id, Complaint.is_sandbox == False)  # noqa: E712
            .limit(3)
            .all()
        )
        sample_texts = [row.text for row in samples]

        summaries.append(
            ClusterSummary(
                cluster_id=c.cluster_id,
                theme_label=c.theme_label,
                sample_count=c.sample_count,
                volume_pct=round((c.sample_count / total_complaints * 100) if total_complaints else 0, 1),
                samples=sample_texts,
            )
        )

    return ClustersResponse(total_clusters=len(summaries), clusters=summaries)


# ---------------------------------------------------------------------------
# GET /clusters/{cluster_id}/complaints
# ---------------------------------------------------------------------------

@app.get(
    "/clusters/{cluster_id}/complaints",
    response_model=ClusterComplaintsResponse,
    tags=["Clusters"],
)
async def get_cluster_complaints(
    cluster_id: int,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Returns a paginated list of every complaint text assigned to a specific cluster.
    Useful for staff to drill into the raw tickets behind a theme label.
    """
    cluster = db.get(Cluster, cluster_id)
    if not cluster:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found.")

    total = (
        db.query(Complaint)
        .filter(Complaint.cluster_id == cluster_id, Complaint.is_sandbox == False)  # noqa: E712
        .count()
    )
    offset = (page - 1) * page_size
    rows = (
        db.query(Complaint)
        .filter(Complaint.cluster_id == cluster_id, Complaint.is_sandbox == False)  # noqa: E712
        .order_by(Complaint.created_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    return ClusterComplaintsResponse(
        cluster_id=cluster_id,
        theme_label=cluster.theme_label,
        total=total,
        page=page,
        page_size=page_size,
        complaints=[
            ComplaintItem(id=r.id, text=r.text, created_at=r.created_at) for r in rows
        ],
    )



@app.get("/sandbox", response_model=SandboxResponse, tags=["Sandbox"])
async def get_sandbox(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Returns a paginated feed of unclassified high-novelty complaints."""
    total = db.query(Complaint).filter(Complaint.is_sandbox == True).count()  # noqa: E712
    offset = (page - 1) * page_size
    rows = (
        db.query(Complaint)
        .filter(Complaint.is_sandbox == True)  # noqa: E712
        .order_by(Complaint.created_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    items = [
        SandboxItem(id=r.id, text=r.text, created_at=r.created_at) for r in rows
    ]

    return SandboxResponse(
        total_sandbox=total,
        page=page,
        page_size=page_size,
        items=items,
    )


# ---------------------------------------------------------------------------
# POST /sandbox/recluster
# ---------------------------------------------------------------------------

@app.post("/sandbox/recluster", response_model=ReclusterResponse, tags=["Sandbox"])
async def recluster_sandbox(
    request: ReclusterRequest = ReclusterRequest(),
    db: Session = Depends(get_db),
):
    """
    Triggers a sub-clustering pass over all accumulated sandbox records.
    Newly formed clusters are persisted; records that remain outliers stay
    in the sandbox.
    """
    query = db.query(Complaint).filter(Complaint.is_sandbox == True)  # noqa: E712
    if request.max_sandbox_items:
        query = query.limit(request.max_sandbox_items)
    sandbox_rows = query.all()

    if not sandbox_rows:
        raise HTTPException(status_code=404, detail="No sandbox records found to recluster.")

    sandbox_texts = [r.text for r in sandbox_rows]

    try:
        result = run_sandbox_recluster(sandbox_texts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recluster error: {exc}") from exc

    # Mark newly classified items as production records in the DB
    newly_classified_texts = {r["text"] for r in result["production"]}
    for row in sandbox_rows:
        if row.text in newly_classified_texts:
            row.is_sandbox = False
            # Find which cluster it was assigned to
            match = next(
                (r for r in result["production"] if r["text"] == row.text), None
            )
            if match:
                row.cluster_id = match["cluster_id"]

    # Upsert new cluster records
    from api.database import Cluster as ClusterModel
    from datetime import datetime, timezone

    for summary in result.get("cluster_summaries", []):
        existing = db.get(ClusterModel, summary["cluster_id"])
        if existing:
            existing.sample_count += summary["count"]
            existing.theme_label = summary["theme_label"]
            existing.last_updated_at = datetime.now(timezone.utc)
        else:
            db.add(
                ClusterModel(
                    cluster_id=summary["cluster_id"],
                    theme_label=summary["theme_label"],
                    sample_count=summary["count"],
                )
            )

    db.commit()

    total_classified = result["stats"]["classified"]
    summaries = [
        ClusterSummary(
            cluster_id=s["cluster_id"],
            theme_label=s["theme_label"],
            sample_count=s["count"],
            volume_pct=round((s["count"] / total_classified * 100) if total_classified else 0, 1),
            samples=s["samples"],
        )
        for s in result["cluster_summaries"]
    ]

    return ReclusterResponse(
        new_clusters_formed=len(result["cluster_summaries"]),
        sandbox_remaining=result["stats"]["sandbox"],
        cluster_summaries=summaries,
    )


# ---------------------------------------------------------------------------
# POST /model/retrain
# ---------------------------------------------------------------------------

@app.post("/model/retrain", response_model=RetrainResponse, tags=["Admin"])
async def retrain_model(
    request: RetrainRequest = RetrainRequest(),
    db: Session = Depends(get_db),
):
    """
    Re-clusters ALL complaints already stored in the database using fresh HDBSCAN
    parameters. Useful when you want more/fewer clusters without re-ingesting data.

    Steps:
      1. Hot-swaps the HDBSCAN engine with the requested tuning params.
      2. Re-embeds all complaint texts via LM Studio.
      3. Refits HDBSCAN on all embeddings.
      4. Atomically updates cluster_id / is_sandbox for every complaint row.
      5. Rebuilds the clusters table with fresh Groq-generated theme labels.
    """
    from datetime import datetime, timezone
    import pandas as pd
    from core.pipeline import _get_components
    from api.database import Cluster as ClusterModel

    # Fetch every stored complaint text
    all_complaints = db.query(Complaint).all()
    if not all_complaints:
        raise HTTPException(
            status_code=404,
            detail="No complaints found in the database to retrain on.",
        )

    # 1. Hot-swap HDBSCAN engine with user-specified params
    set_engine_params(
        min_cluster_size=request.min_cluster_size,
        min_samples=request.min_samples,
        cluster_selection_method=request.cluster_selection_method,
    )

    vectorizer, engine, labeler = _get_components()

    # 2. Re-embed all complaint texts
    texts = [c.text for c in all_complaints]
    try:
        raw_df = pd.DataFrame({"complaint_text": texts})
        valid_df = vectorizer.clean_and_filter(raw_df, "complaint_text")
        embeddings = vectorizer.vectorize(valid_df["complaint_text"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding error: {exc}") from exc

    # 3. Fit fresh HDBSCAN
    try:
        labels = engine.fit_predict(embeddings)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Clustering error: {exc}") from exc

    production_df, sandbox_df = engine.isolate_sandbox(valid_df, labels)

    # 4. Update every complaint row in the DB atomically
    text_to_label: dict[str, int] = dict(
        zip(valid_df["complaint_text"].tolist(), [int(l) for l in labels.tolist()])
    )
    for complaint in all_complaints:
        new_label = text_to_label.get(complaint.text)
        if new_label is None:
            continue  # filtered out by clean_and_filter (too short)
        complaint.cluster_id = new_label
        complaint.is_sandbox = new_label == -1

    # 5. Rebuild clusters table
    db.query(ClusterModel).delete()
    now = datetime.now(timezone.utc)
    cluster_summaries_out = []
    for cluster_id, group in production_df.groupby("Cluster_ID"):
        samples = labeler.get_cluster_samples(group, "complaint_text", top_n=3)
        try:
            theme = labeler.generate_label(int(cluster_id), samples)
        except Exception:
            theme = f"Cluster {cluster_id}"
        db.add(ClusterModel(
            cluster_id=int(cluster_id),
            theme_label=theme,
            sample_count=len(group),
            last_updated_at=now,
        ))
        cluster_summaries_out.append(
            ClusterSummary(
                cluster_id=int(cluster_id),
                theme_label=theme,
                sample_count=len(group),
                volume_pct=round((len(group) / len(production_df) * 100) if len(production_df) else 0, 1),
                samples=samples,
            )
        )

    db.commit()

    return RetrainResponse(
        status="ok",
        total_retrained=len(valid_df),
        clusters_formed=len(cluster_summaries_out),
        sandbox_count=len(sandbox_df),
        cluster_summaries=cluster_summaries_out,
    )


# ---------------------------------------------------------------------------
# POST /model/reset
# ---------------------------------------------------------------------------

@app.post("/model/reset", tags=["Admin"])
async def reset_cluster_model():
    """
    Resets the in-memory HDBSCAN model. The next ingest call will trigger
    a full retrain on all provided texts.
    """
    reset_model()
    return {"status": "ok", "message": "HDBSCAN model reset. Next ingest will retrain from scratch."}



# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
