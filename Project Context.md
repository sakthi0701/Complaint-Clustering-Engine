# Project Context: Complaint Clustering Engine (CCE)

_Last updated: 2026-08-15. This document is the single source of truth for any AI assistant joining this project._

---

## 1. Executive Summary

- **Domain:** Unsupervised NLP, Text Analytics & Customer Operations
- **Core Mission:** Ingest unstructured, unlabeled customer complaint texts, group them into semantic clusters, assign human-readable business category themes via LLM, and automatically isolate emerging/novel issues into a "Sandbox" for further review.
- **Current Status:** Phases 1, 2, and 3 are fully implemented and running.

---

## 2. Tech Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | `all-MiniLM-L6-v2` via LM Studio (OpenAI-compatible local endpoint at `http://127.0.0.1:1234/v1`) |
| Clustering | HDBSCAN (`hdbscan` Python package) with L2-normalized embeddings |
| LLM Labeling | Groq API — model `openai/gpt-oss-120b` (key in `.env` as `GROQ_API_KEY`) |
| API Backend | FastAPI + Uvicorn |
| Database | SQLite via SQLAlchemy (file: `cce_data.db` in project root) |
| Frontend | Streamlit (`frontend/app.py`) calls FastAPI over HTTP via `httpx` |
| Env vars | `python-dotenv` — `.env` file in project root |

---

## 3. Directory Structure

```
CCE/
├── .env                          # GROQ_API_KEY=... (required)
├── cce_data.db                   # SQLite database (auto-created on server start)
├── requirements.txt
├── main.py                       # Dev CLI runner (Phase 1 smoke test only)
│
├── core/
│   ├── __init__.py
│   ├── vectorizer.py             # LM Studio embedding client (LMStudioVectorizer)
│   ├── clustering_engine.py      # HDBSCAN engine with L2 normalization
│   ├── labeler.py                # Groq API theme labeler (ThemeLabeler)
│   └── pipeline.py               # Orchestrator: vectorize → cluster → label
│                                 # Holds module-level singletons for engine state
│
├── api/
│   ├── __init__.py
│   ├── database.py               # SQLAlchemy models: complaints, clusters, pipeline_runs
│   ├── schemas.py                # Pydantic request/response models
│   └── main.py                   # FastAPI server — all endpoints defined here
│
└── frontend/
    └── app.py                    # Streamlit staff dashboard
```

---

## 4. Key Design Decisions

### A. HDBSCAN Clustering
- **Best params for ~200 complaints:** `min_cluster_size=3-5`, `min_samples=1`, `cluster_selection_method='leaf'`
- `min_samples=1` is critical — when defaulted to `min_cluster_size`, HDBSCAN is too conservative and produces only 2-3 clusters.
- `cluster_selection_method='leaf'` gives finer-grained clusters than `'eom'`.
- All embeddings are **L2-normalized** before clustering so euclidean distance approximates cosine similarity.
- The HDBSCAN engine is a **module-level singleton** in `core/pipeline.py`. The first ingest call fits the model; subsequent calls use `approximate_predict`.

### B. Groq Labeling
- **Model:** `openai/gpt-oss-120b`
- **Critical:** Do NOT set `reasoning_effort` for label generation. This model is a reasoning model and the thinking tokens consume the token budget, leaving zero tokens for the actual label. Use `max_completion_tokens=256` minimum.
- **Streaming:** Enabled. Chunks are buffered into a single string.
- **Fallback:** If Groq fails, label defaults to `f"Cluster {cluster_id}"` — pipeline never crashes.

### C. Fit vs. Predict
- **First ingest:** `fit_predict()` — trains HDBSCAN from scratch.
- **Subsequent ingests:** `approximate_predict()` — fast, no retraining.
- **Retrain endpoint (`POST /model/retrain`):** Re-embeds all DB records and refits HDBSCAN with tunable params. Updates all DB rows atomically. Use this to change cluster count without re-ingesting.

### D. Sandbox (Cluster -1)
- Complaints not fitting any cluster density get `is_sandbox=True` in the DB.
- `POST /sandbox/recluster` runs a fresh HDBSCAN fit on sandbox records only.
- Staff can view in the Emerging Radar dashboard panel.

---

## 5. API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/complaints/ingest` | Bulk ingest complaint texts, run pipeline, persist |
| GET | `/clusters` | All active clusters with theme + volume stats |
| GET | `/clusters/{id}/complaints` | Paginated complaints for a specific cluster (page_size up to 100) |
| GET | `/sandbox` | Paginated feed of unclassified complaints |
| POST | `/sandbox/recluster` | Sub-cluster sandbox records |
| POST | `/model/retrain` | Re-cluster all DB data with new HDBSCAN params (body: `min_cluster_size`, `min_samples`, `cluster_selection_method`) |
| POST | `/model/reset` | Clear in-memory HDBSCAN model (next ingest retrains) |

Swagger UI: `http://localhost:8000/docs`

---

## 6. Database Schema (SQLite — `cce_data.db`)

### `complaints`
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | Auto-increment |
| text | TEXT | Raw complaint string |
| cluster_id | INTEGER | -1 = sandbox, NULL = not yet processed |
| is_sandbox | BOOLEAN | True if cluster_id == -1 |
| created_at | DATETIME | UTC |

### `clusters`
| Column | Type | Notes |
|--------|------|-------|
| cluster_id | INTEGER PK | HDBSCAN-assigned ID |
| theme_label | TEXT | Groq-generated 3-5 word label |
| sample_count | INTEGER | Running total of complaints in this cluster |
| last_updated_at | DATETIME | UTC |

### `pipeline_runs`
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | Auto-increment |
| run_at | DATETIME | UTC |
| total_ingested | INTEGER | |
| classified_count | INTEGER | |
| sandbox_count | INTEGER | |
| new_clusters_formed | INTEGER | |

---

## 7. Streamlit Dashboard — `frontend/app.py`

Three navigation panels (sidebar):

1. **Cluster Explorer** — Cards per cluster showing Groq theme label, volume bar, sample pills. Expandable complaint drill-down with **paginated load-more** (10 at a time via `st.session_state`).
2. **Emerging Radar** — Paginated live feed of sandbox (unclassified) complaints.
3. **Ingest & Control** — Three tabs:
   - `Ingest Complaints` — text area → POST `/complaints/ingest`
   - `Re-Cluster Sandbox` — POST `/sandbox/recluster`
   - `Model Admin` — HDBSCAN sliders (`min_cluster_size`, `min_samples`, method radio) → POST `/model/retrain`. Also has reset button.

Dashboard calls the FastAPI server at `http://localhost:8000` (configurable via `CCE_API_URL` env var).

---

## 8. How to Run

### Start the API server
```powershell
cd c:\Users\sakth\Desktop\Projects\CCE
.venv\Scripts\uvicorn api.main:app --reload --port 8000
```

### Start the dashboard
```powershell
cd c:\Users\sakth\Desktop\Projects\CCE
.venv\Scripts\streamlit run frontend/app.py
```

LM Studio must be running with `all-MiniLM-L6-v2` loaded and its server on `http://127.0.0.1:1234`.

### Reset the database (start fresh)
```powershell
# Stop both servers first (Ctrl+C), then:
Remove-Item cce_data.db -ErrorAction SilentlyContinue
# Restart the API server — tables are auto-created on startup
.venv\Scripts\uvicorn api.main:app --reload --port 8000
```

---

## 9. Execution Roadmap

```
[x] Phase 1: Core ML Pipeline
    [x] vectorizer.py — LM Studio embedding client
    [x] clustering_engine.py — HDBSCAN with L2 normalization, min_samples=1, leaf method
    [x] labeler.py — Groq API theme labeler (streaming, no reasoning_effort)
    [x] pipeline.py — Singleton orchestrator

[x] Phase 2: Backend API & Storage
    [x] FastAPI server with 7 endpoints
    [x] SQLite persistence via SQLAlchemy
    [x] Pydantic schemas

[x] Phase 3: Staff Dashboard
    [x] Cluster Explorer with paginated complaint drill-down
    [x] Emerging Radar (sandbox live feed)
    [x] Ingest & Control panel
    [x] Model Admin with HDBSCAN tuning + retrain

[ ] Phase 4: Production Hardening
    [ ] Auto-recluster when sandbox count exceeds threshold N
    [ ] Model drift and cluster convergence checks
    [ ] Embedding caching (avoid re-embedding on retrain)
    [ ] Authentication for the staff dashboard
```

---

## 10. Known Issues & Gotchas

1. **Groq `reasoning_effort` + low `max_completion_tokens`** — causes empty labels. Always omit `reasoning_effort` for labeling. Fixed in `core/labeler.py`.
2. **HDBSCAN with default `min_samples`** — defaults to `min_cluster_size`, making clustering too conservative (produces 2-3 clusters for 200 complaints). Always set `min_samples=1` explicitly.
3. **DB and server must both be restarted for a clean reset** — SQLite file locks prevent deletion while the server holds a connection.
4. **IDE lint error for `hdbscan`** — false positive. The IDE resolves against system Python, not `.venv`. Set IDE interpreter to `.venv\Scripts\python.exe`.
5. **Embedding re-computation on retrain** — `/model/retrain` re-embeds all complaints via LM Studio. For 200 complaints this takes ~2-3 minutes. Future improvement: cache embeddings in the DB.