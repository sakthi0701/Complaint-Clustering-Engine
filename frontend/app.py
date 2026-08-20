"""
app.py — CCE Staff Dashboard (Streamlit)

Three panels:
  1. Cluster Explorer   — Live cluster cards with theme, volume, and sample tickets.
  2. Emerging Radar     — Live sandbox feed of novel/unclassified complaints.
  3. Ingest & Control   — Manual ingest + Re-Cluster Sandbox button.

Requires the FastAPI server to be running at API_BASE_URL.
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("CCE_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="CCE — Complaint Clustering Engine",
    page_icon="𖣘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Dark glassmorphism theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e2e8f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.85);
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 12px 16px;
        backdrop-filter: blur(10px);
    }

    /* Cluster cards */
    .cluster-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
        backdrop-filter: blur(12px);
        transition: border-color 0.2s ease;
    }
    .cluster-card:hover {
        border-color: rgba(129, 140, 248, 0.6);
    }
    .cluster-id-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        padding: 2px 10px;
        border-radius: 20px;
        border: 1px solid rgba(165,180,252,0.3);
        margin-bottom: 8px;
    }
    .cluster-theme {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 6px;
        letter-spacing: 0.03em;
    }
    .cluster-volume {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-bottom: 12px;
    }
    .volume-bar-bg {
        background: rgba(255,255,255,0.07);
        border-radius: 99px;
        height: 5px;
        margin-bottom: 14px;
    }
    .volume-bar-fill {
        background: linear-gradient(90deg, #6366f1, #a78bfa);
        border-radius: 99px;
        height: 5px;
    }
    .sample-pill {
        display: inline-block;
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.25);
        color: #c7d2fe;
        font-size: 0.72rem;
        padding: 4px 10px;
        border-radius: 8px;
        margin: 3px 3px 3px 0;
        line-height: 1.4;
    }

    /* Sandbox cards */
    .sandbox-item {
        background: rgba(239,68,68,0.06);
        border: 1px solid rgba(239,68,68,0.2);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.85rem;
        color: #fca5a5;
    }
    .sandbox-ts {
        font-size: 0.68rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-bottom: 20px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: opacity 0.2s ease, transform 0.1s ease;
    }
    .stButton > button:hover {
        opacity: 0.88;
        transform: translateY(-1px);
    }

    /* Toasts / alerts */
    .alert-success {
        background: rgba(16,185,129,0.12);
        border: 1px solid rgba(16,185,129,0.35);
        border-radius: 10px;
        padding: 12px 16px;
        color: #6ee7b7;
        font-size: 0.85rem;
    }
    .alert-error {
        background: rgba(239,68,68,0.1);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 10px;
        padding: 12px 16px;
        color: #fca5a5;
        font-size: 0.85rem;
    }

    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
    }

    textarea {
        background: rgba(255,255,255,0.05) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_get(path: str, params: dict | None = None) -> dict | None:
    try:
        with httpx.Client(base_url=API_BASE_URL, timeout=60.0) as client:
            r = client.get(path, params=params)
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        st.error("⚠️ Cannot reach the API server. Is `uvicorn api.main:app` running on port 8000?")
        return None
    except Exception as exc:
        st.error(f"API Error: {exc}")
        return None


def api_post(path: str, payload: dict) -> dict | None:
    try:
        with httpx.Client(base_url=API_BASE_URL, timeout=120.0) as client:
            r = client.post(path, json=payload)
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        st.error("⚠️ Cannot reach the API server. Is `uvicorn api.main:app` running on port 8000?")
        return None
    except Exception as exc:
        st.error(f"API Error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Sidebar — Navigation & Status
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 20px 0 10px;">
            <div style="font-size:2.4rem;">𖣘</div>
            <div style="font-size:1.1rem; font-weight:700; color:#f1f5f9; margin-top:6px;">
                CCE Dashboard
            </div>
            <div style="font-size:0.72rem; color:#475569; margin-top:2px;">
                Complaint Clustering Engine
            </div>
        </div>
        <hr style="border-color:rgba(255,255,255,0.07); margin: 10px 0 20px;">
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        ["📊 Cluster Explorer", "🚨 Emerging Radar", "⚡ Ingest & Control"],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        """
        <hr style="border-color:rgba(255,255,255,0.07); margin: 20px 0 10px;">
        <div style="font-size:0.68rem; color:#334155; text-align:center;">
            Powered by HDBSCAN · Groq LLM · FastAPI
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page: Cluster Explorer
# ---------------------------------------------------------------------------

if page == "📊 Cluster Explorer":
    st.markdown(
        '<div class="section-header">📊 Cluster Explorer</div>'
        '<div class="section-sub">Active semantic clusters identified from ingested complaints.</div>',
        unsafe_allow_html=True,
    )

    data = api_get("/clusters")

    if data:
        total = data.get("total_clusters", 0)
        clusters = data.get("clusters", [])

        # Top-line metrics
        col1, col2, col3 = st.columns(3)
        total_complaints = sum(c["sample_count"] for c in clusters)
        with col1:
            st.metric("Active Clusters", total)
        with col2:
            st.metric("Total Classified Complaints", total_complaints)
        with col3:
            sandbox_data = api_get("/sandbox", params={"page": 1, "page_size": 1})
            sandbox_total = sandbox_data.get("total_sandbox", "—") if sandbox_data else "—"
            st.metric("Sandbox (Unclassified)", sandbox_total)

        st.markdown("<br>", unsafe_allow_html=True)

        if not clusters:
            st.info("No clusters found. Use the **Ingest & Control** panel to add complaints.")
        else:
            for cluster in sorted(clusters, key=lambda c: -c["volume_pct"]):
                cid = cluster["cluster_id"]
                theme = cluster["theme_label"]
                vol = cluster["volume_pct"]
                count = cluster["sample_count"]
                samples = cluster.get("samples", [])

                samples_html = "".join(
                    f'<span class="sample-pill">{s[:80]}{"…" if len(s)>80 else ""}</span>'
                    for s in samples
                )

                st.markdown(
                    f"""
                    <div class="cluster-card">
                        <div class="cluster-id-badge">CLUSTER #{cid}</div>
                        <div class="cluster-theme">{theme}</div>
                        <div class="cluster-volume">{count:,} complaints &nbsp;·&nbsp; {vol:.1f}% of volume</div>
                        <div class="volume-bar-bg">
                            <div class="volume-bar-fill" style="width:{min(vol,100):.1f}%;"></div>
                        </div>
                        <div><strong style="font-size:0.75rem;color:#64748b;letter-spacing:.05em;">SAMPLE TICKETS</strong></div>
                        <div style="margin-top:6px;">{samples_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ---- Complaint drill-down: paginated load-more ----
                with st.expander(f"View complaints in this cluster ({count} total)"):
                    # Session state key tracks how many complaints to show for this cluster
                    sk = f"show_n_{cid}"
                    if sk not in st.session_state:
                        st.session_state[sk] = 10

                    n_show = st.session_state[sk]
                    drill = api_get(
                        f"/clusters/{cid}/complaints",
                        params={"page": 1, "page_size": n_show},
                    )

                    if drill and drill.get("complaints"):
                        complaints_list = drill["complaints"]
                        rows_html = "".join(
                            f"""
                            <tr>
                              <td style="padding:7px 10px;color:#64748b;font-size:0.7rem;
                                         white-space:nowrap;border-bottom:1px solid rgba(255,255,255,0.04);">
                                #{c['id']}
                              </td>
                              <td style="padding:7px 10px;color:#e2e8f0;font-size:0.81rem;
                                         line-height:1.5;border-bottom:1px solid rgba(255,255,255,0.04);">
                                {c['text']}
                              </td>
                            </tr>
                            """
                            for c in complaints_list
                        )
                        st.markdown(
                            f"""
                            <table style="width:100%;border-collapse:collapse;">
                              <thead>
                                <tr>
                                  <th style="padding:6px 10px;text-align:left;font-size:0.68rem;
                                             color:#475569;letter-spacing:.07em;
                                             border-bottom:1px solid rgba(255,255,255,0.1);">ID</th>
                                  <th style="padding:6px 10px;text-align:left;font-size:0.68rem;
                                             color:#475569;letter-spacing:.07em;
                                             border-bottom:1px solid rgba(255,255,255,0.1);">COMPLAINT TEXT</th>
                                </tr>
                              </thead>
                              <tbody>{rows_html}</tbody>
                            </table>
                            """,
                            unsafe_allow_html=True,
                        )

                        shown = len(complaints_list)
                        total_in_cluster = drill["total"]
                        remaining = total_in_cluster - shown

                        st.markdown(
                            f"<p style='font-size:0.72rem;color:#475569;margin:8px 0 4px;'>"
                            f"Showing {shown} of {total_in_cluster} complaints</p>",
                            unsafe_allow_html=True,
                        )

                        if remaining > 0:
                            load_more_label = f"Load {min(10, remaining)} more"
                            if st.button(load_more_label, key=f"more_{cid}"):
                                st.session_state[sk] += 10
                                st.rerun()
                        else:
                            st.markdown(
                                "<p style='font-size:0.72rem;color:#334155;'>All complaints loaded.</p>",
                                unsafe_allow_html=True,
                            )
                    elif drill:
                        st.info("No complaint records found for this cluster yet.")

    else:
        st.warning("Could not load cluster data. Make sure the API server is running.")




# ---------------------------------------------------------------------------
# Page: Emerging Radar (Sandbox)
# ---------------------------------------------------------------------------

elif page == "🚨 Emerging Radar":
    st.markdown(
        '<div class="section-header">🚨 Emerging Radar — Sandbox</div>'
        '<div class="section-sub">Novel complaints that did not fit any known cluster. Potential emerging issues.</div>',
        unsafe_allow_html=True,
    )

    page_num = st.number_input("Page", min_value=1, value=1, step=1, key="sandbox_page")
    page_size = st.select_slider("Items per page", options=[10, 20, 50], value=20, key="sandbox_size")

    data = api_get("/sandbox", params={"page": page_num, "page_size": page_size})

    if data:
        total_sandbox = data.get("total_sandbox", 0)
        items = data.get("items", [])

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Total in Sandbox", f"{total_sandbox:,}")
        with col2:
            if total_sandbox > 0:
                st.markdown(
                    f"""
                    <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);
                         border-radius:10px;padding:10px 16px;font-size:0.8rem;color:#fca5a5;">
                        ⚠️ <strong>{total_sandbox}</strong> unclassified complaints detected.
                        Use <em>Ingest &amp; Control → Re-Cluster Sandbox</em> to attempt automatic categorisation.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        if not items:
            st.success("✅ Sandbox is empty — all complaints are classified.")
        else:
            for item in items:
                ts_raw = item.get("created_at", "")
                try:
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).strftime("%d %b %Y, %H:%M UTC")
                except Exception:
                    ts = ts_raw

                st.markdown(
                    f"""
                    <div class="sandbox-item">
                        <div>#{item['id']} — {item['text']}</div>
                        <div class="sandbox-ts">Captured: {ts}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.warning("Could not load sandbox data.")


# ---------------------------------------------------------------------------
# Page: Ingest & Control
# ---------------------------------------------------------------------------

elif page == "⚡ Ingest & Control":
    st.markdown(
        '<div class="section-header">⚡ Ingest &amp; Control Panel</div>'
        '<div class="section-sub">Manually ingest complaint batches or trigger sandbox re-clustering.</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["📥 Ingest Complaints", "🔁 Re-Cluster Sandbox", "⚙️ Model Admin"])

    # -- Tab 1: Ingest --
    with tab1:
        st.markdown("**Paste complaints below — one per line.**")
        complaint_input = st.text_area(
            "Complaint Texts",
            height=220,
            placeholder="App keeps crashing on login\nUnauthorized charge on my account\nCannot transfer funds internationally\n...",
            label_visibility="collapsed",
        )

        if st.button("🚀 Run Pipeline", key="run_ingest_btn", use_container_width=True):
            lines = [ln.strip() for ln in complaint_input.strip().splitlines() if ln.strip()]
            if not lines:
                st.warning("Please enter at least one complaint.")
            else:
                with st.spinner(f"Running pipeline on {len(lines)} complaint(s)…"):
                    result = api_post("/complaints/ingest", {"texts": lines})

                if result:
                    st.markdown(
                        f"""
                        <div class="alert-success">
                            ✅ Pipeline complete — <strong>{result['total_ingested']}</strong> ingested,
                            <strong>{result['classified_count']}</strong> classified into
                            <strong>{len(result['cluster_summaries'])}</strong> cluster(s),
                            <strong>{result['sandbox_count']}</strong> sent to sandbox.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("<br>", unsafe_allow_html=True)

                    for summary in result.get("cluster_summaries", []):
                        with st.expander(f"Cluster #{summary['cluster_id']} — {summary['theme_label']}"):
                            st.write(f"**Count:** {summary['sample_count']} complaints")
                            st.write(f"**Volume:** {summary['volume_pct']}%")
                            st.write("**Samples:**")
                            for s in summary.get("samples", []):
                                st.markdown(f"- {s}")

    # -- Tab 2: Re-Cluster --
    with tab2:
        st.markdown(
            "Trigger a sub-clustering pass over all complaints currently in the **Sandbox**. "
            "Records that form new density clusters will be promoted to production."
        )
        max_items = st.number_input(
            "Max sandbox items to process (0 = all)", min_value=0, value=0, step=10
        )

        if st.button("🔁 Re-Cluster Sandbox", key="recluster_btn", use_container_width=True):
            payload: dict = {}
            if max_items > 0:
                payload["max_sandbox_items"] = int(max_items)

            with st.spinner("Running sandbox sub-clustering…"):
                result = api_post("/sandbox/recluster", payload)

            if result:
                st.markdown(
                    f"""
                    <div class="alert-success">
                        ✅ Re-cluster complete — <strong>{result['new_clusters_formed']}</strong> new cluster(s) formed,
                        <strong>{result['sandbox_remaining']}</strong> complaint(s) remain in sandbox.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if result.get("cluster_summaries"):
                    st.markdown("**Newly formed clusters:**")
                    for s in result["cluster_summaries"]:
                        st.markdown(
                            f"- **Cluster #{s['cluster_id']}** — {s['theme_label']} "
                            f"({s['sample_count']} complaints)"
                        )

    # -- Tab 3: Model Admin --
    with tab3:
        st.markdown(
            '<p style="color:#94a3b8;font-size:0.85rem;">'
            "Tune HDBSCAN parameters and re-cluster <strong>all existing DB records</strong> "
            "without re-ingesting data. Changes take effect immediately after clicking Re-Cluster."
            "</p>",
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            min_cluster_size = st.slider(
                "min_cluster_size",
                min_value=2,
                max_value=30,
                value=5,
                step=1,
                help="Minimum complaints to form a cluster. Lower = more (smaller) clusters.",
            )
            min_samples = st.slider(
                "min_samples",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                help="Controls HDBSCAN conservatism. 1 = most permissive, gives most clusters.",
            )
        with col_b:
            method = st.radio(
                "cluster_selection_method",
                options=["leaf", "eom"],
                index=0,
                help="'leaf' = fine-grained clusters. 'eom' = fewer merged clusters.",
            )
            st.markdown(
                """
                <div style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.25);
                     border-radius:10px;padding:12px;font-size:0.78rem;color:#a5b4fc;margin-top:8px;">
                    <strong>Tuning guide for 200 complaints:</strong><br>
                    min_cluster_size: <strong>3–6</strong><br>
                    min_samples: <strong>1</strong><br>
                    method: <strong>leaf</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Re-Cluster All Data", key="retrain_btn", use_container_width=True):
            payload = {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_method": method,
            }
            with st.spinner(
                f"Re-embedding and re-clustering all DB complaints "
                f"(min_cluster_size={min_cluster_size}, min_samples={min_samples}, method={method})..."
            ):
                result = api_post("/model/retrain", payload)

            if result:
                st.markdown(
                    f"""
                    <div class="alert-success">
                        Re-cluster complete — <strong>{result['clusters_formed']}</strong> cluster(s) formed
                        from <strong>{result['total_retrained']}</strong> complaints.
                        <strong>{result['sandbox_count']}</strong> remain in sandbox.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)
                for s in result.get("cluster_summaries", []):
                    st.markdown(
                        f"- **Cluster #{s['cluster_id']}** — {s['theme_label']} "
                        f"({s['sample_count']} complaints, {s['volume_pct']}%)"
                    )

        st.markdown(
            "<hr style='border-color:rgba(255,255,255,0.07);margin:24px 0 16px;'>",
            unsafe_allow_html=True,
        )
        st.markdown("**Reset In-Memory Model Only** — model cleared, next ingest will retrain.")
        if st.button("Reset Model State", key="reset_model_btn"):
            result = api_post("/model/reset", {})
            if result:
                st.success(result.get("message", "Model reset successfully."))
