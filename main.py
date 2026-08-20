"""
main.py — Dev CLI runner (Phase 1 demonstration).

For production use, start the FastAPI server instead:
  uvicorn api.main:app --reload

This script runs the full pipeline on a small in-memory dataset
and prints the cluster report to stdout — useful for smoke-testing
the core ML components without spinning up the server.
"""
import sys
import os

# Ensure the project root is on sys.path so `core.*` imports resolve.
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from core.vectorizer import LMStudioVectorizer
from core.clustering_engine import ClusteringEngine
from core.labeler import ThemeLabeler

# --- INTERACTIVE CONTROLS ---
MIN_CLUSTER_SIZE = 5
TEXT_COLUMN = "complaint_text"
# ----------------------------


def generate_production_report(df, sandbox_df, labeler: ThemeLabeler):
    """Prints a formatted cluster report to stdout."""
    total_volume = len(df) + len(sandbox_df)

    print("\n" + "=" * 80)
    print("COMPLAINT CLUSTERING ENGINE SYSTEM SUMMARY")
    print("=" * 80 + "\n")

    for cluster_id, group in df.groupby("Cluster_ID"):
        weight = (len(group) / total_volume) * 100
        samples = labeler.get_cluster_samples(group, TEXT_COLUMN, top_n=2)
        theme = labeler.generate_label(int(cluster_id), samples)

        print(f"[CLUSTER ID: {cluster_id}]")
        print(f"Theme Label:  {theme}")
        print(f"Data Weight:  {weight:.1f}% of total volume")
        print("Sample Complaints:")
        for s in samples:
            print(f'  - "{s}"')
        print()

    print("=" * 80)
    print("EARLY WARNING RADAR: EMERGING TRENDS SANDBOX")
    print("=" * 80)
    print(f"[CLUSTER ID: -1] ({labeler.generate_label(-1, [])})")
    print(f"Total Trapped Volume: {len(sandbox_df)} complaints")
    sandbox_samples = labeler.get_cluster_samples(sandbox_df, TEXT_COLUMN, top_n=2)
    print("Sample Anomalies Detected:")
    for s in sandbox_samples:
        print(f'  - "{s}"')
    print()


def main():
    data = {
        TEXT_COLUMN: [
            "The mobile app keeps crashing every time I try to open the transfer tab.",
            "Mobile app is hanging.",
            "Your iOS application is completely broken, it freezes on the login screen.",
            "App freezes when I type my password.",
            "My credit card was charged twice for the same transaction on Tuesday.",
            "An unauthorized fee of $35 appeared on my monthly bank statement.",
            "I got double billed for my subscription this month.",
            "The new crypto wallet feature button is completely missing from my menu.",
            "I am trying to use the crypto feature but it says region not supported.",
        ]
    }
    raw_df = pd.DataFrame(data)

    vectorizer = LMStudioVectorizer()
    valid_df = vectorizer.clean_and_filter(raw_df, TEXT_COLUMN)
    embeddings = vectorizer.vectorize(valid_df[TEXT_COLUMN])

    engine = ClusteringEngine(min_cluster_size=MIN_CLUSTER_SIZE)
    labels = engine.fit_predict(embeddings)
    production_df, sandbox_df = engine.isolate_sandbox(valid_df, labels)

    labeler = ThemeLabeler()
    generate_production_report(production_df, sandbox_df, labeler)


if __name__ == "__main__":
    main()