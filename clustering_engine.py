import hdbscan
import numpy as np
import pandas as pd

class ClusteringEngine:
    def __init__(self, min_cluster_size=15, random_state=42):
        # random_state ensures reproducibility across stochastic processes
        self.min_cluster_size = min_cluster_size
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True # Crucial for real-time scoring of new data
        )
        self.random_state = random_state

    def fit_predict(self, embeddings):
        """Fits the model and returns cluster labels."""
        labels = self.clusterer.fit_predict(embeddings)
        return labels

    def predict_new_complaint(self, new_embeddings):
        """Approximates the cluster for a new incoming complaint without refitting."""
        labels, strengths = hdbscan.approximate_predict(self.clusterer, new_embeddings)
        return labels

    def isolate_sandbox(self, df, labels):
        """Separates known clusters from the -1 outliers (Sandbox)."""
        df['Cluster_ID'] = labels
        sandbox_df = df[df['Cluster_ID'] == -1]
        production_df = df[df['Cluster_ID'] != -1]
        return production_df, sandbox_df