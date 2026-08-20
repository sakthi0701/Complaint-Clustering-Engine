import hdbscan
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize as l2_normalize


class ClusteringEngine:
    """
    HDBSCAN-based clustering engine with L2-normalization and tunable parameters.

    Key parameters for controlling cluster granularity:
      min_cluster_size      - Minimum points to form a cluster. Lower = more clusters.
      min_samples           - Controls conservatism. 1 = most permissive (recommended for text).
      cluster_selection_method
                            - 'leaf' gives more fine-grained clusters.
                            - 'eom' (Excess of Mass) merges aggressively → fewer clusters.
      normalize_embeddings  - L2-normalize vectors before clustering. CRITICAL for text
                              embeddings: converts euclidean distance to cosine-equivalent.
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int = 1,
        cluster_selection_method: str = "leaf",
        normalize_embeddings: bool = True,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.normalize_embeddings = normalize_embeddings
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,          # CRITICAL: set low (1) for more clusters
            metric="euclidean",
            cluster_selection_method=cluster_selection_method,  # 'leaf' > 'eom' for granularity
            prediction_data=True,
        )
        self.is_fitted = False

    def _prepare(self, embeddings) -> np.ndarray:
        """L2-normalizes embeddings so euclidean distance ≈ cosine distance."""
        X = np.array(embeddings, dtype=np.float32)
        if self.normalize_embeddings:
            X = l2_normalize(X, norm="l2")
        return X

    def fit_predict(self, embeddings: list | np.ndarray) -> np.ndarray:
        """Fits the HDBSCAN model on the full embedding matrix and returns labels."""
        X = self._prepare(embeddings)
        labels = self.clusterer.fit_predict(X)
        self.is_fitted = True
        return labels

    def predict_new(self, new_embeddings: list | np.ndarray) -> np.ndarray:
        """
        Approximates cluster membership for incoming complaints without refitting.
        Requires the model to have been fitted first.
        """
        if not self.is_fitted:
            raise RuntimeError("ClusteringEngine must be fitted before calling predict_new().")
        X = self._prepare(new_embeddings)
        labels, _ = hdbscan.approximate_predict(self.clusterer, X)
        return labels

    def isolate_sandbox(self, df: pd.DataFrame, labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Splits into production (label != -1) and sandbox (label == -1) DataFrames."""
        df = df.copy()
        df["Cluster_ID"] = labels
        sandbox_df = df[df["Cluster_ID"] == -1].copy()
        production_df = df[df["Cluster_ID"] != -1].copy()
        return production_df, sandbox_df
