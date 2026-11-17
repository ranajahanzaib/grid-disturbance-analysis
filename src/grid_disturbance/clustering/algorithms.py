"""Unsupervised event clustering and anomaly detection for disturbance signatures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import davies_bouldin_score, silhouette_score


@dataclass
class ClusteringResult:
    """Clustering outcome containing labels, evaluation metrics, and 2D projections."""

    algorithm: str
    n_clusters: int
    cluster_labels: list[int]
    silhouette: float
    davies_bouldin: float
    pca_2d_coordinates: list[list[float]] = field(default_factory=list)
    cluster_sizes: dict[int, int] = field(default_factory=dict)
    representative_indices: dict[int, int] = field(default_factory=dict)
    anomaly_indices: list[int] = field(default_factory=list)


def cluster_disturbance_features(
    feature_matrix: np.ndarray,
    algorithm: Literal["kmeans", "dbscan"] = "kmeans",
    n_clusters: int = 5,
    eps: float = 0.5,
    min_samples: int = 3,
    random_state: int = 42,
) -> ClusteringResult:
    """Cluster signal feature vectors to discover emergent disturbance signatures."""
    x = np.asarray(feature_matrix, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0)

    # 2D PCA projection for visualization
    pca = PCA(n_components=2, random_state=random_state)
    coords_2d = pca.fit_transform(x).tolist()

    if algorithm == "kmeans":
        model = KMeans(n_clusters=min(n_clusters, len(x)), random_state=random_state, n_init="auto")
        labels = model.fit_predict(x)
        k = len(set(labels))
    elif algorithm == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(x)
        k = len(set(labels) - {-1})
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")

    # Compute cluster quality metrics
    sil = 0.0
    db = 0.0
    if len(set(labels)) > 1 and len(set(labels)) < len(x):
        try:
            sil = float(silhouette_score(x, labels))
            db = float(davies_bouldin_score(x, labels))
        except Exception:
            pass

    # Sizes and representative center-closest samples
    sizes: dict[int, int] = {}
    representatives: dict[int, int] = {}
    for c in set(labels):
        indices = np.where(labels == c)[0]
        sizes[int(c)] = len(indices)
        if c != -1 and len(indices) > 0:
            center = np.mean(x[indices], axis=0)
            dists = np.linalg.norm(x[indices] - center, axis=1)
            representatives[int(c)] = int(indices[np.argmin(dists)])

    # Separate Anomaly detection via Isolation Forest
    iso = IsolationForest(contamination=0.08, random_state=random_state)
    anomaly_mask = iso.fit_predict(x) == -1
    anomaly_indices = [int(idx) for idx in np.where(anomaly_mask)[0]]

    return ClusteringResult(
        algorithm=algorithm,
        n_clusters=k,
        cluster_labels=labels.tolist(),
        silhouette=sil,
        davies_bouldin=db,
        pca_2d_coordinates=coords_2d,
        cluster_sizes=sizes,
        representative_indices=representatives,
        anomaly_indices=anomaly_indices,
    )
