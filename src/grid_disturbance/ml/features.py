"""Feature vectorizer and matrix builder with zero-leakage preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from grid_disturbance.signals.features import extract_signal_features, get_feature_names


class SignalFeatureVectorizer:
    """Extracts and scales tabular features across signal collections."""

    def __init__(self, sampling_rate: float = 30.0):
        self.sampling_rate = sampling_rate
        self.feature_names = get_feature_names()
        self.scaler = StandardScaler()
        self.is_fitted = False

    def extract_features_df(self, signals: list[np.ndarray]) -> pd.DataFrame:
        """Extract unscaled feature DataFrame from a list of signals."""
        rows = []
        for sig in signals:
            f_dict = extract_signal_features(sig, self.sampling_rate)
            rows.append({k: f_dict.get(k, 0.0) for k in self.feature_names})
        return pd.DataFrame(rows, columns=self.feature_names)

    def fit(self, signals: list[np.ndarray]) -> SignalFeatureVectorizer:
        """Fit feature scaler strictly on the training set."""
        df = self.extract_features_df(signals)
        x_mat = df.to_numpy(dtype=np.float64)
        # Impute any NaNs in extracted features with 0
        x_mat = np.nan_to_num(x_mat, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler.fit(x_mat)
        self.is_fitted = True
        return self

    def transform(self, signals: list[np.ndarray]) -> np.ndarray:
        """Transform signals into scaled feature matrix using fitted scaler."""
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted before calling transform.")
        df = self.extract_features_df(signals)
        x_mat = df.to_numpy(dtype=np.float64)
        x_mat = np.nan_to_num(x_mat, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(x_mat)

    def fit_transform(self, signals: list[np.ndarray]) -> np.ndarray:
        """Fit on signals and transform in one call."""
        self.fit(signals)
        return self.transform(signals)
