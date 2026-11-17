"""Model explainability, feature attribution, and diagnostic explanations."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance


class ModelExplainer:
    """Provides feature attributions and diagnostic explanations for predictions."""

    def __init__(self, model: Any, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names

    def get_global_importance(
        self,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        top_k: int = 10,
    ) -> dict[str, float]:
        """Compute global feature importance via tree importances or permutation importance."""
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        elif x_val is not None and y_val is not None:
            res = permutation_importance(self.model, x_val, y_val, n_repeats=5, random_state=42)
            imp = res.importances_mean
        else:
            return {}

        imp_dict = {name: float(val) for name, val in zip(self.feature_names, imp, strict=False)}
        sorted_imp = dict(sorted(imp_dict.items(), key=lambda item: item[1], reverse=True)[:top_k])
        return sorted_imp

    def explain_instance(
        self,
        instance_vector: np.ndarray,
        baseline_vector: np.ndarray | None = None,
        top_k: int = 5,
    ) -> list[tuple[str, float, float]]:
        """Explain an individual prediction by computing feature contribution relative to baseline.

        Returns list of (feature_name, feature_value, contribution_score).
        """
        vec = np.asarray(instance_vector).flatten()
        if hasattr(self.model, "feature_importances_"):
            weights = self.model.feature_importances_
        else:
            weights = np.ones(len(vec)) / len(vec)

        # Deviation from baseline or zero
        ref = (
            np.asarray(baseline_vector).flatten()
            if baseline_vector is not None
            else np.zeros_like(vec)
        )
        deviations = np.abs(vec - ref) * weights

        indices = np.argsort(deviations)[::-1][:top_k]
        explanations = []
        for idx in indices:
            name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            explanations.append((name, float(vec[idx]), float(deviations[idx])))

        return explanations
