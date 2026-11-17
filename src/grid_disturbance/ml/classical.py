"""Classical machine learning baselines for disturbance classification."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class ClassicalDisturbanceClassifier:
    """Unified wrapper around classical scikit-learn models for grid disturbances."""

    SUPPORTED_MODELS = [
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "logistic_regression",
        "svm",
    ]

    def __init__(
        self,
        model_type: Literal[
            "random_forest", "extra_trees", "gradient_boosting", "logistic_regression", "svm"
        ] = "random_forest",
        random_state: int = 42,
        class_weight: str | None = "balanced",
        **kwargs: Any,
    ):
        self.model_type = model_type
        self.random_state = random_state
        self.class_weight = class_weight
        self.kwargs = kwargs
        self.model = self._build_model()
        self.classes_: np.ndarray | None = None

    def _build_model(self) -> Any:
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.kwargs.get("n_estimators", 150),
                max_depth=self.kwargs.get("max_depth", 12),
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=-1,
            )
        if self.model_type == "extra_trees":
            return ExtraTreesClassifier(
                n_estimators=self.kwargs.get("n_estimators", 150),
                max_depth=self.kwargs.get("max_depth", 12),
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=-1,
            )
        if self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self.kwargs.get("n_estimators", 100),
                learning_rate=self.kwargs.get("learning_rate", 0.1),
                max_depth=self.kwargs.get("max_depth", 5),
                random_state=self.random_state,
            )
        if self.model_type == "logistic_regression":
            return LogisticRegression(
                max_iter=self.kwargs.get("max_iter", 1000),
                class_weight=self.class_weight,
                random_state=self.random_state,
                C=self.kwargs.get("C", 1.0),
            )
        if self.model_type == "svm":
            return SVC(
                probability=True,
                class_weight=self.class_weight,
                random_state=self.random_state,
                C=self.kwargs.get("C", 1.0),
                kernel=self.kwargs.get("kernel", "rbf"),
            )
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, x: np.ndarray, y: np.ndarray) -> ClassicalDisturbanceClassifier:
        """Fit model to training feature matrix X and labels y."""
        self.model.fit(x, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict disturbance class indices or labels."""
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probability distribution."""
        return self.model.predict_proba(x)

    def get_feature_importances(self, feature_names: list[str]) -> dict[str, float]:
        """Return feature importance dictionary sorted descending."""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            return {}

        feat_imp = {name: float(val) for name, val in zip(feature_names, importances, strict=False)}
        return dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))
