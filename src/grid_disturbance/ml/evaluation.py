"""Comprehensive evaluation metrics and reporting for disturbance models."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ClassPerformanceMetrics(BaseModel):
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    support: int = 0


class ModelEvaluationReport(BaseModel):
    """Evaluation report containing comprehensive multi-class metrics."""

    model_name: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    roc_auc_ovr: float | None = None
    confusion_matrix: list[list[int]] = Field(default_factory=list)
    class_names: list[str] = Field(default_factory=list)
    per_class_metrics: dict[str, ClassPerformanceMetrics] = Field(default_factory=dict)
    total_test_samples: int = 0


def evaluate_classifier_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    class_names: list[str] | None = None,
    model_name: str = "classifier",
) -> ModelEvaluationReport:
    """Compute full suite of classification metrics."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    unique_labels = sorted(list(set(y_t) | set(y_p)))
    if class_names is None:
        c_names = [str(lbl) for lbl in unique_labels]
    else:
        c_names = class_names

    acc = float(accuracy_score(y_t, y_p))
    m_prec = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    m_rec = float(recall_score(y_t, y_p, average="macro", zero_division=0))
    m_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    w_f1 = float(f1_score(y_t, y_p, average="weighted", zero_division=0))

    cm = confusion_matrix(y_t, y_p, labels=unique_labels).tolist()

    # Per-class metrics
    p_class = precision_score(y_t, y_p, average=None, labels=unique_labels, zero_division=0)
    r_class = recall_score(y_t, y_p, average=None, labels=unique_labels, zero_division=0)
    f1_class = f1_score(y_t, y_p, average=None, labels=unique_labels, zero_division=0)

    per_class: dict[str, ClassPerformanceMetrics] = {}
    for idx, lbl in enumerate(unique_labels):
        name = c_names[idx] if idx < len(c_names) else str(lbl)
        support = int(np.sum(y_t == lbl))
        per_class[name] = ClassPerformanceMetrics(
            precision=float(p_class[idx]),
            recall=float(r_class[idx]),
            f1_score=float(f1_class[idx]),
            support=support,
        )

    # ROC-AUC if probabilities available
    roc_auc = None
    if y_proba is not None and len(unique_labels) > 1:
        try:
            # Check if all classes present in y_t
            if len(set(y_t)) == y_proba.shape[1]:
                roc_auc = float(roc_auc_score(y_t, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            roc_auc = None

    return ModelEvaluationReport(
        model_name=model_name,
        accuracy=acc,
        macro_precision=m_prec,
        macro_recall=m_rec,
        macro_f1=m_f1,
        weighted_f1=w_f1,
        roc_auc_ovr=roc_auc,
        confusion_matrix=cm,
        class_names=c_names,
        per_class_metrics=per_class,
        total_test_samples=len(y_t),
    )
