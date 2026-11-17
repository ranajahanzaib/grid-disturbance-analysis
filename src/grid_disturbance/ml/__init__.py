"""Machine learning architecture for power grid disturbance classification."""

from grid_disturbance.ml.calibration import CalibrationResult, compute_calibration_metrics
from grid_disturbance.ml.classical import ClassicalDisturbanceClassifier
from grid_disturbance.ml.datasets import Signal1DDataset, SpectrogramDataset
from grid_disturbance.ml.evaluation import (
    ClassPerformanceMetrics,
    ModelEvaluationReport,
    evaluate_classifier_predictions,
)
from grid_disturbance.ml.explainability import ModelExplainer
from grid_disturbance.ml.features import SignalFeatureVectorizer
from grid_disturbance.ml.spectrogram_cnn import SpectrogramCNN
from grid_disturbance.ml.temporal_cnn import TemporalCNN
from grid_disturbance.ml.training import train_classical_model, train_deep_learning_model

__all__ = [
    "CalibrationResult",
    "ClassPerformanceMetrics",
    "ClassicalDisturbanceClassifier",
    "ModelEvaluationReport",
    "ModelExplainer",
    "Signal1DDataset",
    "SignalFeatureVectorizer",
    "SpectrogramCNN",
    "SpectrogramDataset",
    "TemporalCNN",
    "compute_calibration_metrics",
    "evaluate_classifier_predictions",
    "train_classical_model",
    "train_deep_learning_model",
]
