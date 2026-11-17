"""Tests for ML feature vectorizer, classical classifiers, and evaluation."""

import numpy as np

from grid_disturbance.ml.classical import ClassicalDisturbanceClassifier
from grid_disturbance.ml.evaluation import evaluate_classifier_predictions
from grid_disturbance.ml.features import SignalFeatureVectorizer
from grid_disturbance.signals.features import extract_signal_features, get_feature_names
from grid_disturbance.synthetic.presets import generate_benchmark_dataset


def test_feature_extraction_completeness():
    sig = np.sin(np.linspace(0, 20, 600))
    feats = extract_signal_features(sig, sampling_rate=30.0)
    names = get_feature_names()

    assert len(feats) == len(names)
    assert all(k in feats for k in names)
    assert np.all(np.isfinite(list(feats.values())))


def test_classical_model_training_and_evaluation():
    # Generate small benchmark
    signals, df = generate_benchmark_dataset(samples_per_class=8, random_seed=42)
    classes = sorted(df["disturbance_type"].unique().tolist())
    lbl_map = {c: i for i, c in enumerate(classes)}

    vec = SignalFeatureVectorizer(sampling_rate=30.0)
    x_mat = vec.fit_transform([s.data for s in signals])
    y_vec = np.array([lbl_map[t] for t in df["disturbance_type"]])

    clf = ClassicalDisturbanceClassifier(model_type="random_forest", random_state=42)
    clf.fit(x_mat, y_vec)

    preds = clf.predict(x_mat)
    probs = clf.predict_proba(x_mat)

    report = evaluate_classifier_predictions(y_vec, preds, probs, class_names=classes)
    assert report.accuracy > 0.70
    assert report.macro_f1 > 0.60
    assert len(report.per_class_metrics) == len(classes)
