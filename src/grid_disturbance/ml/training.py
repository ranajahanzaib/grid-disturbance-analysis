"""Training pipelines for classical and deep learning disturbance classifiers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from grid_disturbance.ml.classical import ClassicalDisturbanceClassifier
from grid_disturbance.ml.datasets import Signal1DDataset, SpectrogramDataset
from grid_disturbance.ml.evaluation import ModelEvaluationReport, evaluate_classifier_predictions
from grid_disturbance.ml.features import SignalFeatureVectorizer
from grid_disturbance.ml.spectrogram_cnn import SpectrogramCNN
from grid_disturbance.ml.temporal_cnn import TemporalCNN

logger = logging.getLogger(__name__)


def train_classical_model(
    train_signals: list[np.ndarray],
    train_labels: list[int],
    test_signals: list[np.ndarray],
    test_labels: list[int],
    model_type: str = "random_forest",
    class_names: list[str] | None = None,
    sampling_rate: float = 30.0,
    random_state: int = 42,
) -> tuple[ClassicalDisturbanceClassifier, SignalFeatureVectorizer, ModelEvaluationReport]:
    """Train and evaluate a classical feature-based classifier."""
    vectorizer = SignalFeatureVectorizer(sampling_rate=sampling_rate)
    x_train = vectorizer.fit_transform(train_signals)
    x_test = vectorizer.transform(test_signals)

    y_train = np.asarray(train_labels)
    y_test = np.asarray(test_labels)

    clf = ClassicalDisturbanceClassifier(model_type=model_type, random_state=random_state)  # type: ignore
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test) if hasattr(clf.model, "predict_proba") else None

    report = evaluate_classifier_predictions(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_prob,
        class_names=class_names,
        model_name=f"classical_{model_type}",
    )

    logger.info(
        f"Trained {model_type}: Test Accuracy = {report.accuracy:.3f}, Macro F1 = {report.macro_f1:.3f}"
    )
    return clf, vectorizer, report


def train_deep_learning_model(
    train_signals: list[np.ndarray],
    train_labels: list[int],
    val_signals: list[np.ndarray],
    val_labels: list[int],
    test_signals: list[np.ndarray],
    test_labels: list[int],
    model_type: str = "temporal_cnn",  # "temporal_cnn" or "spectrogram_cnn"
    num_classes: int = 6,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 0.001,
    class_names: list[str] | None = None,
    sampling_rate: float = 30.0,
    device: str = "auto",
) -> tuple[nn.Module, ModelEvaluationReport, dict[str, list[float]]]:
    """Train and evaluate a 1D Temporal CNN or 2D Spectrogram CNN."""
    if device == "auto":
        dev = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    else:
        dev = torch.device(device)

    # Prepare datasets and loaders
    train_ds: Dataset[Any]
    val_ds: Dataset[Any]
    test_ds: Dataset[Any]
    if model_type == "temporal_cnn":
        train_ds = Signal1DDataset(
            train_signals, train_labels, target_length=int(20 * sampling_rate)
        )
        val_ds = Signal1DDataset(val_signals, val_labels, target_length=int(20 * sampling_rate))
        test_ds = Signal1DDataset(test_signals, test_labels, target_length=int(20 * sampling_rate))
        model: nn.Module = TemporalCNN(num_classes=num_classes)
    elif model_type == "spectrogram_cnn":
        train_ds = SpectrogramDataset(train_signals, train_labels, sampling_rate=sampling_rate)
        val_ds = SpectrogramDataset(val_signals, val_labels, sampling_rate=sampling_rate)
        test_ds = SpectrogramDataset(test_signals, test_labels, sampling_rate=sampling_rate)
        model = SpectrogramCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown deep learning model_type: {model_type}")

    model.to(dev)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(batch_y)

        t_loss = t_loss / len(train_ds)

        # Validation
        model.eval()
        v_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
                out = model(batch_x)
                loss = criterion(out, batch_y)
                v_loss += loss.item() * len(batch_y)
                preds = torch.argmax(out, dim=1)
                correct += int(torch.sum(preds == batch_y))

        v_loss = v_loss / len(val_ds) if len(val_ds) > 0 else 0.0
        v_acc = (correct / len(val_ds)) if len(val_ds) > 0 else 0.0

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights:
        model.load_state_dict({k: v.to(dev) for k, v in best_weights.items()})

    # Evaluate on test set
    model.eval()
    test_preds = []
    test_probs = []
    test_trues = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(dev)
            out = model(batch_x)
            prob = torch.softmax(out, dim=1).cpu().numpy()
            pred = np.argmax(prob, axis=1)

            test_preds.extend(pred.tolist())
            test_probs.append(prob)
            test_trues.extend(batch_y.numpy().tolist())

    y_test_pred = np.array(test_preds)
    y_test_true = np.array(test_trues)
    y_test_prob = np.concatenate(test_probs, axis=0) if test_probs else None

    report = evaluate_classifier_predictions(
        y_true=y_test_true,
        y_pred=y_test_pred,
        y_proba=y_test_prob,
        class_names=class_names,
        model_name=model_type,
    )

    logger.info(
        f"Trained {model_type}: Test Accuracy = {report.accuracy:.3f}, Macro F1 = {report.macro_f1:.3f}"
    )
    return model, report, history
