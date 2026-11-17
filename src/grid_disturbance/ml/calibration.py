"""Model probability calibration and reliability curve analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationResult:
    """Expected Calibration Error (ECE) and binning statistics."""

    expected_calibration_error: float
    max_calibration_error: float
    bin_confidences: list[float]
    bin_accuracies: list[float]
    bin_counts: list[int]


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> CalibrationResult:
    """Calculate Expected Calibration Error (ECE) across multi-class predictions."""
    y_t = np.asarray(y_true)
    probs = np.asarray(y_proba)

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_t).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs = []
    bin_accs = []
    bin_counts = []
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (
            (confidences > bin_lower) & (confidences <= bin_upper)
            if i > 0
            else (confidences >= bin_lower) & (confidences <= bin_upper)
        )
        prop_in_bin = float(np.mean(in_bin))

        if np.sum(in_bin) > 0:
            avg_acc = float(np.mean(accuracies[in_bin]))
            avg_conf = float(np.mean(confidences[in_bin]))
            gap = abs(avg_acc - avg_conf)
            ece += prop_in_bin * gap
            mce = max(mce, gap)
            bin_confs.append(avg_conf)
            bin_accs.append(avg_acc)
            bin_counts.append(int(np.sum(in_bin)))
        else:
            bin_confs.append(float((bin_lower + bin_upper) / 2.0))
            bin_accs.append(0.0)
            bin_counts.append(0)

    return CalibrationResult(
        expected_calibration_error=float(ece),
        max_calibration_error=float(mce),
        bin_confidences=bin_confs,
        bin_accuracies=bin_accs,
        bin_counts=bin_counts,
    )
