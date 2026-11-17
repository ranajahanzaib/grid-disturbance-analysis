"""Signal preprocessing, filtering, detrending, and outlier handling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.signal as signal

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingRecord:
    """Audit log of preprocessing steps applied to a signal."""

    steps_applied: list[str] = field(default_factory=list)
    parameters: dict[str, dict] = field(default_factory=dict)
    original_length: int = 0
    final_length: int = 0
    sampling_rate: float = 30.0
    nan_count: int = 0
    outliers_clipped: int = 0


def detect_and_handle_missing(
    x: np.ndarray,
    method: Literal["linear", "nearest", "zero", "drop"] = "linear",
    max_consecutive_nans: int | None = None,
) -> tuple[np.ndarray, int]:
    """Detect and impute missing values (NaN or Inf) in a 1D signal.

    Args:
        x: Input 1D NumPy array.
        method: Interpolation strategy ('linear', 'nearest', 'zero', 'drop').
        max_consecutive_nans: If specified, raise ValueError if gap exceeds this.

    Returns:
        Cleaned array and count of replaced invalid values.
    """
    arr = np.asarray(x, dtype=np.float64).copy()
    invalid_mask = ~np.isfinite(arr)
    invalid_count = int(np.sum(invalid_mask))

    if invalid_count == 0:
        return arr, 0

    if max_consecutive_nans is not None and invalid_count > 0:
        # Check consecutive runs
        runs = np.diff(
            np.where(
                np.concatenate(([invalid_mask[0]], invalid_mask[:-1] != invalid_mask[1:], [True]))
            )[0]
        )[::2]
        if len(runs) > 0 and np.max(runs) > max_consecutive_nans:
            raise ValueError(
                f"Missing data gap of {np.max(runs)} samples exceeds limit of {max_consecutive_nans}."
            )

    if method == "drop":
        return arr[~invalid_mask], invalid_count

    indices = np.arange(len(arr))
    valid_indices = indices[~invalid_mask]
    valid_values = arr[~invalid_mask]

    if len(valid_values) == 0:
        return np.zeros_like(arr), invalid_count

    if method == "linear":
        arr[invalid_mask] = np.interp(indices[invalid_mask], valid_indices, valid_values)
    elif method == "nearest":
        nearest_idx = np.abs(indices[invalid_mask, None] - valid_indices[None, :]).argmin(axis=1)
        arr[invalid_mask] = valid_values[nearest_idx]
    elif method == "zero":
        arr[invalid_mask] = 0.0

    return arr, invalid_count


def detrend_signal(
    x: np.ndarray,
    method: Literal["linear", "constant", "none"] = "linear",
) -> np.ndarray:
    """Remove linear trend or DC offset from signal."""
    arr = np.asarray(x, dtype=np.float64)
    if method == "none":
        return arr.copy()
    if method == "constant":
        return arr - np.mean(arr)
    if method == "linear":
        return signal.detrend(arr, type="linear")
    raise ValueError(f"Unknown detrend method: {method}")


def normalize_signal(
    x: np.ndarray,
    method: Literal["standardize", "robust", "minmax", "none"] = "standardize",
) -> np.ndarray:
    """Normalize or scale 1D signal.

    - standardize: (x - mean) / std
    - robust: (x - median) / IQR
    - minmax: (x - min) / (max - min) * 2 - 1
    """
    arr = np.asarray(x, dtype=np.float64)
    if method == "none":
        return arr.copy()

    if method == "standardize":
        std = np.std(arr)
        if std == 0 or np.isnan(std):
            return arr - np.mean(arr)
        return (arr - np.mean(arr)) / std

    if method == "robust":
        median = np.median(arr)
        q25, q75 = np.percentile(arr, [25, 75])
        iqr = q75 - q25
        if iqr == 0:
            return arr - median
        return (arr - median) / (iqr / 1.349)

    if method == "minmax":
        min_val, max_val = np.min(arr), np.max(arr)
        if max_val == min_val:
            return np.zeros_like(arr)
        return 2.0 * (arr - min_val) / (max_val - min_val) - 1.0

    raise ValueError(f"Unknown normalization method: {method}")


def detect_and_clip_outliers(
    x: np.ndarray,
    method: Literal["mad", "zscore"] = "mad",
    threshold: float = 5.0,
) -> tuple[np.ndarray, int]:
    """Detect and clip extreme statistical outliers.

    Args:
        x: 1D signal.
        method: 'mad' (Median Absolute Deviation) or 'zscore'.
        threshold: Cutoff multiplier (default 5.0).

    Returns:
        Clipped array and number of samples clipped.
    """
    arr = np.asarray(x, dtype=np.float64).copy()
    if method == "mad":
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if mad < 1e-12:
            # Fallback to mean absolute deviation for sparse arrays
            mad = np.mean(np.abs(arr - median))
            if mad < 1e-12:
                return arr, 0
        score = np.abs(arr - median) / (1.4826 * mad)
        upper = median + threshold * 1.4826 * mad
        lower = median - threshold * 1.4826 * mad
    elif method == "zscore":
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-12:
            return arr, 0
        score = np.abs(arr - mean) / std
        upper = mean + threshold * std
        lower = mean - threshold * std
    else:
        raise ValueError(f"Unknown outlier method: {method}")

    outliers = score > threshold
    outlier_count = int(np.sum(outliers))
    if outlier_count > 0:
        arr = np.clip(arr, lower, upper)

    return arr, outlier_count


def design_filter(
    filter_type: Literal["bandpass", "lowpass", "highpass", "notch"],
    sampling_rate: float,
    cutoff: float | tuple[float, float],
    order: int = 4,
    q_factor: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Design a digital Butterworth or Notch filter.

    Returns (b, a) transfer function coefficients.
    """
    nyq = 0.5 * sampling_rate

    if filter_type == "notch":
        # Notch filter for specific harmonic (e.g. 60Hz or local inter-harmonic)
        w0 = float(cutoff) / nyq if isinstance(cutoff, (int, float)) else float(cutoff[0]) / nyq
        if w0 >= 1.0 or w0 <= 0.0:
            raise ValueError(f"Notch frequency {cutoff} must be within 0 and Nyquist {nyq}")
        b, a = signal.iirnotch(w0, q_factor)
        return b, a

    if filter_type == "bandpass":
        if not isinstance(cutoff, (list, tuple)) or len(cutoff) != 2:
            raise ValueError("Bandpass cutoff must be a tuple/list of (low, high) frequencies.")
        low, high = cutoff
        low_norm = max(low / nyq, 1e-4)
        high_norm = min(high / nyq, 0.9999)
        if low_norm >= high_norm:
            raise ValueError(f"Low cutoff {low} must be < high cutoff {high}")
        b, a = signal.butter(order, [low_norm, high_norm], btype="bandpass")
        return b, a

    if filter_type == "lowpass":
        fc = float(cutoff) if isinstance(cutoff, (int, float)) else float(cutoff[0])
        fc_norm = min(fc / nyq, 0.9999)
        b, a = signal.butter(order, fc_norm, btype="lowpass")
        return b, a

    if filter_type == "highpass":
        fc = float(cutoff) if isinstance(cutoff, (int, float)) else float(cutoff[0])
        fc_norm = max(fc / nyq, 1e-4)
        b, a = signal.butter(order, fc_norm, btype="highpass")
        return b, a

    raise ValueError(f"Unknown filter type: {filter_type}")


def apply_filter(
    x: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    zero_phase: bool = True,
) -> np.ndarray:
    """Apply IIR/FIR filter with zero-phase filtering (filtfilt) by default."""
    arr = np.asarray(x, dtype=np.float64)
    # Ensure signal is sufficiently long for filtfilt
    padlen = max(len(b), len(a)) * 3
    if len(arr) <= padlen or not zero_phase:
        return signal.lfilter(b, a, arr)
    return signal.filtfilt(b, a, arr)


def resample_with_antialiasing(
    x: np.ndarray,
    orig_sr: float,
    target_sr: float,
) -> np.ndarray:
    """Resample signal with proper anti-aliasing low-pass filter."""
    if np.isclose(orig_sr, target_sr):
        return np.asarray(x, dtype=np.float64).copy()

    arr = np.asarray(x, dtype=np.float64)
    num_samples = int(np.round(len(arr) * target_sr / orig_sr))

    # signal.resample applies Fourier method with anti-aliasing windowing
    return signal.resample(arr, num_samples)


def segment_signal(
    x: np.ndarray,
    window_samples: int,
    overlap_samples: int,
) -> list[np.ndarray]:
    """Segment a 1D signal into overlapping windows.

    Returns a list of window arrays.
    """
    arr = np.asarray(x, dtype=np.float64)
    step = window_samples - overlap_samples
    if step <= 0:
        raise ValueError("Overlap must be strictly less than window length.")

    segments = []
    for start in range(0, len(arr) - window_samples + 1, step):
        segments.append(arr[start : start + window_samples])
    return segments
