"""Envelope extraction, exponential curve fitting, and damping estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.optimize as opt
import scipy.signal as signal


@dataclass
class EnvelopeFitResult:
    """Results of exponential envelope fitting and damping estimation."""

    initial_amplitude: float
    damping_coefficient: float  # alpha in A * exp(-alpha * t)
    damping_ratio: float  # zeta = alpha / sqrt(alpha^2 + omega^2)
    oscillation_frequency: float  # f (Hz)
    decay_rate: float  # alpha if > 0 else 0
    growth_rate: float  # -alpha if alpha < 0 else 0
    r_squared: float
    rmse: float
    confidence: float
    is_damped: bool
    is_growing: bool
    is_sustained: bool


def extract_hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """Extract amplitude envelope using the magnitude of the analytic signal."""
    arr = np.asarray(x, dtype=np.float64)
    analytic = signal.hilbert(arr)
    return np.abs(analytic)


def extract_peak_envelope(
    x: np.ndarray,
    distance: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract upper and lower envelopes by interpolating local extrema.

    Returns:
        upper_envelope, lower_envelope, peak_indices.
    """
    arr = np.asarray(x, dtype=np.float64)
    peaks, _ = signal.find_peaks(arr, distance=distance)
    troughs, _ = signal.find_peaks(-arr, distance=distance)

    indices = np.arange(len(arr))

    if len(peaks) >= 2:
        upper = np.interp(indices, peaks, arr[peaks])
    else:
        upper = np.full_like(arr, np.max(arr))

    if len(troughs) >= 2:
        lower = np.interp(indices, troughs, arr[troughs])
    else:
        lower = np.full_like(arr, np.min(arr))

    return upper, lower, peaks


def fit_exponential_envelope(
    envelope: np.ndarray,
    sampling_rate: float,
    oscillation_frequency: float,
    start_index: int = 0,
    end_index: int | None = None,
) -> EnvelopeFitResult:
    """Fit exponential model: y(t) = A * exp(-alpha * t) to the amplitude envelope.

    Calculates damping ratio:
        zeta = alpha / sqrt(alpha^2 + (2 * pi * f)^2)
    """
    if end_index is None:
        end_index = len(envelope)

    # Detect disturbance onset (envelope peak) if starting from beginning
    if start_index == 0 and len(envelope) > 20:
        peak_idx = int(np.argmax(envelope))
        # If peak is within the first 65% of the signal, start decay fit from peak
        if 0 < peak_idx < int(len(envelope) * 0.65):
            start_index = peak_idx

    y = np.asarray(envelope[start_index:end_index], dtype=np.float64)
    n = len(y)
    if n < 10 or np.max(y) <= 1e-12:
        return EnvelopeFitResult(
            initial_amplitude=0.0,
            damping_coefficient=0.0,
            damping_ratio=0.0,
            oscillation_frequency=oscillation_frequency,
            decay_rate=0.0,
            growth_rate=0.0,
            r_squared=0.0,
            rmse=0.0,
            confidence=0.0,
            is_damped=False,
            is_growing=False,
            is_sustained=True,
        )

    t = np.arange(n) / sampling_rate
    a_init = float(np.percentile(y, 90))

    def exp_func(t_vals: np.ndarray, a: float, alpha: float) -> np.ndarray:
        return a * np.exp(-alpha * t_vals)

    try:
        popt, _ = opt.curve_fit(
            exp_func,
            t,
            y,
            p0=[a_init, 0.05],
            bounds=([0.0, -10.0], [a_init * 10.0, 20.0]),
            maxfev=1000,
        )
        fitted_a, alpha = float(popt[0]), float(popt[1])
        y_pred = exp_func(t, fitted_a, alpha)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    except Exception:
        # Fallback to linear fit of log envelope for positive values
        positive_mask = y > 1e-6
        if np.sum(positive_mask) > 10:
            log_y = np.log(y[positive_mask])
            slope, intercept = np.polyfit(t[positive_mask], log_y, 1)
            alpha = -float(slope)
            fitted_a = float(np.exp(intercept))
            r_squared = 0.5
            rmse = float(np.std(y))
        else:
            alpha = 0.0
            fitted_a = a_init
            r_squared = 0.0
            rmse = float(np.std(y))

    # Calculate damping ratio zeta = alpha / sqrt(alpha^2 + omega^2)
    omega = 2.0 * np.pi * max(oscillation_frequency, 0.01)
    denom = np.sqrt(alpha**2 + omega**2)
    damping_ratio = float(alpha / denom) if denom > 1e-8 else 0.0

    decay_rate = max(0.0, alpha)
    growth_rate = max(0.0, -alpha)

    # Classifications
    is_damped = damping_ratio > 0.03 and r_squared > 0.40
    is_growing = damping_ratio < -0.01 and r_squared > 0.40
    is_sustained = abs(damping_ratio) <= 0.03 or r_squared < 0.30

    # Confidence calculation: depends on goodness of fit, duration in cycles
    num_cycles = (n / sampling_rate) * max(oscillation_frequency, 0.1)
    cycle_factor = min(1.0, num_cycles / 5.0)
    confidence = float(np.clip(r_squared * cycle_factor, 0.0, 1.0))

    return EnvelopeFitResult(
        initial_amplitude=fitted_a,
        damping_coefficient=alpha,
        damping_ratio=damping_ratio,
        oscillation_frequency=oscillation_frequency,
        decay_rate=decay_rate,
        growth_rate=growth_rate,
        r_squared=r_squared,
        rmse=rmse,
        confidence=confidence,
        is_damped=is_damped,
        is_growing=is_growing,
        is_sustained=is_sustained,
    )
