"""Natural oscillation and ring-down decay analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from grid_disturbance.signals.envelopes import extract_hilbert_envelope, fit_exponential_envelope
from grid_disturbance.signals.spectral import analyze_spectrum


@dataclass
class RingdownEvidence:
    """Quantitative evidence supporting or contradicting a natural ring-down event."""

    is_ringdown: bool
    likelihood: float
    damping_classification: Literal[
        "well_damped", "poorly_damped", "undamped_growing", "indeterminate"
    ]
    damping_ratio: float
    decay_rate: float
    initial_amplitude: float
    oscillation_frequency: float
    duration_sec: float
    envelope_r2: float
    supporting_points: list[str] = field(default_factory=list)
    contradicting_points: list[str] = field(default_factory=list)
    confidence: float = 0.0


def analyze_ringdown(
    x: np.ndarray,
    sampling_rate: float,
    well_damped_threshold: float = 0.05,
    undamped_threshold: float = 0.00,
) -> RingdownEvidence:
    """Analyze whether an oscillatory disturbance exhibits natural ring-down characteristics."""
    arr = np.asarray(x, dtype=np.float64)
    n = len(arr)
    duration = n / sampling_rate

    spec = analyze_spectrum(arr, sampling_rate)
    f0 = spec.dominant_frequency

    env = extract_hilbert_envelope(arr)
    fit_res = fit_exponential_envelope(env, sampling_rate, f0)

    zeta = fit_res.damping_ratio
    r2 = fit_res.r_squared
    alpha = fit_res.decay_rate

    score = 0.0
    supporting: list[str] = []
    contradicting: list[str] = []

    # 1. Exponential fit quality (weight: 0.35)
    if r2 >= 0.70:
        score += 0.35
        supporting.append(f"Strong exponential envelope decay fit (R² = {r2:.2f}).")
    elif r2 >= 0.40:
        score += 0.20
        supporting.append(f"Moderate exponential envelope decay fit (R² = {r2:.2f}).")
    else:
        contradicting.append(
            f"Poor exponential envelope fit (R² = {r2:.2f}); decay does not follow natural mode response."
        )

    # 2. Measurable positive damping (weight: 0.35)
    if zeta >= well_damped_threshold:
        score += 0.35
        supporting.append(
            f"Measurable positive damping ratio (ζ = {zeta * 100:.1f}%), satisfying well-damped criteria (≥5%)."
        )
    elif zeta > undamped_threshold:
        score += 0.25
        supporting.append(
            f"Low positive damping ratio (ζ = {zeta * 100:.1f}%), indicating a poorly-damped natural mode."
        )
    else:
        contradicting.append(
            f"Zero or negative damping (ζ = {zeta * 100:.1f}%), incompatible with a stable natural ring-down."
        )

    # 3. Energy attenuation over time (weight: 0.30)
    # Compare first quarter vs last quarter RMS
    q_len = max(5, n // 4)
    first_q_rms = np.sqrt(np.mean(arr[:q_len] ** 2))
    last_q_rms = np.sqrt(np.mean(arr[-q_len:] ** 2))
    attenuation_ratio = (last_q_rms / first_q_rms) if first_q_rms > 1e-12 else 1.0

    if attenuation_ratio <= 0.30:
        score += 0.30
        supporting.append(
            f"Significant amplitude attenuation ({attenuation_ratio * 100:.0f}% residual energy at tail)."
        )
    elif attenuation_ratio <= 0.60:
        score += 0.15
        supporting.append(
            f"Moderate amplitude attenuation ({attenuation_ratio * 100:.0f}% residual energy at tail)."
        )
    else:
        contradicting.append(
            f"No significant tail attenuation ({attenuation_ratio * 100:.0f}% residual energy); oscillation is sustained."
        )

    likelihood = float(np.clip(score, 0.0, 1.0))
    is_ringdown = likelihood >= 0.55

    # Damping classification
    d_class: Literal["well_damped", "poorly_damped", "undamped_growing", "indeterminate"]
    if not is_ringdown and r2 < 0.30:
        d_class = "indeterminate"
    elif zeta >= well_damped_threshold:
        d_class = "well_damped"
    elif zeta > undamped_threshold:
        d_class = "poorly_damped"
    else:
        d_class = "undamped_growing"

    confidence = float(np.clip(r2 * min(1.0, duration / 5.0), 0.0, 1.0))

    return RingdownEvidence(
        is_ringdown=is_ringdown,
        likelihood=likelihood,
        damping_classification=d_class,
        damping_ratio=zeta,
        decay_rate=alpha,
        initial_amplitude=fit_res.initial_amplitude,
        oscillation_frequency=f0,
        duration_sec=duration,
        envelope_r2=r2,
        supporting_points=supporting,
        contradicting_points=contradicting,
        confidence=confidence,
    )
