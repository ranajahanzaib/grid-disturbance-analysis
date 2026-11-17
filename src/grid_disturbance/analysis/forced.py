"""Forced oscillation detection and characterization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.signal as signal

from grid_disturbance.signals.envelopes import extract_hilbert_envelope
from grid_disturbance.signals.oscillations import estimate_frequency_drift
from grid_disturbance.signals.spectral import analyze_spectrum


@dataclass
class ForcedOscillationEvidence:
    """Quantitative evidence supporting or contradicting forced oscillation hypothesis."""

    is_forced: bool
    likelihood: float  # [0.0, 1.0]
    dominant_frequency: float
    persistence_ratio: float  # Fraction of time dominant frequency persists
    frequency_drift_hz: float
    spectral_concentration: float
    envelope_coefficient_of_variation: float
    has_harmonics: bool
    supporting_points: list[str] = field(default_factory=list)
    contradicting_points: list[str] = field(default_factory=list)
    confidence: float = 0.0


def analyze_forced_oscillation(
    x: np.ndarray,
    sampling_rate: float,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
) -> ForcedOscillationEvidence:
    """Analyze whether an oscillatory disturbance is a forced oscillation.

    Criteria for forced oscillations:
    1. Sustained presence (does not decay exponentially over time).
    2. High spectral concentration / purity (energy focused in a narrow line).
    3. High frequency stability (minimal frequency drift over time).
    4. Low envelope coefficient of variation (nearly constant amplitude).
    5. Presence of harmonic components (often produced by non-linear periodic excitation).
    """
    arr = np.asarray(x, dtype=np.float64)
    n = len(arr)
    if n < int(3 * sampling_rate):
        return ForcedOscillationEvidence(
            is_forced=False,
            likelihood=0.0,
            dominant_frequency=0.0,
            persistence_ratio=0.0,
            frequency_drift_hz=0.0,
            spectral_concentration=0.0,
            envelope_coefficient_of_variation=1.0,
            has_harmonics=False,
            contradicting_points=[
                "Signal duration is too short for reliable persistence analysis."
            ],
            confidence=0.0,
        )

    # 1. Spectral analysis
    spec = analyze_spectrum(arr, sampling_rate)
    f0 = spec.dominant_frequency

    # 2. Sliding window persistence of dominant frequency
    w_len = int(window_sec * sampling_rate)
    s_len = int(step_sec * sampling_rate)
    active_windows = 0
    total_windows = 0

    if len(arr) >= w_len and f0 > 0.05:
        for start in range(0, len(arr) - w_len + 1, s_len):
            total_windows += 1
            seg = arr[start : start + w_len]
            seg_fft = np.abs(np.fft.rfft(seg * signal.windows.hann(len(seg))))
            seg_freqs = np.fft.rfftfreq(len(seg), d=1.0 / sampling_rate)

            # Check if peak in window is within +/-0.08 Hz of overall dominant frequency
            mask = (seg_freqs >= max(0.05, f0 - 0.08)) & (seg_freqs <= f0 + 0.08)
            if np.sum(mask) > 0:
                seg_dom = seg_freqs[np.argmax(seg_fft)]
                if abs(seg_dom - f0) <= 0.08:
                    active_windows += 1

    persistence = (active_windows / total_windows) if total_windows > 0 else 0.0

    # 3. Frequency drift
    drift = estimate_frequency_drift(arr, sampling_rate, window_sec=window_sec, step_sec=step_sec)

    # 4. Envelope variance
    env = extract_hilbert_envelope(arr)
    env_mean = float(np.mean(env))
    env_std = float(np.std(env))
    env_cv = (env_std / env_mean) if env_mean > 1e-12 else 1.0

    # 5. Harmonics check
    has_harmonics = any(p.is_harmonic_of is not None for p in spec.peaks)

    # Scoring evidence
    score = 0.0
    supporting: list[str] = []
    contradicting: list[str] = []

    # Persistence factor (weight: 0.35)
    if persistence >= 0.75:
        score += 0.35
        supporting.append(
            f"Dominant frequency ({f0:.2f} Hz) persisted across {persistence * 100:.0f}% of time windows."
        )
    elif persistence >= 0.50:
        score += 0.20
        supporting.append(
            f"Dominant frequency ({f0:.2f} Hz) observed in {persistence * 100:.0f}% of time windows."
        )
    else:
        contradicting.append(
            f"Dominant frequency was transient; only persisted in {persistence * 100:.0f}% of windows."
        )

    # Spectral concentration factor (weight: 0.25)
    if spec.spectral_concentration >= 0.40:
        score += 0.25
        supporting.append(
            f"High spectral concentration ({spec.spectral_concentration:.2f}), indicating sharp tonal excitation."
        )
    elif spec.spectral_concentration >= 0.25:
        score += 0.15
    else:
        contradicting.append(
            f"Low spectral concentration ({spec.spectral_concentration:.2f}); energy is dispersed across frequencies."
        )

    # Frequency stability (weight: 0.20)
    if drift <= 0.04:
        score += 0.20
        supporting.append(f"Stable frequency with minimal drift ({drift:.3f} Hz).")
    elif drift <= 0.10:
        score += 0.10
    else:
        contradicting.append(
            f"Measurable frequency drift ({drift:.3f} Hz) suggests non-stationary or natural response."
        )

    # Amplitude stability / low CV (weight: 0.15)
    if env_cv <= 0.35:
        score += 0.15
        supporting.append(f"Sustained amplitude with low envelope variation (CV = {env_cv:.2f}).")
    elif env_cv <= 0.55:
        score += 0.05
    else:
        # High envelope CV indicates decay, growth, or transient burst
        score -= 0.35
        contradicting.append(
            f"High envelope variation (CV = {env_cv:.2f}) indicates decay or transient behavior."
        )

    # Check tail attenuation (forced oscillations do not decay to zero)
    q_len = max(5, n // 4)
    first_q_rms = float(np.sqrt(np.mean(arr[:q_len] ** 2)))
    last_q_rms = float(np.sqrt(np.mean(arr[-q_len:] ** 2)))
    if first_q_rms > 1e-4:
        tail_ratio = last_q_rms / first_q_rms
        if tail_ratio < 0.35:
            score -= 0.30
            contradicting.append(
                f"Amplitude strongly decays ({tail_ratio * 100:.0f}% residual tail energy); inconsistent with sustained forced excitation."
            )

    # Harmonics bonus (weight: 0.05)
    if has_harmonics:
        score += 0.05
        supporting.append(
            "Detected harmonic spectral peaks typical of non-linear forced periodic excitation."
        )

    likelihood = float(np.clip(score, 0.0, 1.0))
    is_forced = likelihood >= 0.55 and env_cv < 0.65
    confidence = float(np.clip(abs(likelihood - 0.50) * 2.0, 0.0, 1.0))

    return ForcedOscillationEvidence(
        is_forced=is_forced,
        likelihood=likelihood,
        dominant_frequency=f0,
        persistence_ratio=persistence,
        frequency_drift_hz=drift,
        spectral_concentration=spec.spectral_concentration,
        envelope_coefficient_of_variation=env_cv,
        has_harmonics=has_harmonics,
        supporting_points=supporting,
        contradicting_points=contradicting,
        confidence=confidence,
    )
