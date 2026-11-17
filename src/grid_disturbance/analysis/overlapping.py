"""Multi-mode and overlapping oscillatory disturbance analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from grid_disturbance.signals.decomposition import DecomposedComponent, separate_spectral_components
from grid_disturbance.signals.spectral import analyze_spectrum


@dataclass
class OverlappingEventsResult:
    """Analysis of multiple simultaneous or overlapping oscillatory modes."""

    has_multiple_events: bool
    num_components: int
    components: list[DecomposedComponent] = field(default_factory=list)
    dominant_frequency: float = 0.0
    secondary_frequency: float | None = None
    frequency_separation_hz: float = 0.0
    summary: str = ""


def analyze_overlapping_events(
    x: np.ndarray,
    sampling_rate: float = 30.0,
    min_peak_prominence: float = 0.08,
    min_freq_separation: float = 0.15,
) -> OverlappingEventsResult:
    """Detect and characterize multiple simultaneous oscillatory components."""
    arr = np.asarray(x, dtype=np.float64)
    from grid_disturbance.signals.spectral import SpectralPeak

    spec = analyze_spectrum(arr, sampling_rate)

    # Filter peaks: ignore harmonics of dominant peak to avoid mistaking non-linearities for independent events
    independent_peaks: list[SpectralPeak] = []
    for p in spec.peaks:
        if p.is_harmonic_of is None:
            # Check separation from existing selected peaks
            is_separated = True
            for ip in independent_peaks:
                if abs(p.frequency - ip.frequency) < min_freq_separation:
                    is_separated = False
                    break
            if is_separated:
                independent_peaks.append(p)

    f0 = spec.dominant_frequency
    has_multiple = len(independent_peaks) >= 2
    f_sec = independent_peaks[1].frequency if len(independent_peaks) > 1 else None
    freq_sep = abs(f0 - f_sec) if f_sec is not None else 0.0

    # Separate components
    peak_freqs = [p.frequency for p in independent_peaks[:3]]
    components = separate_spectral_components(arr, sampling_rate, peak_freqs)

    if has_multiple:
        summary = (
            f"Detected {len(independent_peaks)} distinct independent oscillatory modes: "
            f"Primary = {f0:.2f} Hz, Secondary = {f_sec:.2f} Hz (separation: {freq_sep:.2f} Hz)."
        )
    else:
        summary = f"Single dominant mode detected at {f0:.2f} Hz."

    return OverlappingEventsResult(
        has_multiple_events=has_multiple,
        num_components=len(components),
        components=components,
        dominant_frequency=f0,
        secondary_frequency=f_sec,
        frequency_separation_hz=freq_sep,
        summary=summary,
    )
