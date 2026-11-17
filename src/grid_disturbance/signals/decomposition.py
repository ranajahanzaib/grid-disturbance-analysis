"""Signal decomposition and component separation for multi-mode disturbances."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as signal

from grid_disturbance.signals.filtering import apply_filter, design_filter


@dataclass
class DecomposedComponent:
    """An isolated oscillatory component from multi-event decomposition."""

    component_id: int
    center_frequency: float
    bandwidth: float
    signal_waveform: np.ndarray
    rms_amplitude: float
    energy_percentage: float
    start_time_sec: float
    end_time_sec: float
    is_sustained: bool


def separate_spectral_components(
    x: np.ndarray,
    sampling_rate: float,
    peak_frequencies: list[float],
    bandwidth_ratio: float = 0.20,
    min_bandwidth: float = 0.10,
) -> list[DecomposedComponent]:
    """Separate overlapping oscillatory components using adaptive bandpass filtering.

    Args:
        x: Input signal array.
        sampling_rate: Sampling frequency in Hz.
        peak_frequencies: Center frequencies of identified modes.
        bandwidth_ratio: Fractional bandwidth around center frequency (e.g. 0.20 = +/-10%).
        min_bandwidth: Minimum bandwidth in Hz to prevent filter instability.

    Returns:
        List of DecomposedComponent objects.
    """
    arr = np.asarray(x, dtype=np.float64)
    total_energy = float(np.sum(arr**2))
    n = len(arr)
    duration = n / sampling_rate
    nyq = 0.5 * sampling_rate

    components: list[DecomposedComponent] = []

    for i, fc in enumerate(peak_frequencies):
        if fc <= 0.05 or fc >= nyq:
            continue

        half_bw = max(min_bandwidth / 2.0, (fc * bandwidth_ratio) / 2.0)
        low_cut = max(0.05, fc - half_bw)
        high_cut = min(nyq - 0.1, fc + half_bw)

        if low_cut >= high_cut:
            continue

        try:
            b, a = design_filter(
                filter_type="bandpass",
                sampling_rate=sampling_rate,
                cutoff=(low_cut, high_cut),
                order=3,
            )
            filtered = apply_filter(arr, b, a, zero_phase=True)
        except Exception:
            continue

        comp_energy = float(np.sum(filtered**2))
        energy_pct = (comp_energy / total_energy * 100.0) if total_energy > 0 else 0.0
        rms_amp = float(np.sqrt(np.mean(filtered**2)))

        # Detect active duration using envelope threshold (above 15% max envelope)
        env = np.abs(signal.hilbert(filtered))
        max_env = np.max(env)
        active_mask = env > (0.15 * max_env) if max_env > 1e-12 else np.zeros(n, dtype=bool)

        if np.any(active_mask):
            active_indices = np.where(active_mask)[0]
            start_t = float(active_indices[0] / sampling_rate)
            end_t = float(active_indices[-1] / sampling_rate)
        else:
            start_t = 0.0
            end_t = duration

        # Sustained check: active for > 70% of signal duration
        is_sustained = (end_t - start_t) >= (0.70 * duration)

        components.append(
            DecomposedComponent(
                component_id=i + 1,
                center_frequency=float(fc),
                bandwidth=float(high_cut - low_cut),
                signal_waveform=filtered,
                rms_amplitude=rms_amp,
                energy_percentage=energy_pct,
                start_time_sec=start_t,
                end_time_sec=end_t,
                is_sustained=is_sustained,
            )
        )

    # Sort components by descending energy
    components.sort(key=lambda c: c.energy_percentage, reverse=True)
    return components
