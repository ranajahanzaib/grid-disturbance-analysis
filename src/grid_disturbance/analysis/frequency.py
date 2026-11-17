"""Power grid frequency stability and excursion analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.signal as signal


@dataclass
class FrequencyInstabilityReport:
    """Quantitative characterization of frequency excursions, drift, and RoCoF."""

    is_frequency_instability: bool
    initial_rocof_hz_per_sec: float
    max_rocof_hz_per_sec: float
    frequency_nadir_hz: float | None = None
    frequency_zenith_hz: float | None = None
    max_frequency_deviation_hz: float = 0.0
    settling_frequency_deviation_hz: float = 0.0
    drift_rate_hz_per_sec: float = 0.0
    has_frequency_oscillation: bool = False
    oscillation_frequency: float | None = None
    severity: str = "normal"  # "normal", "moderate_excursion", "severe_instability"
    summary_points: list[str] = field(default_factory=list)


def analyze_frequency_instability(
    freq_signal: np.ndarray,
    sampling_rate: float = 30.0,
    nominal_frequency: float = 60.0,
    rocof_window_samples: int = 15,  # 0.5 sec window at 30 Hz
) -> FrequencyInstabilityReport:
    """Analyze frequency excursions, RoCoF, drift, and distinguish from oscillations."""
    arr = np.asarray(freq_signal, dtype=np.float64)
    n = len(arr)
    dt = 1.0 / sampling_rate

    # If input is already centered around 0 (deviation), reconstruct or compute delta
    if np.abs(np.mean(arr)) < 10.0:
        delta_f = arr.copy()
        f_abs = nominal_frequency + delta_f
    else:
        f_abs = arr.copy()
        delta_f = f_abs - nominal_frequency

    # 1. Compute RoCoF via moving slope or Savitzky-Golay derivative
    if n >= 9:
        rocof = signal.savgol_filter(
            f_abs, window_length=min(15, n if n % 2 != 0 else n - 1), polyorder=2, deriv=1, delta=dt
        )
    else:
        rocof = np.gradient(f_abs, dt)

    initial_rocof = float(rocof[min(rocof_window_samples, n - 1)])
    max_rocof = float(np.max(np.abs(rocof)))

    # 2. Extreme deviations (Nadir and Zenith)
    min_val = float(np.min(f_abs))
    max_val = float(np.max(f_abs))
    max_dev = float(np.max(np.abs(delta_f)))

    nadir = min_val if min_val < (nominal_frequency - 0.05) else None
    zenith = max_val if max_val > (nominal_frequency + 0.05) else None

    # 3. Settling deviation (mean of last 15% of samples)
    tail_len = max(5, int(n * 0.15))
    settling_dev = float(np.mean(delta_f[-tail_len:]))

    # 4. Long-term drift rate (linear slope across entire window)
    t = np.arange(n) * dt
    slope, _ = np.polyfit(t, f_abs, 1)
    drift_rate = float(slope)

    # 5. Check whether frequency exhibits oscillatory behavior
    detrended = signal.detrend(delta_f)
    fft_vals = np.abs(np.fft.rfft(detrended * signal.windows.hann(n)))
    freqs = np.fft.rfftfreq(n, d=dt)
    mask = (freqs >= 0.05) & (freqs <= 3.0)
    has_osc = False
    osc_f = None
    if np.sum(mask) > 0 and np.max(fft_vals[mask]) > (0.15 * np.sum(fft_vals)):
        has_osc = True
        osc_f = float(freqs[mask][np.argmax(fft_vals[mask])])

    # Assess severity: distinguish true frequency excursion from pure AC oscillation
    summary: list[str] = []
    is_instability = False
    severity = "normal"

    # A genuine frequency excursion involves sustained DC shift, high RoCoF, or monotonic drift
    has_dc_excursion = (
        abs(settling_dev) >= 0.08
        or abs(drift_rate) >= 0.02
        or (nadir is not None and nadir < nominal_frequency - 0.20 and abs(settling_dev) > 0.05)
    )

    if (max_dev >= 0.50 or abs(max_rocof) >= 0.80) and has_dc_excursion:
        is_instability = True
        severity = "severe_instability"
        summary.append(
            f"Severe frequency excursion: max deviation {max_dev:.3f} Hz, peak RoCoF {max_rocof:.3f} Hz/s."
        )
    elif (
        max_dev >= 0.15 or abs(max_rocof) >= 0.30 or abs(drift_rate) >= 0.03
    ) and has_dc_excursion:
        is_instability = True
        severity = "moderate_excursion"
        summary.append(
            f"Moderate frequency disturbance: deviation {max_dev:.3f} Hz, RoCoF {max_rocof:.3f} Hz/s."
        )
    else:
        summary.append(
            f"Frequency remains within normal operating tolerances or represents stationary oscillatory deviation (max dev {max_dev:.3f} Hz)."
        )

    if nadir is not None:
        summary.append(
            f"Frequency nadir reached at {nadir:.3f} Hz ({nadir - nominal_frequency:+.3f} Hz)."
        )
    if zenith is not None:
        summary.append(
            f"Frequency zenith reached at {zenith:.3f} Hz ({zenith - nominal_frequency:+.3f} Hz)."
        )
    if has_osc and osc_f:
        summary.append(f"Superimposed frequency oscillation detected at {osc_f:.2f} Hz.")

    return FrequencyInstabilityReport(
        is_frequency_instability=is_instability,
        initial_rocof_hz_per_sec=initial_rocof,
        max_rocof_hz_per_sec=max_rocof,
        frequency_nadir_hz=nadir,
        frequency_zenith_hz=zenith,
        max_frequency_deviation_hz=max_dev,
        settling_frequency_deviation_hz=settling_dev,
        drift_rate_hz_per_sec=drift_rate,
        has_frequency_oscillation=has_osc,
        oscillation_frequency=osc_f,
        severity=severity,
        summary_points=summary,
    )
