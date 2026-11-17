"""Modal parameter estimation, Prony/Matrix Pencil analysis, and modulation characterization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as la
import scipy.signal as signal

from grid_disturbance.signals.envelopes import fit_exponential_envelope


@dataclass
class OscillationMode:
    """An estimated oscillatory mode."""

    frequency: float  # Frequency in Hz
    damping_ratio: float  # Dimensionless damping ratio zeta
    damping_factor: float  # Real pole component sigma (decay rate 1/s)
    amplitude: float  # Initial amplitude
    phase: float  # Phase in radians
    energy_fraction: float  # Estimated fraction of total signal energy
    is_stable: bool  # True if damping_factor > 0


@dataclass
class OscillationCharacterization:
    """Complete oscillation characterization result."""

    modes: list[OscillationMode] = field(default_factory=list)
    dominant_frequency: float = 0.0
    primary_damping_ratio: float = 0.0
    frequency_drift_hz: float = 0.0
    amplitude_modulation_index: float = 0.0
    is_sustained: bool = False
    is_damped: bool = False
    is_growing: bool = False
    mode_count: int = 0
    confidence: float = 0.0


def matrix_pencil_modal_estimation(
    x: np.ndarray,
    sampling_rate: float,
    num_modes: int = 3,
    p_pencil_parameter: int | None = None,
) -> list[OscillationMode]:
    """Estimate modal frequencies and damping using the Matrix Pencil Method.

    Matrix Pencil is numerically superior and more noise-robust than classical Prony.
    """
    arr = np.asarray(x, dtype=np.float64)
    n = len(arr)
    if n < 20:
        return []

    # Downsample or truncate if too large for matrix operations (matrix pencil is O(N^3))
    if n > 600:
        step = int(np.ceil(n / 600))
        arr = arr[::step]
        dt = step / sampling_rate
    else:
        dt = 1.0 / sampling_rate

    n = len(arr)
    # Pencil parameter L usually between N/3 and N/2
    l_param = n // 3 if p_pencil_parameter is None else min(p_pencil_parameter, n // 2)
    l_param = max(num_modes + 1, min(l_param, n - num_modes - 1))

    # Construct Hankel data matrix Y: size (N - L) x (L + 1)
    y_mat = la.hankel(arr[: n - l_param], arr[n - l_param - 1 :])

    # SVD of Y
    try:
        u, s, vh = la.svd(y_mat, full_matrices=False)
    except Exception:
        return []

    # Truncate to 2 * num_modes (conjugate pole pairs)
    k = min(2 * num_modes, len(s))
    # Filter singular values to only keep significant energy
    s_thresh = s[0] * 1e-4
    k = max(2, min(k, int(np.sum(s > s_thresh))))

    v_k = vh[:k, :].conj().T

    # Form Y1 and Y2 by removing last and first column of V_k
    y1 = v_k[:-1, :]
    y2 = v_k[1:, :]

    # Solve generalized eigenvalue problem: Y2 - lambda * Y1
    try:
        y1_pinv = la.pinv(y1)
        z_mat = y1_pinv @ y2
        poles = la.eigvals(z_mat)
    except Exception:
        return []

    # Map discrete poles z = exp((sigma + j*omega)*dt) to continuous poles
    modes: list[OscillationMode] = []
    seen_freqs: list[float] = []

    for z in poles:
        if abs(z) < 1e-6 or abs(z) > 10.0:
            continue
        s_pole = np.log(z) / dt
        sigma = -s_pole.real  # Decay rate
        omega = abs(s_pole.imag)
        freq = omega / (2.0 * np.pi)

        # Filter to grid oscillation band (0.05 Hz to Nyquist)
        if freq < 0.05 or freq > (0.5 / dt):
            continue

        # Check for conjugate duplicates
        is_duplicate = False
        for sf in seen_freqs:
            if abs(freq - sf) < 0.05:
                is_duplicate = True
                break
        if is_duplicate:
            continue

        seen_freqs.append(freq)

        denom = np.sqrt(sigma**2 + omega**2)
        zeta = sigma / denom if denom > 1e-6 else 0.0

        modes.append(
            OscillationMode(
                frequency=float(freq),
                damping_ratio=float(zeta),
                damping_factor=float(sigma),
                amplitude=float(abs(z)),
                phase=float(np.angle(z)),
                energy_fraction=1.0 / len(poles),
                is_stable=sigma > 0,
            )
        )

    # Sort by descending frequency or importance
    modes.sort(key=lambda m: m.frequency)
    return modes[:num_modes]


def estimate_frequency_drift(
    x: np.ndarray,
    sampling_rate: float,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
) -> float:
    """Estimate frequency drift (in Hz) by tracking dominant frequency across sliding windows."""
    arr = np.asarray(x, dtype=np.float64)
    w_len = int(window_sec * sampling_rate)
    s_len = int(step_sec * sampling_rate)

    if len(arr) < 2 * w_len:
        return 0.0

    dom_freqs = []
    for start in range(0, len(arr) - w_len + 1, s_len):
        seg = arr[start : start + w_len]
        fft_vals = np.abs(np.fft.rfft(seg * signal.windows.hann(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / sampling_rate)
        # Search within 0.1 to 14 Hz
        mask = (freqs >= 0.1) & (freqs <= 14.0)
        if np.sum(mask) > 0 and np.max(fft_vals[mask]) > 1e-6:
            dom_freqs.append(float(freqs[mask][np.argmax(fft_vals[mask])]))

    if len(dom_freqs) < 2:
        return 0.0

    drift = float(np.max(dom_freqs) - np.min(dom_freqs))
    return drift


def estimate_amplitude_modulation(
    envelope: np.ndarray,
    sampling_rate: float,
) -> tuple[float, float]:
    """Estimate amplitude modulation index and modulation frequency.

    Returns:
        (am_index, mod_frequency_hz)
    """
    env = np.asarray(envelope, dtype=np.float64)
    if len(env) < 20 or np.max(env) <= 1e-12:
        return 0.0, 0.0

    a_max = float(np.percentile(env, 95))
    a_min = float(np.percentile(env, 5))
    denom = a_max + a_min
    am_index = float((a_max - a_min) / denom) if denom > 1e-6 else 0.0

    # Find dominant frequency in the envelope variation
    detrended_env = env - np.mean(env)
    fft_vals = np.abs(np.fft.rfft(detrended_env))
    freqs = np.fft.rfftfreq(len(env), d=1.0 / sampling_rate)

    # Exclude DC, search between 0.02 and 2.0 Hz
    mask = (freqs >= 0.02) & (freqs <= 2.0)
    if np.sum(mask) > 0 and np.max(fft_vals[mask]) > 1e-6:
        mod_freq = float(freqs[mask][np.argmax(fft_vals[mask])])
    else:
        mod_freq = 0.0

    return am_index, mod_freq


def characterize_oscillations(
    x: np.ndarray,
    sampling_rate: float,
    dominant_frequency: float,
) -> OscillationCharacterization:
    """Comprehensive oscillation analysis combining modal estimation, envelope fit, and drift."""
    arr = np.asarray(x, dtype=np.float64)
    analytic = signal.hilbert(arr)
    env = np.abs(analytic)

    # Matrix pencil modes
    modes = matrix_pencil_modal_estimation(arr, sampling_rate, num_modes=3)

    # Exponential envelope fit
    fit_res = fit_exponential_envelope(env, sampling_rate, dominant_frequency)

    # Amplitude modulation
    am_index, _ = estimate_amplitude_modulation(env, sampling_rate)

    # Frequency drift
    drift = estimate_frequency_drift(arr, sampling_rate)

    primary_zeta = fit_res.damping_ratio
    # If matrix pencil found a mode close to dominant frequency, corroborate
    if len(modes) > 0:
        for m in modes:
            if abs(m.frequency - dominant_frequency) < 0.15:
                # Weighted average with envelope fit
                primary_zeta = 0.6 * fit_res.damping_ratio + 0.4 * m.damping_ratio
                break

    return OscillationCharacterization(
        modes=modes,
        dominant_frequency=dominant_frequency,
        primary_damping_ratio=primary_zeta,
        frequency_drift_hz=drift,
        amplitude_modulation_index=am_index,
        is_sustained=fit_res.is_sustained,
        is_damped=fit_res.is_damped,
        is_growing=fit_res.is_growing,
        mode_count=max(1, len(modes)),
        confidence=fit_res.confidence,
    )
