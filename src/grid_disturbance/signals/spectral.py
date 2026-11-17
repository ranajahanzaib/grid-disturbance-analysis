"""Frequency-domain spectral analysis and feature extraction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.signal as signal


@dataclass
class SpectralPeak:
    """Represents an identified spectral peak."""

    frequency: float
    magnitude: float
    power: float
    prominence: float
    width: float
    is_harmonic_of: float | None = None
    harmonic_order: int | None = None


@dataclass
class SpectralResult:
    """Full spectral analysis results."""

    frequencies: np.ndarray
    psd: np.ndarray
    fft_magnitudes: np.ndarray
    dominant_frequency: float
    dominant_power: float
    peaks: list[SpectralPeak] = field(default_factory=list)
    spectral_centroid: float = 0.0
    spectral_bandwidth: float = 0.0
    spectral_entropy: float = 0.0
    spectral_flatness: float = 0.0
    spectral_concentration: float = 0.0
    total_energy: float = 0.0


def compute_fft(
    x: np.ndarray,
    sampling_rate: float,
    zero_padding_factor: int = 2,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute one-sided Fast Fourier Transform (FFT).

    Returns:
        frequencies, magnitudes, phase (radians).
    """
    arr = np.asarray(x, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    win = signal.get_window(window, n) if window != "none" else np.ones(n)
    windowed = arr * win

    n_fft = n * zero_padding_factor
    fft_vals = np.fft.rfft(windowed, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sampling_rate)

    # Scaling factor for amplitude preservation
    scale = 2.0 / np.sum(win) if np.sum(win) > 0 else 2.0 / n
    magnitudes = np.abs(fft_vals) * scale
    phases = np.angle(fft_vals)

    return freqs, magnitudes, phases


def compute_welch_psd(
    x: np.ndarray,
    sampling_rate: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch Power Spectral Density (PSD).

    Returns:
        frequencies, psd.
    """
    arr = np.asarray(x, dtype=np.float64)
    if len(arr) < 8:
        return np.array([0.0]), np.array([0.0])

    nperseg = min(nperseg, len(arr))
    if noverlap is None:
        noverlap = nperseg // 2
    else:
        noverlap = min(noverlap, nperseg - 1)

    freqs, psd = signal.welch(
        arr,
        fs=sampling_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
    )
    return freqs, psd


def extract_spectral_peaks(
    frequencies: np.ndarray,
    power: np.ndarray,
    min_prominence: float = 0.05,
    top_k: int = 5,
    freq_range: tuple[float, float] = (0.05, 14.0),
) -> list[SpectralPeak]:
    """Extract prominent spectral peaks and identify harmonic relationships."""
    if len(frequencies) == 0 or len(power) == 0:
        return []

    # Mask within frequency range
    mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
    sub_freqs = frequencies[mask]
    sub_power = power[mask]

    if len(sub_freqs) < 3 or np.max(sub_power) <= 1e-12:
        return []

    # Relative prominence scaling
    max_p = np.max(sub_power)
    norm_power = sub_power / max_p

    peak_indices, properties = signal.find_peaks(
        norm_power,
        prominence=min_prominence,
        distance=max(1, int(len(sub_freqs) * 0.01)),
    )

    if len(peak_indices) == 0:
        # Fallback to absolute maximum
        idx_max = int(np.argmax(sub_power))
        return [
            SpectralPeak(
                frequency=float(sub_freqs[idx_max]),
                magnitude=float(np.sqrt(sub_power[idx_max])),
                power=float(sub_power[idx_max]),
                prominence=1.0,
                width=0.0,
            )
        ]

    prominences = properties["prominences"]
    widths, _, _, _ = signal.peak_widths(norm_power, peak_indices, rel_height=0.5)

    freq_step = sub_freqs[1] - sub_freqs[0] if len(sub_freqs) > 1 else 1.0

    peaks: list[SpectralPeak] = []
    for idx, prom, w in zip(peak_indices, prominences, widths, strict=False):
        peaks.append(
            SpectralPeak(
                frequency=float(sub_freqs[idx]),
                magnitude=float(np.sqrt(sub_power[idx])),
                power=float(sub_power[idx]),
                prominence=float(prom),
                width=float(w * freq_step),
            )
        )

    # Sort descending by power
    peaks.sort(key=lambda p: p.power, reverse=True)
    peaks = peaks[:top_k]

    # Identify harmonic relationships relative to the primary dominant peak
    if len(peaks) > 1:
        f0 = peaks[0].frequency
        if f0 > 0.02:
            for p in peaks[1:]:
                ratio = p.frequency / f0
                nearest_int = round(ratio)
                if nearest_int >= 2 and abs(ratio - nearest_int) < 0.08:
                    p.is_harmonic_of = f0
                    p.harmonic_order = nearest_int

    return peaks


def compute_spectral_centroid_and_spread(
    frequencies: np.ndarray,
    power: np.ndarray,
) -> tuple[float, float]:
    """Calculate spectral centroid and spectral spread (bandwidth)."""
    p_sum = np.sum(power)
    if p_sum <= 1e-14:
        return 0.0, 0.0

    centroid = float(np.sum(frequencies * power) / p_sum)
    variance = float(np.sum(((frequencies - centroid) ** 2) * power) / p_sum)
    spread = float(np.sqrt(max(0.0, variance)))
    return centroid, spread


def compute_spectral_entropy(power: np.ndarray) -> float:
    """Calculate spectral entropy (measure of signal randomness/complexity)."""
    p_sum = np.sum(power)
    if p_sum <= 1e-14:
        return 0.0

    prob = power / p_sum
    # Mask out zeros to avoid log(0)
    prob_nonzero = prob[prob > 1e-14]
    entropy = -float(np.sum(prob_nonzero * np.log2(prob_nonzero)))

    # Normalize by max possible entropy log2(N)
    max_entropy = np.log2(len(power)) if len(power) > 1 else 1.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def compute_spectral_flatness(power: np.ndarray) -> float:
    """Calculate spectral flatness (Wiener entropy: geometric mean / arithmetic mean).

    Values near 1 indicate white noise; values near 0 indicate tonal / peaked spectrum.
    """
    arr = np.asarray(power, dtype=np.float64)
    if len(arr) == 0 or np.max(arr) <= 1e-14:
        return 0.0

    p_norm = arr / np.max(arr)
    p_floor = np.maximum(p_norm, 1e-12)
    geometric_mean = np.exp(np.mean(np.log(p_floor)))
    arithmetic_mean = np.mean(p_floor)

    return float(np.clip(geometric_mean / arithmetic_mean, 0.0, 1.0))


def compute_spectral_concentration(
    power: np.ndarray,
    peak_indices: list[int] | None = None,
    bandwidth_bins: int = 3,
) -> float:
    """Calculate ratio of power concentrated around dominant peak(s) vs total power.

    High concentration (~0.5 - 0.95) indicates single-mode oscillation (e.g. forced oscillation).
    """
    total = np.sum(power)
    if total <= 1e-14:
        return 0.0

    if peak_indices is None or len(peak_indices) == 0:
        dom_idx = int(np.argmax(power))
        peak_indices = [dom_idx]

    concentrated = 0.0
    n = len(power)
    for idx in peak_indices[:2]:
        start = max(0, idx - bandwidth_bins)
        end = min(n, idx + bandwidth_bins + 1)
        concentrated += np.sum(power[start:end])

    return float(np.clip(concentrated / total, 0.0, 1.0))


def analyze_spectrum(
    x: np.ndarray,
    sampling_rate: float,
    nperseg: int = 256,
    zero_padding: int = 2,
    freq_range: tuple[float, float] = (0.05, 14.0),
) -> SpectralResult:
    """Complete spectral analysis pipeline combining FFT, Welch PSD, and metrics."""
    freqs, psd = compute_welch_psd(x, sampling_rate, nperseg=nperseg)
    fft_freqs, fft_mag, _ = compute_fft(x, sampling_rate, zero_padding_factor=zero_padding)

    peaks = extract_spectral_peaks(freqs, psd, freq_range=freq_range)
    dom_freq = peaks[0].frequency if len(peaks) > 0 else 0.0
    dom_power = peaks[0].power if len(peaks) > 0 else 0.0

    centroid, spread = compute_spectral_centroid_and_spread(freqs, psd)
    entropy = compute_spectral_entropy(psd)
    flatness = compute_spectral_flatness(psd)
    concentration = compute_spectral_concentration(psd)
    total_energy = float(np.sum(x**2))

    return SpectralResult(
        frequencies=freqs,
        psd=psd,
        fft_magnitudes=fft_mag,
        dominant_frequency=dom_freq,
        dominant_power=dom_power,
        peaks=peaks,
        spectral_centroid=centroid,
        spectral_bandwidth=spread,
        spectral_entropy=entropy,
        spectral_flatness=flatness,
        spectral_concentration=concentration,
        total_energy=total_energy,
    )


def compute_autocorrelation(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Compute normalized 1D autocorrelation."""
    arr = np.asarray(x, dtype=np.float64)
    arr = arr - np.mean(arr)
    var = np.var(arr)
    if var <= 1e-14 or len(arr) == 0:
        return np.ones(1)

    corr = signal.correlate(arr, arr, mode="full", method="auto")
    mid = len(corr) // 2
    r_xx = corr[mid:] / (len(arr) * var)

    if max_lag is not None:
        r_xx = r_xx[: max_lag + 1]

    return r_xx


def compute_cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cross-correlation and lag vector between two channels."""
    arr_x = np.asarray(x, dtype=np.float64)
    arr_y = np.asarray(y, dtype=np.float64)

    arr_x = arr_x - np.mean(arr_x)
    arr_y = arr_y - np.mean(arr_y)

    std_x = np.std(arr_x)
    std_y = np.std(arr_y)
    denom = (std_x * std_y * len(arr_x)) if (std_x > 0 and std_y > 0) else 1.0

    corr = signal.correlate(arr_x, arr_y, mode="full", method="auto") / denom
    lags = signal.correlation_lags(len(arr_x), len(arr_y), mode="full")

    if max_lag is not None:
        mask = np.abs(lags) <= max_lag
        corr = corr[mask]
        lags = lags[mask]

    return lags, corr
