"""Time-frequency transforms: STFT, spectrograms, and Hilbert analytic signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as signal


@dataclass
class STFTResult:
    """Result of Short-Time Fourier Transform."""

    times: np.ndarray
    frequencies: np.ndarray
    stft_matrix: np.ndarray  # Complex STFT matrix
    spectrogram: np.ndarray  # |STFT|^2 power


@dataclass
class AnalyticSignalResult:
    """Result of Hilbert transform analytic signal analysis."""

    analytic_signal: np.ndarray  # Complex z(t) = x(t) + j H{x(t)}
    envelope: np.ndarray  # Instantaneous amplitude A(t) = |z(t)|
    phase: np.ndarray  # Instantaneous unwrapped phase phi(t)
    instantaneous_frequency: np.ndarray  # f(t) = (1 / 2*pi) * dphi/dt


def compute_stft(
    x: np.ndarray,
    sampling_rate: float,
    nperseg: int = 128,
    noverlap: int = 96,
    window: str = "hann",
) -> STFTResult:
    """Compute Short-Time Fourier Transform and power spectrogram."""
    arr = np.asarray(x, dtype=np.float64)
    n = len(arr)
    nperseg = min(nperseg, n)
    if noverlap >= nperseg:
        noverlap = nperseg // 2

    freqs, times, zxx = signal.stft(
        arr,
        fs=sampling_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
    )
    spectrogram = np.abs(zxx) ** 2

    return STFTResult(
        times=times,
        frequencies=freqs,
        stft_matrix=zxx,
        spectrogram=spectrogram,
    )


def compute_spectrogram_matrix(
    x: np.ndarray,
    sampling_rate: float,
    output_shape: tuple[int, int] = (64, 64),
    nperseg: int = 128,
    noverlap: int = 96,
) -> np.ndarray:
    """Generate fixed-size 2D spectrogram matrix suitable for CNN input or visualization."""
    stft_res = compute_stft(x, sampling_rate, nperseg=nperseg, noverlap=noverlap)
    spec = stft_res.spectrogram

    # Convert to log scale (dB) with dynamic range clipping
    spec_db = 10.0 * np.log10(np.maximum(spec, 1e-12))
    spec_db = np.clip(spec_db, np.max(spec_db) - 60.0, np.max(spec_db))

    # Normalize to [0, 1]
    denom = np.max(spec_db) - np.min(spec_db)
    if denom > 1e-6:
        norm_spec = (spec_db - np.min(spec_db)) / denom
    else:
        norm_spec = np.zeros_like(spec_db)

    # Resize to output_shape using 2D interpolation
    from scipy.ndimage import zoom

    zoom_factors = (output_shape[0] / norm_spec.shape[0], output_shape[1] / norm_spec.shape[1])
    resized = zoom(norm_spec, zoom_factors, order=1)
    return np.asarray(resized, dtype=np.float32)


def compute_analytic_signal(
    x: np.ndarray,
    sampling_rate: float,
) -> AnalyticSignalResult:
    """Compute analytic signal, instantaneous amplitude, and instantaneous frequency."""
    arr = np.asarray(x, dtype=np.float64)
    z = signal.hilbert(arr)
    envelope = np.abs(z)

    # Unwrapped phase
    phase = np.unwrap(np.angle(z))

    # Instantaneous frequency: (1 / 2*pi) * d(phase)/dt
    dt = 1.0 / sampling_rate
    diff_phase = np.diff(phase)
    inst_freq = np.concatenate(([diff_phase[0]], diff_phase)) / (2.0 * np.pi * dt)

    # Median filter instantaneous frequency to suppress numerical phase jumps
    if len(inst_freq) > 5:
        inst_freq = signal.medfilt(inst_freq, kernel_size=5)

    return AnalyticSignalResult(
        analytic_signal=z,
        envelope=envelope,
        phase=phase,
        instantaneous_frequency=inst_freq,
    )
