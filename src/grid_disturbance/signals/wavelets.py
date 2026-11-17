"""Continuous and discrete wavelet analysis for power grid oscillations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt


@dataclass
class CWTResult:
    """Continuous Wavelet Transform result."""

    coefficients: np.ndarray  # Complex or real CWT coefficients (scales x time)
    frequencies: np.ndarray  # Pseudo-frequencies corresponding to scales (Hz)
    scales: np.ndarray
    scalogram: np.ndarray  # Normalized power |coefficients|^2


@dataclass
class DWTResult:
    """Discrete Wavelet multiresolution decomposition result."""

    approx_coefficients: np.ndarray
    detail_coefficients: list[np.ndarray]
    level_energies: list[float]
    relative_energies: list[float]


def compute_cwt(
    x: np.ndarray,
    sampling_rate: float,
    wavelet: str = "morl",
    total_scales: int = 64,
    min_freq: float = 0.05,
    max_freq: float = 14.0,
) -> CWTResult:
    """Compute Continuous Wavelet Transform and scalogram."""
    arr = np.asarray(x, dtype=np.float64)

    # Convert frequency bounds to wavelet scales
    # scale = central_freq / (freq * dt)
    dt = 1.0 / sampling_rate
    central_freq = pywt.central_frequency(wavelet)

    min_scale = central_freq / (max_freq * dt)
    max_scale = central_freq / (min_freq * dt)
    scales = np.geomspace(min_scale, max_scale, num=total_scales)

    coefs, freqs = pywt.cwt(arr, scales, wavelet, sampling_period=dt)
    scalogram = np.abs(coefs) ** 2

    # Normalize scalogram
    max_val = np.max(scalogram)
    if max_val > 1e-12:
        scalogram = scalogram / max_val

    return CWTResult(
        coefficients=coefs,
        frequencies=freqs,
        scales=scales,
        scalogram=scalogram,
    )


def compute_dwt_decomposition(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int = 5,
) -> DWTResult:
    """Perform Multiresolution Discrete Wavelet Transform (DWT)."""
    arr = np.asarray(x, dtype=np.float64)
    max_level = pywt.dwt_max_level(len(arr), pywt.Wavelet(wavelet).dec_len)
    actual_level = min(level, max_level)

    coeffs = pywt.wavedec(arr, wavelet=wavelet, level=actual_level)
    cA = coeffs[0]
    cDs = coeffs[1:]

    # Compute energy per band
    energies = [float(np.sum(cA**2))] + [float(np.sum(cD**2)) for cD in cDs]
    total_energy = sum(energies)
    rel_energies = [e / total_energy if total_energy > 0 else 0.0 for e in energies]

    return DWTResult(
        approx_coefficients=cA,
        detail_coefficients=cDs,
        level_energies=energies,
        relative_energies=rel_energies,
    )


def compute_scalogram_matrix(
    x: np.ndarray,
    sampling_rate: float,
    output_shape: tuple[int, int] = (64, 64),
    wavelet: str = "morl",
) -> np.ndarray:
    """Generate normalized 2D scalogram image for CNN or visualization."""
    cwt_res = compute_cwt(x, sampling_rate, wavelet=wavelet, total_scales=output_shape[0])
    scalogram = cwt_res.scalogram

    # Resize to output_shape
    from scipy.ndimage import zoom

    zoom_factors = (output_shape[0] / scalogram.shape[0], output_shape[1] / scalogram.shape[1])
    resized = zoom(scalogram, zoom_factors, order=1)
    return np.asarray(np.clip(resized, 0.0, 1.0), dtype=np.float32)
