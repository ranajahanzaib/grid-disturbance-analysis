"""Feature engineering module for time-series signal classification."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from grid_disturbance.signals.envelopes import extract_hilbert_envelope, fit_exponential_envelope
from grid_disturbance.signals.oscillations import (
    estimate_amplitude_modulation,
    estimate_frequency_drift,
)
from grid_disturbance.signals.spectral import (
    analyze_spectrum,
    compute_autocorrelation,
)
from grid_disturbance.signals.wavelets import compute_dwt_decomposition


def extract_time_domain_features(x: np.ndarray, sampling_rate: float) -> dict[str, float]:
    """Extract statistical and morphological time-domain features."""
    arr = np.asarray(x, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {}

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    var_val = float(np.var(arr))
    rms_val = float(np.sqrt(np.mean(arr**2)))
    p2p_val = float(np.ptp(arr))
    crest_factor = float(np.max(np.abs(arr)) / rms_val) if rms_val > 1e-12 else 0.0

    # Skewness and kurtosis
    skew_val = float(stats.skew(arr)) if std_val > 1e-12 else 0.0
    kurt_val = float(stats.kurtosis(arr)) if std_val > 1e-12 else 0.0

    # Zero-crossing rate
    zero_crossings = np.sum(np.diff(np.signbit(arr)))
    zcr = float(zero_crossings / (n - 1)) if n > 1 else 0.0

    # Signal energy
    energy = float(np.sum(arr**2))

    # Autocorrelation lag-1
    acorr = compute_autocorrelation(arr, max_lag=min(30, n // 2))
    acorr_lag1 = float(acorr[1]) if len(acorr) > 1 else 1.0

    # Autocorrelation peak beyond lag-5 (periodicity detector)
    if len(acorr) > 10:
        acorr_peak = float(np.max(acorr[5:]))
    else:
        acorr_peak = 0.0

    return {
        "td_mean": mean_val,
        "td_std": std_val,
        "td_var": var_val,
        "td_rms": rms_val,
        "td_peak_to_peak": p2p_val,
        "td_crest_factor": crest_factor,
        "td_skewness": skew_val,
        "td_kurtosis": kurt_val,
        "td_zero_crossing_rate": zcr,
        "td_energy": energy,
        "td_autocorr_lag1": acorr_lag1,
        "td_autocorr_peak": acorr_peak,
    }


def extract_frequency_domain_features(
    x: np.ndarray,
    sampling_rate: float,
) -> dict[str, float]:
    """Extract frequency and spectral distribution features."""
    spec_res = analyze_spectrum(x, sampling_rate)

    dom_peak = spec_res.peaks[0] if len(spec_res.peaks) > 0 else None
    dom_freq = dom_peak.frequency if dom_peak else 0.0
    dom_prominence = dom_peak.prominence if dom_peak else 0.0
    dom_width = dom_peak.width if dom_peak else 0.0

    # Sub-synchronous and electromechanical band energy ratios
    freqs = spec_res.frequencies
    psd = spec_res.psd
    total_p = np.sum(psd) if len(psd) > 0 else 1.0

    inter_area_mask = (freqs >= 0.1) & (freqs < 1.0)
    plant_mask = (freqs >= 1.0) & (freqs < 2.0)
    subsynch_mask = (freqs >= 2.0) & (freqs < 10.0)

    inter_area_energy = float(np.sum(psd[inter_area_mask]) / total_p) if total_p > 0 else 0.0
    plant_energy = float(np.sum(psd[plant_mask]) / total_p) if total_p > 0 else 0.0
    subsynch_energy = float(np.sum(psd[subsynch_mask]) / total_p) if total_p > 0 else 0.0

    # Check for harmonics
    has_harmonics = 1.0 if any(p.is_harmonic_of is not None for p in spec_res.peaks) else 0.0

    return {
        "fd_dominant_freq": dom_freq,
        "fd_dominant_power": spec_res.dominant_power,
        "fd_peak_prominence": dom_prominence,
        "fd_peak_width": dom_width,
        "fd_peak_count": float(len(spec_res.peaks)),
        "fd_spectral_centroid": spec_res.spectral_centroid,
        "fd_spectral_bandwidth": spec_res.spectral_bandwidth,
        "fd_spectral_entropy": spec_res.spectral_entropy,
        "fd_spectral_flatness": spec_res.spectral_flatness,
        "fd_spectral_concentration": spec_res.spectral_concentration,
        "fd_inter_area_ratio": inter_area_energy,
        "fd_plant_ratio": plant_energy,
        "fd_subsynch_ratio": subsynch_energy,
        "fd_has_harmonics": has_harmonics,
    }


def extract_time_frequency_features(
    x: np.ndarray,
    sampling_rate: float,
) -> dict[str, float]:
    """Extract wavelet and multi-resolution decomposition features."""
    dwt_res = compute_dwt_decomposition(x, wavelet="db4", level=4)
    rel_energies = dwt_res.relative_energies

    # Ensure 5 energy slots
    energies_padded = rel_energies + [0.0] * max(0, 5 - len(rel_energies))

    return {
        "tf_wavelet_energy_approx": energies_padded[0],
        "tf_wavelet_energy_d1": energies_padded[1],
        "tf_wavelet_energy_d2": energies_padded[2],
        "tf_wavelet_energy_d3": energies_padded[3],
        "tf_wavelet_energy_d4": energies_padded[4],
    }


def extract_oscillation_features(
    x: np.ndarray,
    sampling_rate: float,
    dominant_frequency: float,
) -> dict[str, float]:
    """Extract damping, decay/growth, envelope stability, and modulation features."""
    arr = np.asarray(x, dtype=np.float64)
    env = extract_hilbert_envelope(arr)

    # Envelope variance / coefficient of variation
    env_mean = float(np.mean(env))
    env_std = float(np.std(env))
    env_cv = float(env_std / env_mean) if env_mean > 1e-12 else 0.0

    # Exponential fit
    fit_res = fit_exponential_envelope(env, sampling_rate, dominant_frequency)

    # Modulation and drift
    am_index, _ = estimate_amplitude_modulation(env, sampling_rate)
    drift = estimate_frequency_drift(arr, sampling_rate)

    return {
        "osc_damping_ratio": fit_res.damping_ratio,
        "osc_damping_coeff": fit_res.damping_coefficient,
        "osc_decay_rate": fit_res.decay_rate,
        "osc_growth_rate": fit_res.growth_rate,
        "osc_envelope_cv": env_cv,
        "osc_fit_r2": fit_res.r_squared,
        "osc_fit_rmse": fit_res.rmse,
        "osc_fit_confidence": fit_res.confidence,
        "osc_am_index": am_index,
        "osc_frequency_drift": drift,
        "osc_is_sustained": 1.0 if fit_res.is_sustained else 0.0,
        "osc_is_damped": 1.0 if fit_res.is_damped else 0.0,
        "osc_is_growing": 1.0 if fit_res.is_growing else 0.0,
    }


def extract_signal_features(
    x: np.ndarray,
    sampling_rate: float = 30.0,
) -> dict[str, float]:
    """Extract complete comprehensive feature set for a 1D signal.

    Returns a flat dictionary of named scalar features.
    """
    arr = np.asarray(x, dtype=np.float64)
    td = extract_time_domain_features(arr, sampling_rate)
    fd = extract_frequency_domain_features(arr, sampling_rate)
    tf = extract_time_frequency_features(arr, sampling_rate)
    dom_f = fd.get("fd_dominant_freq", 0.0)
    osc = extract_oscillation_features(arr, sampling_rate, dom_f)

    # Merge into flat dictionary
    features: dict[str, float] = {}
    features.update(td)
    features.update(fd)
    features.update(tf)
    features.update(osc)

    return features


def get_feature_names() -> list[str]:
    """Get the fixed sorted list of all feature names."""
    # Generate on dummy array to guarantee consistency
    dummy = np.sin(np.linspace(0, 10, 300))
    feats = extract_signal_features(dummy, 30.0)
    return sorted(feats.keys())
