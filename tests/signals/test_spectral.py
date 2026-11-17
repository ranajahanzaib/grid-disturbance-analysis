"""Tests for spectral analysis, FFT, PSD, and metrics."""

import numpy as np

from grid_disturbance.signals.spectral import (
    compute_autocorrelation,
    compute_fft,
    compute_spectral_entropy,
    compute_spectral_flatness,
    compute_welch_psd,
    extract_spectral_peaks,
)


def test_fft_frequency_recovery():
    sr = 30.0
    t = np.arange(600) / sr
    true_f = 1.25
    sig = 2.0 * np.cos(2 * np.pi * true_f * t)

    freqs, mags, _ = compute_fft(sig, sampling_rate=sr, zero_padding_factor=2)
    dom_idx = np.argmax(mags)
    est_f = freqs[dom_idx]
    assert abs(est_f - true_f) < 0.05
    assert abs(mags[dom_idx] - 2.0) < 0.20


def test_psd_and_peak_extraction():
    sr = 30.0
    t = np.arange(600) / sr
    sig = 1.5 * np.cos(2 * np.pi * 0.80 * t) + 0.8 * np.cos(2 * np.pi * 2.40 * t)
    freqs, psd = compute_welch_psd(sig, sampling_rate=sr, nperseg=256)
    peaks = extract_spectral_peaks(freqs, psd, top_k=2)

    assert len(peaks) >= 2
    peak_freqs = [p.frequency for p in peaks]
    assert any(abs(f - 0.80) < 0.10 for f in peak_freqs)
    assert any(abs(f - 2.40) < 0.15 for f in peak_freqs)


def test_spectral_entropy_and_flatness():
    # Pure tone vs White noise
    n = 600
    tone_psd = np.zeros(n)
    tone_psd[50] = 100.0  # single line

    white_psd = np.ones(n)  # flat spectrum

    tone_entropy = compute_spectral_entropy(tone_psd)
    white_entropy = compute_spectral_entropy(white_psd)
    assert tone_entropy < white_entropy

    tone_flatness = compute_spectral_flatness(tone_psd)
    white_flatness = compute_spectral_flatness(white_psd)
    assert tone_flatness < 0.05
    assert white_flatness > 0.90


def test_autocorrelation_periodicity():
    sr = 30.0
    t = np.arange(300) / sr
    f0 = 1.0  # 1 sec period = 30 samples lag
    sig = np.sin(2 * np.pi * f0 * t)
    acorr = compute_autocorrelation(sig, max_lag=60)

    # First peak after zero lag should be near lag 30
    assert np.argmax(acorr[10:45]) + 10 == 30
