"""Tests for signal filtering, detrending, and preprocessing."""

import numpy as np

from grid_disturbance.signals.filtering import (
    apply_filter,
    design_filter,
    detect_and_clip_outliers,
    detect_and_handle_missing,
    detrend_signal,
    normalize_signal,
    resample_with_antialiasing,
    segment_signal,
)


def test_missing_data_interpolation():
    arr = np.array([1.0, 2.0, np.nan, 4.0, np.inf, 6.0])
    cleaned, count = detect_and_handle_missing(arr, method="linear")
    assert count == 2
    assert np.all(np.isfinite(cleaned))
    assert np.isclose(cleaned[2], 3.0)
    assert np.isclose(cleaned[4], 5.0)


def test_detrend_linear():
    t = np.linspace(0, 10, 200)
    trend = 2.5 * t + 1.0
    signal_wave = np.sin(2 * np.pi * 1.0 * t) + trend
    detrended = detrend_signal(signal_wave, method="linear")
    # Linear slope should be eliminated
    assert abs(np.polyfit(t, detrended, 1)[0]) < 1e-4


def test_normalize_standardize():
    arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    normed = normalize_signal(arr, method="standardize")
    assert np.isclose(np.mean(normed), 0.0)
    assert np.isclose(np.std(normed), 1.0)


def test_outlier_clipping_mad():
    arr = np.ones(100) * 2.0
    arr[10] = 50.0  # extreme outlier
    clipped, count = detect_and_clip_outliers(arr, method="mad", threshold=4.0)
    assert count >= 1
    assert clipped[10] < 50.0


def test_bandpass_filtering():
    sr = 30.0
    t = np.arange(300) / sr
    # Mix low freq (0.01 Hz drift), target freq (1.0 Hz), high freq (14.5 Hz noise)
    sig = 5.0 + np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 14.5 * t)
    b, a = design_filter("bandpass", sampling_rate=sr, cutoff=(0.2, 5.0), order=4)
    filtered = apply_filter(sig, b, a, zero_phase=True)

    # 1.0 Hz component should be preserved; DC offset should be rejected
    assert abs(np.mean(filtered)) < 0.10
    assert np.max(filtered) > 0.80


def test_resampling_antialiasing():
    orig_sr = 60.0
    target_sr = 30.0
    t = np.arange(600) / orig_sr
    sig = np.sin(2 * np.pi * 1.0 * t)
    resampled = resample_with_antialiasing(sig, orig_sr, target_sr)
    assert len(resampled) == 300


def test_segment_signal():
    arr = np.arange(100)
    windows = segment_signal(arr, window_samples=20, overlap_samples=10)
    assert len(windows) == 9
    assert len(windows[0]) == 20
