"""Tests for modal estimation, frequency drift, and wavelets."""

import numpy as np

from grid_disturbance.signals.oscillations import (
    estimate_amplitude_modulation,
    estimate_frequency_drift,
)
from grid_disturbance.signals.wavelets import compute_cwt, compute_dwt_decomposition


def test_frequency_drift_estimation():
    sr = 30.0
    t = np.arange(600) / sr
    # Frequency ramps from 0.50 Hz to 1.10 Hz (drift ~ 0.60 Hz)
    phase = 2 * np.pi * (0.50 * t + 0.015 * t**2)
    sig = np.sin(phase)

    drift = estimate_frequency_drift(sig, sampling_rate=sr, window_sec=4.0, step_sec=1.0)
    assert drift > 0.30


def test_amplitude_modulation_estimation():
    sr = 30.0
    t = np.arange(600) / sr
    m_true = 0.50
    envelope = 1.0 + m_true * np.cos(2 * np.pi * 0.15 * t)

    am_idx, mod_f = estimate_amplitude_modulation(envelope, sampling_rate=sr)
    assert abs(am_idx - m_true) < 0.10
    assert abs(mod_f - 0.15) < 0.05


def test_cwt_and_dwt_wavelets():
    sr = 30.0
    t = np.arange(300) / sr
    sig = np.sin(2 * np.pi * 1.0 * t)

    cwt_res = compute_cwt(sig, sampling_rate=sr, total_scales=32)
    assert cwt_res.scalogram.shape[0] == 32
    assert cwt_res.scalogram.shape[1] == 300

    dwt_res = compute_dwt_decomposition(sig, level=3)
    assert len(dwt_res.detail_coefficients) == 3
    assert np.isclose(sum(dwt_res.relative_energies), 1.0)
