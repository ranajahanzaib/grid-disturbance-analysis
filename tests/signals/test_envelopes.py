"""Tests for envelope analysis, exponential fitting, and damping estimation."""

import numpy as np

from grid_disturbance.signals.envelopes import (
    extract_hilbert_envelope,
    fit_exponential_envelope,
)


def test_hilbert_envelope_recovery():
    sr = 30.0
    t = np.arange(300) / sr
    modulating = 2.0 + 0.5 * np.cos(2 * np.pi * 0.1 * t)
    carrier = np.cos(2 * np.pi * 1.5 * t)
    sig = modulating * carrier

    env = extract_hilbert_envelope(sig)
    # Average envelope should match average modulating amplitude
    assert abs(np.mean(env) - 2.0) < 0.20


def test_exponential_damping_fit_accuracy():
    sr = 30.0
    t = np.arange(600) / sr
    f0 = 0.60
    zeta_true = 0.08  # 8% damping
    omega_d = 2 * np.pi * f0
    alpha_true = omega_d * (zeta_true / np.sqrt(1.0 - zeta_true**2))

    a0 = 2.0
    envelope = a0 * np.exp(-alpha_true * t)

    res = fit_exponential_envelope(envelope, sampling_rate=sr, oscillation_frequency=f0)
    assert res.is_damped
    assert not res.is_growing
    assert abs(res.damping_ratio - zeta_true) < 0.02
    assert res.r_squared > 0.95
