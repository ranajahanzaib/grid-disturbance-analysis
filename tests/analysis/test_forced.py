"""Tests for forced oscillation detection."""

from grid_disturbance.analysis.forced import analyze_forced_oscillation


def test_forced_oscillation_detection(clean_forced_signal):
    ev = analyze_forced_oscillation(clean_forced_signal.data, sampling_rate=30.0)
    assert ev.is_forced
    assert ev.likelihood > 0.60
    assert abs(ev.dominant_frequency - 0.85) < 0.10
    assert ev.persistence_ratio >= 0.70
    assert len(ev.supporting_points) > 0


def test_forced_oscillation_rejects_ringdown(clean_ringdown_signal):
    ev = analyze_forced_oscillation(clean_ringdown_signal.data, sampling_rate=30.0)
    # Natural ring-down should not be accepted as a forced oscillation
    assert not ev.is_forced
    assert ev.likelihood < 0.50
