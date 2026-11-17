"""Tests for ringdown analysis and forced vs. natural discrimination."""

from grid_disturbance.analysis.detector import discriminate_oscillation
from grid_disturbance.analysis.ringdown import analyze_ringdown


def test_ringdown_detection_well_damped(clean_ringdown_signal):
    ev = analyze_ringdown(clean_ringdown_signal.data, sampling_rate=30.0)
    assert ev.is_ringdown
    assert ev.damping_classification == "well_damped"
    assert ev.damping_ratio >= 0.05
    assert ev.envelope_r2 > 0.70


def test_ringdown_poorly_damped(synthetic_generator):
    sig = synthetic_generator.generate_poorly_damped(frequency=0.45, damping_ratio=0.015)
    ev = analyze_ringdown(sig.data, sampling_rate=30.0)
    assert ev.damping_classification == "poorly_damped"
    assert ev.damping_ratio < 0.05


def test_discriminate_forced_vs_natural(clean_forced_signal, clean_ringdown_signal):
    res_forced = discriminate_oscillation(clean_forced_signal.data, sampling_rate=30.0)
    assert res_forced.primary_classification == "forced_oscillation"
    assert res_forced.forced_likelihood > res_forced.natural_likelihood

    res_ring = discriminate_oscillation(clean_ringdown_signal.data, sampling_rate=30.0)
    assert res_ring.primary_classification == "natural_ringdown"
    assert res_ring.natural_likelihood > res_ring.forced_likelihood


def test_discriminate_returns_indeterminate_on_ambiguous(synthetic_generator):
    # Baseline ambient noise has neither clear forced persistence nor exponential decay
    noise_sig = synthetic_generator.generate_baseline(sampling_rate=30.0, duration_sec=10.0)
    res = discriminate_oscillation(noise_sig.data, sampling_rate=30.0)
    # Must handle without crash and return low confidence or indeterminate
    assert res.primary_classification in ["indeterminate", "natural_ringdown", "forced_oscillation"]
    assert res.confidence < 0.65
