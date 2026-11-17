"""Tests for synthetic signal generator across all 12 disturbance types."""

import numpy as np

from grid_disturbance.synthetic.generator import SyntheticSignalGenerator


def test_all_12_synthetic_types_generated(synthetic_generator):
    gen = synthetic_generator
    s1 = gen.generate_baseline()
    s2 = gen.generate_forced_oscillation()
    s3 = gen.generate_damped_ringdown()
    s4 = gen.generate_poorly_damped()
    s5 = gen.generate_growing_oscillation()
    s6 = gen.generate_frequency_excursion()
    s7 = gen.generate_frequency_drift()
    s8 = gen.generate_generator_electromechanical("inter_area")
    s9 = gen.generate_generator_electromechanical("local_plant")
    s10 = gen.generate_overlapping_events()
    s11 = gen.generate_amplitude_modulated()
    s12 = gen.generate_frequency_modulated()

    signals = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12]
    for s in signals:
        assert len(s.data) > 100
        assert np.all(np.isfinite(s.data))
        assert s.ground_truth.disturbance_type is not None


def test_generator_determinism():
    gen1 = SyntheticSignalGenerator(random_seed=12345)
    sig1 = gen1.generate_forced_oscillation()

    gen2 = SyntheticSignalGenerator(random_seed=12345)
    sig2 = gen2.generate_forced_oscillation()

    assert np.allclose(sig1.data, sig2.data)
