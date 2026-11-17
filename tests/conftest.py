"""Pytest test configuration and shared fixtures."""

import numpy as np
import pytest

from grid_disturbance.synthetic.generator import SyntheticSignalGenerator
from grid_disturbance.synthetic.scenarios import generate_multichannel_forced_oscillation_scenario


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_generator():
    return SyntheticSignalGenerator(random_seed=42)


@pytest.fixture
def clean_forced_signal(synthetic_generator):
    return synthetic_generator.generate_forced_oscillation(
        frequency=0.85, amplitude=1.0, sampling_rate=30.0, duration_sec=20.0, snr_db=30.0
    )


@pytest.fixture
def clean_ringdown_signal(synthetic_generator):
    return synthetic_generator.generate_damped_ringdown(
        frequency=0.60,
        damping_ratio=0.08,
        amplitude=1.5,
        sampling_rate=30.0,
        duration_sec=20.0,
        snr_db=30.0,
    )


@pytest.fixture
def multichannel_scenario():
    return generate_multichannel_forced_oscillation_scenario(
        frequency=0.80, sampling_rate=30.0, duration_sec=20.0
    )
