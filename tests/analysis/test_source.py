"""Tests for oscillation source localization and boundary validation."""

import numpy as np

from grid_disturbance.analysis.source import assess_oscillation_source


def test_source_assessment_single_channel_refusal():
    single_sig = np.sin(np.linspace(0, 20, 600))
    res = assess_oscillation_source(single_sig, channel_names=["PMU_BUS_1"], sampling_rate=30.0)

    # Must refuse physical localization on single channel
    assert not res.is_supported
    assert res.candidate_source_channel is None
    assert res.capability_level == "signal_characterization_only"
    assert len(res.missing_information) > 0
    assert len(res.limitations) > 0


def test_source_assessment_multichannel(multichannel_scenario):
    res = assess_oscillation_source(
        multichannel_scenario.signals,
        multichannel_scenario.channel_names,
        sampling_rate=multichannel_scenario.sampling_rate,
        disturbance_frequency=multichannel_scenario.true_source_frequency,
    )

    assert res.is_supported
    assert res.capability_level == "source_direction_assessment"
    # Ground truth source is Bus 1
    assert res.candidate_source_channel == "PMU_BUS_1"
    assert res.confidence > 0.50
    assert len(res.channel_relationships) == 3
