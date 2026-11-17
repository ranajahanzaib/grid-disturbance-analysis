"""Unit tests for configuration loading and validation."""

from grid_disturbance.config import AppConfig, load_config


def test_default_config_instantiation():
    cfg = AppConfig()
    assert cfg.signal.nominal_frequency == 60.0
    assert cfg.signal.default_sampling_rate == 30.0
    assert cfg.preprocessing.filter.low_cutoff == 0.05
    assert cfg.preprocessing.filter.high_cutoff == 14.0
    assert cfg.analysis.damping.well_damped_threshold == 0.05


def test_load_config_from_file():
    cfg = load_config("configs/default.yaml")
    assert cfg.system.project_name == "grid-disturbance-analysis"
    assert cfg.analysis.source_assessment.coherence_threshold == 0.60
