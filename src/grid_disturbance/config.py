"""Configuration system with Pydantic validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class OscillationBandsConfig(BaseModel):
    """Oscillation frequency bands (Hz)."""

    inter_area: tuple[float, float] = (0.1, 1.0)
    local_plant: tuple[float, float] = (1.0, 2.0)
    control_torsional: tuple[float, float] = (2.0, 10.0)


class SignalConfig(BaseModel):
    """Signal configuration parameters."""

    nominal_frequency: float = 60.0
    default_sampling_rate: float = 30.0
    nyquist_fraction: float = 0.95
    oscillation_bands: OscillationBandsConfig = Field(default_factory=OscillationBandsConfig)


class FilterConfig(BaseModel):
    """Filter configuration."""

    enabled: bool = True
    type: str = "bandpass"
    low_cutoff: float = 0.05
    high_cutoff: float = 14.0
    order: int = 4


class OutlierConfig(BaseModel):
    """Outlier detection configuration."""

    method: str = "mad"
    threshold: float = 5.0


class PreprocessingConfig(BaseModel):
    """Signal preprocessing configuration."""

    detrend: str = "linear"
    remove_dc: bool = True
    normalization: str = "standardize"
    filter: FilterConfig = Field(default_factory=FilterConfig)
    outlier_detection: OutlierConfig = Field(default_factory=OutlierConfig)


class FFTConfig(BaseModel):
    zero_padding_factor: int = 2


class WelchConfig(BaseModel):
    nperseg: int = 256
    noverlap: int = 128
    window: str = "hann"


class STFTConfig(BaseModel):
    nperseg: int = 128
    noverlap: int = 96
    window: str = "hann"


class CWTConfig(BaseModel):
    wavelet: str = "morl"
    total_scales: int = 64


class DampingConfig(BaseModel):
    min_confidence: float = 0.60
    well_damped_threshold: float = 0.05
    undamped_threshold: float = 0.00


class ForcedOscillationConfig(BaseModel):
    persistence_ratio_threshold: float = 0.65
    spectral_purity_threshold: float = 0.40
    frequency_stability_tolerance: float = 0.05


class SourceAssessmentConfig(BaseModel):
    coherence_threshold: float = 0.60
    min_channels_required: int = 2


class AnalysisConfig(BaseModel):
    """Analysis algorithms configuration."""

    fft: FFTConfig = Field(default_factory=FFTConfig)
    welch: WelchConfig = Field(default_factory=WelchConfig)
    stft: STFTConfig = Field(default_factory=STFTConfig)
    cwt: CWTConfig = Field(default_factory=CWTConfig)
    damping: DampingConfig = Field(default_factory=DampingConfig)
    forced_oscillation: ForcedOscillationConfig = Field(default_factory=ForcedOscillationConfig)
    source_assessment: SourceAssessmentConfig = Field(default_factory=SourceAssessmentConfig)


class SystemConfig(BaseModel):
    """System level configuration."""

    project_name: str = "grid-disturbance-analysis"
    version: str = "0.1.0"
    random_seed: int = 42
    log_level: str = "INFO"
    device: str = "auto"
    cache_dir: str = ".cache"
    database_path: str = "data/storage.db"


class AppConfig(BaseModel):
    """Root application configuration."""

    system: SystemConfig = Field(default_factory=SystemConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load application configuration from YAML file or return defaults."""
    if config_path is None:
        default_yaml = Path("configs/default.yaml")
        if default_yaml.is_file():
            config_path = default_yaml
        else:
            return AppConfig()

    path = Path(config_path)
    if not path.is_file():
        return AppConfig()

    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    return AppConfig.model_validate(data)
