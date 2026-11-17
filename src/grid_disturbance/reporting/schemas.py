"""Pydantic schemas for structured diagnostic reporting and exports."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class ClassificationSection(BaseModel):
    predicted_class: str
    confidence: float
    alternative_classifications: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""


class OscillationCharacteristicsSection(BaseModel):
    dominant_frequency_hz: float
    estimated_amplitude: float
    damping_ratio: float
    damping_classification: str
    decay_rate: float
    growth_rate: float
    duration_sec: float
    frequency_drift_hz: float
    amplitude_modulation_index: float
    is_sustained: bool


class SignalEvidenceSection(BaseModel):
    dominant_peaks: list[dict[str, float]] = Field(default_factory=list)
    spectral_concentration: float
    spectral_entropy: float
    frequency_stability_hz: float
    envelope_coefficient_of_variation: float
    time_frequency_observations: list[str] = Field(default_factory=list)


class ModelEvidenceSection(BaseModel):
    model_name: str = "ensemble"
    prediction_confidence: float
    top_contributing_features: list[dict[str, Any]] = Field(default_factory=list)
    prediction_uncertainty: float = 0.0


class SourceAssessmentSection(BaseModel):
    is_supported: bool
    candidate_origin_channel: str | None = None
    candidate_origin_region: str | None = None
    confidence: float = 0.0
    evidence_points: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class StructuredDiagnosticReport(BaseModel):
    """Rigorous diagnostic report for power system disturbance analysis."""

    report_id: str
    event_id: str
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    event_classification: ClassificationSection
    oscillation_characteristics: OscillationCharacteristicsSection
    signal_evidence: SignalEvidenceSection
    model_evidence: ModelEvidenceSection
    source_assessment: SourceAssessmentSection
    direct_measurements: dict[str, Any] = Field(default_factory=dict)
    derived_features: dict[str, float] = Field(default_factory=dict)
    plain_language_summary: str
    technical_observations: list[str] = Field(default_factory=list)
    engineering_limitations: list[str] = Field(default_factory=list)
