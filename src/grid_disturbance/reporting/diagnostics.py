"""Builder converting diagnostic analysis results into structured reports."""

from __future__ import annotations

import uuid

import numpy as np

from grid_disturbance.analysis.diagnosis import DiagnosticResult
from grid_disturbance.reporting.schemas import (
    ClassificationSection,
    ModelEvidenceSection,
    OscillationCharacteristicsSection,
    SignalEvidenceSection,
    SourceAssessmentSection,
    StructuredDiagnosticReport,
)


def create_structured_report(diag: DiagnosticResult) -> StructuredDiagnosticReport:
    """Build a complete StructuredDiagnosticReport from a DiagnosticResult."""
    # Build classification section
    cls_sec = ClassificationSection(
        predicted_class=diag.disturbance_type,
        confidence=diag.confidence,
        alternative_classifications=diag.model_predictions,
        rationale=diag.discrimination.summary_verdict,
    )

    # Build oscillation section
    osc_sec = OscillationCharacteristicsSection(
        dominant_frequency_hz=diag.oscillations.dominant_frequency,
        estimated_amplitude=float(
            np.sqrt(diag.spectrum.dominant_power) if diag.spectrum.dominant_power > 0 else 1.0
        ),
        damping_ratio=diag.oscillations.primary_damping_ratio,
        damping_classification=diag.discrimination.ringdown_evidence.damping_classification,
        decay_rate=diag.discrimination.ringdown_evidence.decay_rate,
        growth_rate=diag.derived_features.get("growth_rate", 0.0),
        duration_sec=float(diag.direct_measurements.get("duration_sec", 20.0)),
        frequency_drift_hz=diag.oscillations.frequency_drift_hz,
        amplitude_modulation_index=diag.oscillations.amplitude_modulation_index,
        is_sustained=diag.oscillations.is_sustained,
    )

    # Build signal evidence
    peaks_info = [
        {"frequency": p.frequency, "power": p.power, "prominence": p.prominence}
        for p in diag.spectrum.peaks[:5]
    ]
    tf_obs = []
    if diag.overlapping_events.has_multiple_events:
        tf_obs.append(diag.overlapping_events.summary)
    if diag.frequency_stability.is_frequency_instability:
        tf_obs.extend(diag.frequency_stability.summary_points)

    sig_sec = SignalEvidenceSection(
        dominant_peaks=peaks_info,
        spectral_concentration=diag.spectrum.spectral_concentration,
        spectral_entropy=diag.spectrum.spectral_entropy,
        frequency_stability_hz=diag.oscillations.frequency_drift_hz,
        envelope_coefficient_of_variation=diag.derived_features.get("envelope_cv", 0.0),
        time_frequency_observations=tf_obs,
    )

    # Build model evidence
    mod_sec = ModelEvidenceSection(
        model_name="signal_physics_hybrid",
        prediction_confidence=diag.confidence,
        top_contributing_features=[
            {"feature": k, "value": v} for k, v in list(diag.derived_features.items())[:6]
        ],
        prediction_uncertainty=float(max(0.0, 1.0 - diag.confidence)),
    )

    # Build source assessment
    src = diag.source_assessment
    src_sec = SourceAssessmentSection(
        is_supported=src.is_supported,
        candidate_origin_channel=src.candidate_source_channel,
        candidate_origin_region=src.candidate_source_region,
        confidence=src.confidence,
        evidence_points=src.supporting_evidence,
        missing_information=src.missing_information,
        limitations=src.limitations,
    )

    return StructuredDiagnosticReport(
        report_id=str(uuid.uuid4())[:8],
        event_id=diag.event_id,
        event_classification=cls_sec,
        oscillation_characteristics=osc_sec,
        signal_evidence=sig_sec,
        model_evidence=mod_sec,
        source_assessment=src_sec,
        direct_measurements=diag.direct_measurements,
        derived_features=diag.derived_features,
        plain_language_summary=diag.summary,
        technical_observations=diag.engineering_interpretations,
        engineering_limitations=src.limitations + diag.discrimination.limitations,
    )
