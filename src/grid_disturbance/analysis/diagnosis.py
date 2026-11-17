"""Unified Disturbance Diagnostic Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from grid_disturbance.analysis.detector import (
    OscillationDiscriminationResult,
    discriminate_oscillation,
)
from grid_disturbance.analysis.frequency import (
    FrequencyInstabilityReport,
    analyze_frequency_instability,
)
from grid_disturbance.analysis.generator import (
    GeneratorOscillationHypothesis,
    evaluate_generator_hypothesis,
)
from grid_disturbance.analysis.overlapping import (
    OverlappingEventsResult,
    analyze_overlapping_events,
)
from grid_disturbance.analysis.source import SourceAssessmentResult, assess_oscillation_source
from grid_disturbance.data.preprocessing import SignalPreprocessor
from grid_disturbance.data.validation import SignalValidationReport, validate_signal_array
from grid_disturbance.signals.oscillations import (
    OscillationCharacterization,
    characterize_oscillations,
)
from grid_disturbance.signals.spectral import SpectralResult, analyze_spectrum


@dataclass
class DiagnosticResult:
    """Comprehensive diagnostic evaluation combining signal analytics and ML predictions."""

    event_id: str
    disturbance_type: str
    confidence: float
    summary: str
    validation: SignalValidationReport
    spectrum: SpectralResult
    oscillations: OscillationCharacterization
    discrimination: OscillationDiscriminationResult
    frequency_stability: FrequencyInstabilityReport
    generator_hypothesis: GeneratorOscillationHypothesis
    overlapping_events: OverlappingEventsResult
    source_assessment: SourceAssessmentResult
    direct_measurements: dict[str, Any] = field(default_factory=dict)
    derived_features: dict[str, float] = field(default_factory=dict)
    model_predictions: dict[str, float] = field(default_factory=dict)
    engineering_interpretations: list[str] = field(default_factory=list)


class DisturbanceDiagnosticEngine:
    """High-level analysis engine coordinating signal processing, detection, and source assessment."""

    def __init__(self, preprocessor: SignalPreprocessor | None = None):
        self.preprocessor = preprocessor or SignalPreprocessor()

    def diagnose_signal(
        self,
        signal_data: np.ndarray,
        sampling_rate: float = 30.0,
        channel_names: list[str] | None = None,
        event_id: str = "evt_001",
        ml_model: Any = None,  # Optional trained ML model
    ) -> DiagnosticResult:
        """Run complete diagnostic workflow on 1D or multi-channel signal."""
        raw_arr = np.asarray(signal_data, dtype=np.float64)

        if raw_arr.ndim == 1:
            primary_sig = raw_arr
            multichannel = False
            channels = channel_names or ["channel_1"]
        else:
            primary_sig = raw_arr[0, :]
            multichannel = True
            channels = channel_names or [f"channel_{i + 1}" for i in range(raw_arr.shape[0])]

        # 1. Validation
        val_report = validate_signal_array(primary_sig, sampling_rate=sampling_rate)

        # 2. Preprocessing
        clean_sig, _ = self.preprocessor.process(primary_sig, sampling_rate=sampling_rate)

        # 3. Spectral analysis
        spec = analyze_spectrum(clean_sig, sampling_rate=sampling_rate)
        f0 = spec.dominant_frequency

        # 4. Oscillation characterization
        osc = characterize_oscillations(
            clean_sig, sampling_rate=sampling_rate, dominant_frequency=f0
        )

        # 5. Discrimination: Forced vs Natural
        discrim = discriminate_oscillation(clean_sig, sampling_rate=sampling_rate)

        # 6. Frequency stability
        freq_report = analyze_frequency_instability(clean_sig, sampling_rate=sampling_rate)

        # 7. Generator hypothesis
        gen_hyp = evaluate_generator_hypothesis(clean_sig, sampling_rate=sampling_rate)

        # 8. Overlapping events
        overlap = analyze_overlapping_events(clean_sig, sampling_rate=sampling_rate)

        # 9. Source assessment
        if multichannel and raw_arr.shape[0] > 1:
            source_res = assess_oscillation_source(
                raw_arr, channels, sampling_rate=sampling_rate, disturbance_frequency=f0
            )
        else:
            source_res = assess_oscillation_source(
                clean_sig, channels[:1], sampling_rate=sampling_rate, disturbance_frequency=f0
            )

        # 10. ML model prediction if available
        ml_preds: dict[str, float] = {}
        if ml_model is not None:
            try:
                from grid_disturbance.signals.features import extract_signal_features

                feats = extract_signal_features(clean_sig, sampling_rate=sampling_rate)
                feat_names = sorted(feats.keys())
                x_vec = np.array([[feats[k] for k in feat_names]])
                if hasattr(ml_model, "predict_proba"):
                    probs = ml_model.predict_proba(x_vec)[0]
                    classes = getattr(
                        ml_model, "classes_", [f"class_{i}" for i in range(len(probs))]
                    )
                    ml_preds = {str(c): float(p) for c, p in zip(classes, probs, strict=False)}
            except Exception:
                pass

        # Determine primary event classification
        if (
            freq_report.is_frequency_instability
            and freq_report.max_frequency_deviation_hz > 0.25
            and not discrim.forced_evidence.is_forced
        ):
            dist_type = "frequency_excursion"
            conf = 0.85
        elif overlap.has_multiple_events and overlap.num_components >= 2:
            dist_type = "overlapping_events"
            conf = 0.80
        elif discrim.primary_classification == "forced_oscillation":
            dist_type = "forced_oscillation"
            conf = discrim.confidence
        elif discrim.primary_classification == "natural_ringdown":
            if discrim.ringdown_evidence.damping_classification == "poorly_damped":
                dist_type = "poorly_damped"
            else:
                dist_type = "damped_ringdown"
            conf = discrim.confidence
        elif osc.is_growing:
            dist_type = "growing_oscillation"
            conf = 0.85
        else:
            dist_type = "baseline_noise" if spec.dominant_power < 0.05 else "indeterminate"
            conf = 0.60

        interpretations: list[str] = [
            f"Overall behavior categorized as '{dist_type}' with confidence {conf * 100:.1f}%.",
            discrim.summary_verdict,
            f"Generator hypothesis: {gen_hyp.mode_band} ({gen_hyp.evidence_level}).",
        ]
        if source_res.is_supported and source_res.candidate_source_channel:
            interpretations.append(
                f"Origin analysis points to candidate source region '{source_res.candidate_source_region}'."
            )
        else:
            interpretations.append(
                "Source localization is unsupported: insufficient multichannel information."
            )

        summary_text = " ".join(interpretations)

        direct_meas = {
            "num_samples": len(clean_sig),
            "sampling_rate_hz": sampling_rate,
            "duration_sec": len(clean_sig) / sampling_rate,
            "num_channels": len(channels),
            "channel_list": ", ".join(channels),
        }

        derived_f = {
            "dominant_frequency_hz": f0,
            "damping_ratio": osc.primary_damping_ratio,
            "spectral_concentration": spec.spectral_concentration,
            "frequency_drift_hz": osc.frequency_drift_hz,
            "envelope_cv": discrim.forced_evidence.envelope_coefficient_of_variation,
            "max_rocof_hz_per_sec": freq_report.max_rocof_hz_per_sec,
        }

        return DiagnosticResult(
            event_id=event_id,
            disturbance_type=dist_type,
            confidence=conf,
            summary=summary_text,
            validation=val_report,
            spectrum=spec,
            oscillations=osc,
            discrimination=discrim,
            frequency_stability=freq_report,
            generator_hypothesis=gen_hyp,
            overlapping_events=overlap,
            source_assessment=source_res,
            direct_measurements=direct_meas,
            derived_features=derived_f,
            model_predictions=ml_preds,
            engineering_interpretations=interpretations,
        )
