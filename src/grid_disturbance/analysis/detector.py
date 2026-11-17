"""Forced versus natural oscillation discrimination engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from grid_disturbance.analysis.forced import ForcedOscillationEvidence, analyze_forced_oscillation
from grid_disturbance.analysis.ringdown import RingdownEvidence, analyze_ringdown


@dataclass
class OscillationDiscriminationResult:
    """Rigorous comparison and discrimination between forced and natural oscillation mechanisms."""

    primary_classification: Literal["forced_oscillation", "natural_ringdown", "indeterminate"]
    forced_likelihood: float
    natural_likelihood: float
    confidence: float
    dominant_frequency: float
    forced_evidence: ForcedOscillationEvidence
    ringdown_evidence: RingdownEvidence
    summary_verdict: str
    key_findings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


def discriminate_oscillation(
    x: np.ndarray,
    sampling_rate: float = 30.0,
) -> OscillationDiscriminationResult:
    """Compare evidence for forced vs. natural oscillation hypotheses.

    Does not force an arbitrary binary choice if evidence is ambiguous.
    """
    arr = np.asarray(x, dtype=np.float64)

    forced_ev = analyze_forced_oscillation(arr, sampling_rate)
    ringdown_ev = analyze_ringdown(arr, sampling_rate)

    p_forced = forced_ev.likelihood
    p_natural = ringdown_ev.likelihood
    f0 = forced_ev.dominant_frequency

    findings: list[str] = []
    limitations: list[str] = []

    # Check for indeterminate or ambiguous case
    diff = abs(p_forced - p_natural)
    classification: Literal["forced_oscillation", "natural_ringdown", "indeterminate"]
    if max(p_forced, p_natural) < 0.45 or (diff < 0.15 and max(p_forced, p_natural) < 0.65):
        classification = "indeterminate"
        verdict = (
            f"Indeterminate oscillatory behavior at {f0:.2f} Hz. "
            f"Evidence is insufficiently distinct between sustained forced excitation (score {p_forced:.2f}) "
            f"and natural transient ring-down (score {p_natural:.2f})."
        )
        confidence = float(np.clip(1.0 - diff, 0.20, 0.50))
        limitations.append(
            "Signal does not exhibit clear exponential decay or high stationary persistence."
        )

    elif p_forced > p_natural:
        classification = "forced_oscillation"
        confidence = float(np.clip(p_forced * (1.0 - p_natural * 0.5), 0.50, 0.98))
        verdict = (
            f"Disturbance exhibits high likelihood ({p_forced * 100:.0f}%) of being a FORCED OSCILLATION at {f0:.2f} Hz. "
            f"Sustained persistence ({forced_ev.persistence_ratio * 100:.0f}%) and stable frequency "
            f"indicate persistent periodic driving input."
        )
        findings.extend(forced_ev.supporting_points)

    else:
        classification = "natural_ringdown"
        confidence = float(np.clip(p_natural * (1.0 - p_forced * 0.5), 0.50, 0.98))
        verdict = (
            f"Disturbance exhibits high likelihood ({p_natural * 100:.0f}%) of being a NATURAL RING-DOWN event at {f0:.2f} Hz "
            f"with estimated damping ratio ζ = {ringdown_ev.damping_ratio * 100:.1f}% ({ringdown_ev.damping_classification})."
        )
        findings.extend(ringdown_ev.supporting_points)

    return OscillationDiscriminationResult(
        primary_classification=classification,
        forced_likelihood=p_forced,
        natural_likelihood=p_natural,
        confidence=confidence,
        dominant_frequency=f0,
        forced_evidence=forced_ev,
        ringdown_evidence=ringdown_ev,
        summary_verdict=verdict,
        key_findings=findings,
        limitations=limitations,
    )
