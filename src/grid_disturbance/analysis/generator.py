"""Generator-related electromechanical oscillation analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from grid_disturbance.signals.spectral import analyze_spectrum


@dataclass
class GeneratorOscillationHypothesis:
    """Evaluates whether an observed oscillation resembles known generator electromechanical modes."""

    is_generator_related: bool
    mode_band: Literal["inter_area", "local_plant", "sub_synchronous", "outside_electromechanical"]
    dominant_frequency: float
    confidence: float
    evidence_level: str  # "hypothesis", "suggestive", "insufficient_data"
    rationales: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


def evaluate_generator_hypothesis(
    x: np.ndarray,
    sampling_rate: float = 30.0,
    has_generator_metadata: bool = False,
) -> GeneratorOscillationHypothesis:
    """Assess whether oscillation signature aligns with generator electromechanical dynamics.

    IMPORTANT: Treats generator attribution strictly as a hypothesis unless
    field-level rotor angle or machine terminal PMU measurements exist.
    """
    spec = analyze_spectrum(x, sampling_rate)
    f0 = spec.dominant_frequency

    rationales: list[str] = []
    limitations: list[str] = []

    if not has_generator_metadata:
        limitations.append(
            "Generator attribution is an analytical hypothesis. Direct machine rotor angle "
            "or generator-terminal PMU measurements are not confirmed present in metadata."
        )

    # Classify band
    mode_band: Literal["inter_area", "local_plant", "sub_synchronous", "outside_electromechanical"]
    if 0.1 <= f0 < 1.0:
        mode_band = "inter_area"
        is_gen = True
        confidence = 0.70
        rationales.append(
            f"Frequency {f0:.2f} Hz falls within the typical inter-area electromechanical band (0.1–1.0 Hz), "
            "characterizing coherent groups of generators swinging against each other across transmission corridors."
        )
    elif 1.0 <= f0 <= 2.0:
        mode_band = "local_plant"
        is_gen = True
        confidence = 0.65
        rationales.append(
            f"Frequency {f0:.2f} Hz falls within the local plant electromechanical band (1.0–2.0 Hz), "
            "characterizing an individual generator or power plant oscillating against the bulk grid."
        )
    elif 2.0 < f0 <= 10.0:
        mode_band = "sub_synchronous"
        is_gen = True
        confidence = 0.50
        rationales.append(
            f"Frequency {f0:.2f} Hz falls in the sub-synchronous / control interaction band (2.0–10.0 Hz)."
        )
    else:
        mode_band = "outside_electromechanical"
        is_gen = False
        confidence = 0.85
        rationales.append(
            f"Frequency {f0:.2f} Hz lies outside standard electromechanical oscillation bands."
        )

    evidence_level = (
        "suggestive" if (is_gen and spec.spectral_concentration > 0.40) else "hypothesis"
    )

    return GeneratorOscillationHypothesis(
        is_generator_related=is_gen,
        mode_band=mode_band,
        dominant_frequency=f0,
        confidence=confidence,
        evidence_level=evidence_level,
        rationales=rationales,
        limitations=limitations,
    )
