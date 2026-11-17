"""Configurable signal preprocessing pipeline with complete audit trails."""

from __future__ import annotations

import numpy as np

from grid_disturbance.config import PreprocessingConfig
from grid_disturbance.signals.filtering import (
    PreprocessingRecord,
    apply_filter,
    design_filter,
    detect_and_clip_outliers,
    detect_and_handle_missing,
    detrend_signal,
    normalize_signal,
)


class SignalPreprocessor:
    """Preprocesses 1D or 2D grid signals with audit tracking."""

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()

    def process(
        self,
        x: np.ndarray,
        sampling_rate: float = 30.0,
    ) -> tuple[np.ndarray, PreprocessingRecord]:
        """Apply configured preprocessing pipeline.

        Returns:
            (processed_array, record_audit)
        """
        original = np.asarray(x, dtype=np.float64)
        arr = original.copy()

        record = PreprocessingRecord(
            original_length=len(arr) if arr.ndim == 1 else arr.shape[-1],
            sampling_rate=sampling_rate,
        )

        # 1. Missing data handling
        arr, nan_count = detect_and_handle_missing(arr, method="linear")
        record.nan_count = nan_count
        if nan_count > 0:
            record.steps_applied.append("handle_missing")
            record.parameters["handle_missing"] = {"method": "linear", "nan_count": nan_count}

        # 2. Outlier detection and clipping
        if self.config.outlier_detection.method != "none":
            arr, clipped = detect_and_clip_outliers(
                arr,
                method=self.config.outlier_detection.method,  # type: ignore
                threshold=self.config.outlier_detection.threshold,
            )
            record.outliers_clipped = clipped
            if clipped > 0:
                record.steps_applied.append("clip_outliers")
                record.parameters["clip_outliers"] = {
                    "method": self.config.outlier_detection.method,
                    "threshold": self.config.outlier_detection.threshold,
                    "clipped_count": clipped,
                }

        # 3. Detrending
        if self.config.detrend != "none":
            arr = detrend_signal(arr, method=self.config.detrend)  # type: ignore
            record.steps_applied.append(f"detrend_{self.config.detrend}")

        # 4. DC offset removal
        if self.config.remove_dc:
            arr = arr - np.mean(arr)
            record.steps_applied.append("remove_dc")

        # 5. Filtering
        if self.config.filter.enabled and self.config.filter.type != "none":
            nyq = 0.5 * sampling_rate
            high_cutoff = min(self.config.filter.high_cutoff, nyq * 0.95)
            low_cutoff = self.config.filter.low_cutoff

            if self.config.filter.type == "bandpass" and low_cutoff < high_cutoff:
                b, a = design_filter(
                    filter_type="bandpass",
                    sampling_rate=sampling_rate,
                    cutoff=(low_cutoff, high_cutoff),
                    order=self.config.filter.order,
                )
                arr = apply_filter(arr, b, a, zero_phase=True)
                record.steps_applied.append("bandpass_filter")
                record.parameters["bandpass_filter"] = {
                    "low_cutoff": low_cutoff,
                    "high_cutoff": high_cutoff,
                    "order": self.config.filter.order,
                }

        # 6. Normalization
        if self.config.normalization != "none":
            arr = normalize_signal(arr, method=self.config.normalization)  # type: ignore
            record.steps_applied.append(f"normalize_{self.config.normalization}")

        record.final_length = len(arr) if arr.ndim == 1 else arr.shape[-1]
        return arr, record
