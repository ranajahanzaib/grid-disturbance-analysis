"""Signal and dataset validation using Pydantic and Pandera."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class SignalValidationReport(BaseModel):
    """Validation report for an ingested time series signal."""

    is_valid: bool = True
    num_samples: int = 0
    num_channels: int = 1
    sampling_rate: float = 30.0
    duration_sec: float = 0.0
    has_nans: bool = False
    nan_count: int = 0
    has_infs: bool = False
    is_constant: bool = False
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def validate_signal_array(
    data: np.ndarray,
    sampling_rate: float = 30.0,
    min_samples: int = 30,
) -> SignalValidationReport:
    """Validate a 1D or 2D signal array against technical and mathematical criteria."""
    report = SignalValidationReport(sampling_rate=sampling_rate)

    if not isinstance(data, np.ndarray):
        try:
            data = np.asarray(data, dtype=np.float64)
        except Exception as e:
            report.is_valid = False
            report.errors.append(f"Failed to convert input to NumPy array: {e}")
            return report

    if data.ndim == 1:
        report.num_channels = 1
        report.num_samples = len(data)
    elif data.ndim == 2:
        # Expecting shape (num_samples, num_channels) or (num_channels, num_samples)
        if data.shape[0] < data.shape[1]:
            report.num_channels = data.shape[0]
            report.num_samples = data.shape[1]
        else:
            report.num_channels = data.shape[1]
            report.num_samples = data.shape[0]
    else:
        report.is_valid = False
        report.errors.append(f"Expected 1D or 2D array, got ndim={data.ndim}")
        return report

    if report.num_samples < min_samples:
        report.is_valid = False
        report.errors.append(f"Signal length {report.num_samples} is below minimum {min_samples}")
        return report

    report.duration_sec = report.num_samples / sampling_rate

    nan_count = int(np.sum(np.isnan(data)))
    if nan_count > 0:
        report.has_nans = True
        report.nan_count = nan_count
        report.warnings.append(f"Signal contains {nan_count} NaN values.")

    inf_count = int(np.sum(np.isinf(data)))
    if inf_count > 0:
        report.has_infs = True
        report.errors.append(f"Signal contains {inf_count} Inf values.")
        report.is_valid = False

    # Check for flatline / constant signal
    std = float(np.nanstd(data))
    if std < 1e-12:
        report.is_constant = True
        report.warnings.append("Signal is constant (zero variance).")

    return report


def validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
    time_column: str | None = None,
) -> tuple[bool, list[str]]:
    """Validate structure and monotonicity of time series dataframe."""
    errors: list[str] = []

    if df.empty:
        return False, ["DataFrame is empty."]

    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")

    if time_column and time_column in df.columns:
        time_series = df[time_column]
        if not time_series.is_monotonic_increasing:
            errors.append(f"Time column '{time_column}' is not strictly monotonically increasing.")

    return len(errors) == 0, errors
