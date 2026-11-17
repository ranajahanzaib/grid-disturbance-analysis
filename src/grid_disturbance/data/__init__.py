"""Data ingestion, validation, preprocessing, and dataset management."""

from grid_disturbance.data.gesl import GESLDataQualityReport, GESLDatasetManager
from grid_disturbance.data.loaders import load_signal_from_file
from grid_disturbance.data.manifest import DatasetManifest, PartitionStats, compute_file_sha256
from grid_disturbance.data.preprocessing import SignalPreprocessor
from grid_disturbance.data.splitting import split_dataset
from grid_disturbance.data.validation import (
    SignalValidationReport,
    validate_dataframe_schema,
    validate_signal_array,
)

__all__ = [
    "DatasetManifest",
    "GESLDataQualityReport",
    "GESLDatasetManager",
    "PartitionStats",
    "SignalPreprocessor",
    "SignalValidationReport",
    "compute_file_sha256",
    "load_signal_from_file",
    "split_dataset",
    "validate_dataframe_schema",
    "validate_signal_array",
]
