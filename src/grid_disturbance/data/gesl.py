"""Grid Event Signature Library (GESL) dataset adapter and ingestion manager."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from grid_disturbance.data.manifest import DatasetManifest, PartitionStats, compute_file_sha256
from grid_disturbance.data.splitting import split_dataset
from grid_disturbance.synthetic.presets import generate_benchmark_dataset

logger = logging.getLogger(__name__)


class GESLDataQualityReport(BaseModel):
    """Quality and summary statistics for a registered GESL or grid dataset."""

    dataset_name: str
    is_valid: bool = True
    total_records: int = 0
    total_events: int = 0
    num_channels: int = 0
    sampling_rate: float = 30.0
    class_distribution: dict[str, int] = Field(default_factory=dict)
    missing_value_percentage: float = 0.0
    duplicate_records_count: int = 0
    has_event_grouping: bool = False
    anomalous_records_count: int = 0
    quality_notes: list[str] = Field(default_factory=list)


class GESLDatasetManager:
    """Manages GESL data discovery, schema validation, registration, and preparation."""

    EXPECTED_COLUMNS = [
        "record_id",
        "disturbance_type",
        "sampling_rate",
    ]

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.manifest_dir = self.data_dir / "manifests"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    def inspect_dataset(self, source_path: str | Path | None = None) -> GESLDataQualityReport:
        """Inspect a raw or registered dataset and generate data quality report."""
        if source_path is None:
            # Check if there is a registered raw file or processed file
            csv_candidates = list(self.raw_dir.glob("*.csv")) + list(
                self.processed_dir.glob("*.csv")
            )
            if not csv_candidates:
                return GESLDataQualityReport(
                    dataset_name="No dataset found",
                    is_valid=False,
                    quality_notes=[
                        "No dataset found in data/raw or data/processed. Run 'grid-disturbance data prepare' to generate benchmark data or supply a GESL CSV."
                    ],
                )
            source_path = csv_candidates[0]

        path = Path(source_path)
        if not path.is_file():
            return GESLDataQualityReport(
                dataset_name=str(path),
                is_valid=False,
                quality_notes=[f"Specified file does not exist: {path}"],
            )

        df = pd.read_csv(path)
        return self._evaluate_quality(df, dataset_name=path.name)

    def _evaluate_quality(self, df: pd.DataFrame, dataset_name: str) -> GESLDataQualityReport:
        """Compute technical data quality metrics on a DataFrame."""
        total_records = len(df)
        notes: list[str] = []

        # Check classes
        target_col = "disturbance_type" if "disturbance_type" in df.columns else "event_type"
        if target_col in df.columns:
            class_dist = df[target_col].value_counts().to_dict()
        else:
            class_dist = {"unknown": total_records}
            notes.append("No disturbance_type or event_type column found.")

        # Check event grouping
        has_grouping = "event_id" in df.columns
        total_events = int(df["event_id"].nunique()) if has_grouping else total_records
        if not has_grouping:
            notes.append(
                "No event_id column found. Evaluation splits will use stratified fallback."
            )

        # Check missing values
        null_count = int(df.isnull().sum().sum())
        total_cells = df.size
        missing_pct = (null_count / total_cells * 100.0) if total_cells > 0 else 0.0
        if missing_pct > 0:
            notes.append(f"Missing values present: {missing_pct:.2f}% of data cells.")

        # Check duplicates
        dup_count = (
            int(df.duplicated(subset=["record_id"]).sum())
            if "record_id" in df.columns
            else int(df.duplicated().sum())
        )
        if dup_count > 0:
            notes.append(f"Found {dup_count} duplicate records.")

        sr = float(df["sampling_rate"].iloc[0]) if "sampling_rate" in df.columns else 30.0

        return GESLDataQualityReport(
            dataset_name=dataset_name,
            is_valid=total_records > 0,
            total_records=total_records,
            total_events=total_events,
            num_channels=len(df.columns),
            sampling_rate=sr,
            class_distribution=class_dist,
            missing_value_percentage=missing_pct,
            duplicate_records_count=dup_count,
            has_event_grouping=has_grouping,
            quality_notes=notes,
        )

    def prepare_benchmark_dataset(
        self,
        samples_per_class: int = 50,
        output_name: str = "gesl_synthetic_benchmark.csv",
        random_seed: int = 42,
    ) -> tuple[Path, DatasetManifest]:
        """Prepare and register a reproducible benchmark dataset using synthetic grid events.

        Saves signal metadata and signal array files with zero-leakage splits and manifest.
        """
        signals, df = generate_benchmark_dataset(
            samples_per_class=samples_per_class,
            random_seed=random_seed,
        )

        out_csv = self.processed_dir / output_name
        df.to_csv(out_csv, index=False)

        # Save individual waveform files in processed/waveforms/
        waveforms_dir = self.processed_dir / "waveforms"
        waveforms_dir.mkdir(exist_ok=True)
        for sig, (_, row) in zip(signals, df.iterrows(), strict=False):
            rec_id = row["record_id"]
            np.save(waveforms_dir / f"{rec_id}.npy", sig.data)

        # Split dataset
        train_df, val_df, test_df, strat = split_dataset(
            df,
            target_column="disturbance_type",
            group_column="event_id",
            random_seed=random_seed,
        )

        # Update CSV with split tags
        full_split_df = pd.concat([train_df, val_df, test_df])
        full_split_df.to_csv(out_csv, index=False)

        checksum = compute_file_sha256(out_csv)

        manifest = DatasetManifest(
            dataset_name=output_name,
            dataset_version="1.0.0",
            source="synthetic_gesl_compatible",
            description="Reproducible multi-class power system disturbance benchmark",
            checksum_sha256=checksum,
            total_records=len(df),
            sampling_rate=30.0,
            split_strategy=strat,
            partitions={
                "train": PartitionStats(
                    num_records=len(train_df),
                    class_distribution=train_df["disturbance_type"].value_counts().to_dict(),
                ),
                "validation": PartitionStats(
                    num_records=len(val_df),
                    class_distribution=val_df["disturbance_type"].value_counts().to_dict(),
                ),
                "test": PartitionStats(
                    num_records=len(test_df),
                    class_distribution=test_df["disturbance_type"].value_counts().to_dict(),
                ),
            },
        )

        manifest_path = self.manifest_dir / f"{Path(output_name).stem}_manifest.json"
        manifest.save(manifest_path)

        logger.info(f"Prepared benchmark dataset at {out_csv} with manifest at {manifest_path}")
        return out_csv, manifest

    def register_local_gesl_file(
        self,
        file_path: str | Path,
        event_column: str = "event_id",
        target_column: str = "disturbance_type",
    ) -> tuple[Path, DatasetManifest]:
        """Register an existing user-supplied GESL CSV file into the workspace."""
        src_path = Path(file_path)
        if not src_path.is_file():
            raise FileNotFoundError(f"Source file not found: {src_path}")

        df = pd.read_csv(src_path)
        dest_csv = self.raw_dir / src_path.name
        df.to_csv(dest_csv, index=False)

        # Split
        train_df, val_df, test_df, strat = split_dataset(
            df,
            target_column=target_column,
            group_column=event_column if event_column in df.columns else None,
        )

        proc_csv = self.processed_dir / f"processed_{src_path.name}"
        pd.concat([train_df, val_df, test_df]).to_csv(proc_csv, index=False)

        checksum = compute_file_sha256(proc_csv)
        manifest = DatasetManifest(
            dataset_name=src_path.name,
            source="gesl_local_registration",
            checksum_sha256=checksum,
            total_records=len(df),
            split_strategy=strat,
            partitions={
                "train": PartitionStats(num_records=len(train_df)),
                "validation": PartitionStats(num_records=len(val_df)),
                "test": PartitionStats(num_records=len(test_df)),
            },
        )
        manifest_path = self.manifest_dir / f"{src_path.stem}_manifest.json"
        manifest.save(manifest_path)

        return proc_csv, manifest
