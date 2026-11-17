"""Dataset manifest management and provenance tracking."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class PartitionStats(BaseModel):
    num_records: int = 0
    class_distribution: dict[str, int] = Field(default_factory=dict)


class DatasetManifest(BaseModel):
    """Manifest describing a registered or prepared dataset."""

    dataset_name: str
    dataset_version: str = "1.0.0"
    source: str = "synthetic"  # "gesl", "synthetic", or "local_file"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    description: str = ""
    checksum_sha256: str = ""
    total_records: int = 0
    sampling_rate: float = 30.0
    nominal_frequency: float = 60.0
    split_strategy: str = "event_grouped"
    partitions: dict[str, PartitionStats] = Field(default_factory=dict)
    preprocessing_applied: dict[str, str | bool | float] = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)

    def save(self, file_path: str | Path) -> None:
        """Save manifest to JSON."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, file_path: str | Path) -> DatasetManifest:
        """Load manifest from JSON."""
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)


def compute_file_sha256(file_path: str | Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha = hashlib.sha256()
    path = Path(file_path)
    with path.open("rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()
