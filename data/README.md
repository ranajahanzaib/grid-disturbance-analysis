# Dataset Storage & Directory Structure

This directory stores raw, processed, and synthetic synchrophasor PMU measurements.

## Subdirectories:

- `raw/`: Locally imported or downloaded raw GESL CSV / Parquet recordings.
- `processed/`: Validated, preprocessed, and partition-tagged datasets ready for ML training and benchmarking.
  - `waveforms/`: Extracted 1D NumPy waveform files (`.npy`) indexed by `record_id`.
- `synthetic/`: Ground-truth synthetic scenarios generated for algorithm validation and unit testing.
- `manifests/`: Dataset provenance manifests (`*_manifest.json`) recording checksums, sample counts, split strategies, and creation timestamps.
- `storage.db`: SQLite database storing historical disturbance analyses and structured diagnostic reports.

## Ingestion Commands:

```bash
# Prepare reproducible benchmark dataset
grid-disturbance data prepare --samples 50

# Inspect dataset quality
grid-disturbance data inspect

# Check registered manifests
grid-disturbance data stats
```
