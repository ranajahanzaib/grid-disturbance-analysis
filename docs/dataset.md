# Dataset Integration & Provenance: GESL

## 1. Grid Event Signature Library (GESL)

### 1.1 Dataset Overview

The Grid Event Signature Library (GESL) is an open-access repository of real-world and high-fidelity power system disturbance waveforms funded by the U.S. Department of Energy (DOE) Office of Electricity and curated jointly by Oak Ridge National Laboratory (ORNL), Lawrence Livermore National Laboratory (LLNL), and Pacific Northwest National Laboratory (PNNL).

- **Portal:** [https://openenergyhub.ornl.gov/explore/dataset/gesl/](https://openenergyhub.ornl.gov/explore/dataset/gesl/)
- **Total Record Scale:** Approximately 203,400 records.
- **Forced Oscillation Records:** ~167,400 records.
- **Ring-Down Records:** ~36,000 records.

### 1.2 Schema & Attributes

Standard synchrophasor PMU recordings in GESL include:

- `record_id`: Unique identifier for each waveform snippet.
- `event_id`: Physical disturbance event grouping (e.g., specific generator trip or oscillating asset).
- `timestamp`: UTC or relative time in seconds.
- `sampling_rate`: Synchrophasor reporting rate (typically 30 Hz or 60 Hz).
- `frequency`: Frequency in Hz or frequency deviation \(\Delta f\).
- `voltage_magnitude`: Bus positive-sequence voltage magnitude (kV or per-unit).
- `active_power`: Real power injection / flow (MW or per-unit).
- `reactive_power`: Reactive power injection / flow (MVAR or per-unit).
- `disturbance_type`: Primary ground-truth disturbance label.

### 1.3 Access Protocol & Authentication

- Bulk downloads of raw utility-donated GESL waveforms require user registration and access approval through the ORNL OpenEnergy Hub portal.
- `grid-disturbance-analysis` provides an ingestion adapter (`GESLDatasetManager`) with:
  - Local CSV/Parquet dataset registration (`register_local_gesl_file`).
  - Automated format detection.
  - Manifest and SHA256 checksum tracking.
  - Seamless fallback to bundled synthetic benchmark generation (`prepare_benchmark_dataset`) so that users can run the entire workflow without requiring immediate ORNL credentials.

---

## 2. Dataset Management Commands

```bash
# Inspect dataset quality, schema, and class distribution
grid-disturbance data inspect

# Prepare reproducible benchmark dataset
grid-disturbance data prepare --samples 50

# Validate an ingested file for sampling consistency and defects
grid-disturbance data validate sample.csv

# View registered dataset manifests and checksums
grid-disturbance data stats
```
