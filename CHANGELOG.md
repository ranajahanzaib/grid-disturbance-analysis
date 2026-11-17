# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-17

### Added

- **Core Signal Engine (`grid_disturbance.signals`):**
  - Preprocessing pipeline with audit trails, detrending, filtering, outlier clipping, and anti-aliasing resampling.
  - Spectral analysis: FFT, Welch PSD, spectral centroid, spectral bandwidth, spectral entropy, and Wiener flatness.
  - Time-frequency analysis: STFT spectrograms, Continuous Wavelet Transform (CWT), and Discrete Wavelet Transform (DWT).
  - Exponential envelope fitting, Hilbert analytic transform, and damping ratio (\(\zeta\)) estimation.
  - Matrix Pencil modal estimation, frequency drift tracking, and amplitude modulation characterization.
  - Multi-component spectral decomposition.
- **Disturbance Analysis & Discrimination (`grid_disturbance.analysis`):**
  - Forced oscillation detector with persistence, spectral concentration, and envelope variation checks.
  - Natural ring-down detector with damping classification (well-damped, poorly-damped, undamped/growing).
  - Forced vs. natural oscillation discriminator supporting indeterminate class results.
  - Frequency instability analyzer tracking RoCoF, nadir, zenith, and quasi-steady excursions.
  - Generator electromechanical oscillation hypothesis evaluation (inter-area vs. local plant modes).
  - Multi-channel PMU cross-spectral density, coherence, and source direction assessment with single-channel refusal boundaries.
  - High-level `DisturbanceDiagnosticEngine`.
- **Machine Learning Architecture (`grid_disturbance.ml`):**
  - Standardized 34-dimensional feature vectorizer with zero-leakage training-set scalers.
  - Classical classifiers (Random Forest, Extra Trees, Gradient Boosting, Logistic Regression, SVM).
  - 1D Temporal CNN (`TemporalCNN`) for raw signal windows.
  - 2D Time-Frequency CNN (`SpectrogramCNN`) on spectrograms and scalograms.
  - Comprehensive evaluator (`accuracy`, `macro_f1`, `weighted_f1`, `confusion_matrix`, `per_class_metrics`).
  - Expected Calibration Error (ECE) reliability metrics and model explainability.
- **Unsupervised Discovery (`grid_disturbance.clustering`):**
  - Feature clustering (K-Means, DBSCAN) with 2D PCA latent projections.
  - Anomaly detection via Isolation Forest.
- **Synthetic Signal Generator (`grid_disturbance.synthetic`):**
  - Deterministic generator modeling 12 distinct power system disturbance scenarios with ground truth.
  - 4-bus and 2-area multi-channel PMU network scenarios.
- **Diagnostic Reporting & Persistence (`grid_disturbance.reporting` & `storage`):**
  - Pydantic schema for `StructuredDiagnosticReport`.
  - Self-contained HTML and JSON export generators.
  - Local SQLite persistence repository.
- **Interactive Interfaces:**
  - Full-featured Typer CLI (`grid-disturbance`).
  - Streamlit multi-tab diagnostic engineering dashboard.
- **Testing & Quality Assurance:**
  - 31 unit and integration tests across DSP, ML, synthetic generation, and end-to-end pipelines.
  - Ruff formatting and linting, Mypy static type checking.
