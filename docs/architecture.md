# System Architecture & Design Specification

## 1. High-Level Architecture Overview

`grid-disturbance-analysis` is structured as a decoupled, layered scientific computing engine designed for power grid synchrophasor disturbance analysis, oscillation characterization, and origin assessment.

```mermaid
flowchart TD
    subgraph Ingestion ["Data Ingestion & Validation"]
        A[Synchrophasor CSV / Parquet / PMU Stream] --> B[Signal Validation Engine]
        B --> C[Preprocessing Pipeline]
        C --> D[Audit Trail & Transformation Logger]
    end

    subgraph CoreEngine ["Digital Signal Processing Engine"]
        D --> E[Spectral & PSD Analysis]
        D --> F[Time-Frequency Transforms STFT / CWT / DWT]
        D --> G[Envelope & Damping Estimator]
        D --> H[Modal Decomposition Matrix Pencil]
    end

    subgraph Diagnostics ["Disturbance Diagnostic Layer"]
        E & F & G & H --> I[Forced Oscillation Detector]
        E & F & G & H --> J[Natural Ring-Down Detector]
        E & F & G & H --> K[Frequency Instability Analyzer]
        E & F & G & H --> L[Overlapping Modes Separator]
        E & F & G & H --> M[Multi-Channel Source Assessment]
        I & J --> N[Forced vs Natural Discriminator]
    end

    subgraph ML ["Machine Learning Architecture"]
        D --> O[SignalFeatureVectorizer]
        O --> P[Classical Classifiers RF / ET / GBDT / LR]
        D --> Q[1D Temporal CNN]
        F --> R[2D Spectrogram CNN]
    end

    subgraph Output ["Explainability & Reporting"]
        N & K & L & M & P & Q & R --> S[Disturbance Diagnostic Engine]
        S --> T[Structured Diagnostic Report JSON]
        S --> U[Interactive HTML Report]
        S --> V[SQLite Persistence]
        S --> W[Streamlit Diagnostic App & Typer CLI]
    end
```

## 2. Architectural Boundaries & Data Contracts

The architecture strictly decouples signal processing from presentation interfaces:

1. **Core Signal Processing (`grid_disturbance.signals`):**
   - Independent of Streamlit, Typer, and PyTorch. Operates strictly on NumPy arrays and SciPy algorithms.
2. **Analysis & Disturbance Logic (`grid_disturbance.analysis`):**
   - Implements power system physical domain rules (damping thresholds, persistence checks, cross-spectral coherence).
   - Distinguishes direct measurements, derived features, and model hypotheses.
3. **Machine Learning (`grid_disturbance.ml`):**
   - Consumes standardized tabular features or raw windows.
   - Enforces zero-leakage grouped splitting (`GroupShuffleSplit` on `event_id`).
4. **Diagnostic Reporting (`grid_disturbance.reporting`):**
   - Produces strictly validated Pydantic models (`StructuredDiagnosticReport`).
5. **Persistence (`grid_disturbance.storage`):**
   - Self-contained SQLite database for persistent analysis history.
6. **Interfaces (`grid_disturbance.cli` and `grid_disturbance.app`):**
   - Thin presentation layers delegating to core packages.
