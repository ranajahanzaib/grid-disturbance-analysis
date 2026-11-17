# Scientific Methodology & Validation Standards

## 1. Scientific Verification Framework

To guarantee scientific credibility and eliminate fabricated outputs, the system adheres to strict engineering validation principles:

### 1.1 Synthetic Ground-Truth Recovery

Every algorithm in the DSP engine is validated against synthetic signals with analytically known parameters:

- **Frequency Recovery:** FFT and Welch PSD peak recovery tested across multiple frequency bands (0.1 to 10 Hz) with tolerance \(|\hat{f} - f\_{\text{true}}| < 0.05\text{ Hz}\).
- **Damping Recovery:** Exponential envelope fits validated against known damping ratios (\(\zeta = 1.5\%\), \(8.0\%\), \(12.0\%\)) with tolerance \(|\hat{\zeta} - \zeta\_{\text{true}}| < 0.02\).
- **Phase Lag Recovery:** Multi-channel cross-spectral phase angles validated against known synthetic delays (\(-20^\circ, -50^\circ, -95^\circ\)) with \(< 1.0^\circ\) error.

### 1.2 Separation of Observation from Inference

All diagnostic reports strictly segregate:

1. **Direct Measurements:** Raw sample count, sampling frequency, observed duration, channel list.
2. **Derived Features:** Measured FFT peaks, spectral concentration, envelope coefficient of variation, RoCoF.
3. **Model Predictions:** Machine learning class probabilities and classifier confidence scores.
4. **Engineering Interpretations:** Plain-language physical deductions explicitly qualified by confidence boundaries and stated limitations.

### 1.3 Physical Causation Boundaries

The engine enforces boundaries on automated claims:

- A single measurement channel can **never** locate a physical disturbance source.
- Attributing an oscillation to an internal generator component is strictly treated as a hypothesis unless machine rotor-angle telemetry or unit-terminal PMUs are present.
- Non-linear harmonics are cross-checked before declaring multiple independent simultaneous physical events.
