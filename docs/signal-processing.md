# Digital Signal Processing Specification

## 1. Mathematical Models & Principles

### 1.1 Damped Oscillation Model

Natural electromechanical ring-downs in power systems following a disturbance (fault, generator trip, line switching) are modeled as:

\[
x(t) = A e^{-\alpha t} \cos(2\pi f_d t + \phi) + \epsilon(t)
\]

where:

- \(A\): Initial disturbance amplitude at onset.
- \(\alpha\): Exponential decay rate (\(\text{s}^{-1}\)). Positive \(\alpha > 0\) indicates attenuation; \(\alpha < 0\) indicates exponential dynamic instability (growing oscillations).
- \(f_d\): Damped oscillation frequency (\(\text{Hz}\)), with \(\omega_d = 2\pi f_d\).
- \(\phi\): Initial phase angle in radians.
- \(\epsilon(t)\): Ambient colored power system noise.

### 1.2 Damping Ratio Estimation

The dimensionless damping ratio \(\zeta\) is defined by:

\[
\zeta = \frac{\alpha}{\sqrt{\alpha^2 + \omega_d^2}} = \frac{\alpha}{\sqrt{\alpha^2 + (2\pi f_d)^2}}
\]

For typical low-damping power system modes (\(\zeta \ll 1\)), this is approximated as:

\[
\zeta \approx \frac{\alpha}{2\pi f_d}
\]

Classification criteria:

- **Well-Damped Mode:** \(\zeta \ge 5.0\%\) (North American NERC / WECC grid standard threshold).
- **Poorly Damped Mode:** \(0.0\% < \zeta < 5.0\%\). Prone to inter-area resonance and cascading amplification.
- **Undamped / Growing Mode:** \(\zeta \le 0.0\%\). Dynamic instability requiring rapid operator or remedial action scheme intervention.

---

## 2. Spectral Analysis

### 2.1 Fast Fourier Transform (FFT)

One-sided amplitude spectrum scaled by coherent window gain:

\[
X(k) = \sum\_{n=0}^{N-1} x[n] w[n] e^{-j 2\pi k n / N}
\]

Scaled magnitude:

\[
|X(f)| = \frac{2}{\sum\_{n=0}^{N-1} w[n]} |X(k)|
\]

### 2.2 Welch Power Spectral Density (PSD)

Averages periodograms over overlapping segments to reduce variance:

\[
P*{xx}(f) = \frac{1}{K} \sum*{i=1}^K I_i(f)
\]

### 2.3 Spectral Metrics

- **Spectral Centroid:**
  \[
  f_c = \frac{\sum f \cdot P(f)}{\sum P(f)}
  \]
- **Spectral Spread (Bandwidth):**
  \[
  \sigma_f = \sqrt{\frac{\sum (f - f_c)^2 P(f)}{\sum P(f)}}
  \]
- **Spectral Flatness (Wiener Entropy):**
  \[
  \text{Flatness} = \frac{\exp\left(\frac{1}{N}\sum \ln P(f_i)\right)}{\frac{1}{N}\sum P(f_i)}
  \]
  Near 0 indicates sharp tonal oscillation; near 1 indicates flat ambient white/pink noise.
- **Spectral Concentration:** Ratio of power concentrated within \(\pm 3\) frequency bins around dominant peak to total spectral power.

---

## 3. Time-Frequency Representations

### 3.1 Short-Time Fourier Transform (STFT)

\[
\text{STFT}\{x[n]\}(m, \omega) = \sum\_{n=-\infty}^{\infty} x[n] w[n-m] e^{-j \omega n}
\]

### 3.2 Continuous Wavelet Transform (CWT)

\[
W(a, b) = \frac{1}{\sqrt{|a|}} \int\_{-\infty}^{\infty} x(t) \psi^\*\left(\frac{t-b}{a}\right) dt
\]
Using complex Morlet wavelets (\(\psi(t) = \pi^{-1/4} e^{j \omega_0 t} e^{-t^2/2}\)) mapped to pseudo-frequencies.

### 3.3 Analytic Signal & Hilbert Transform

\[
z(t) = x(t) + j \mathcal{H}\{x(t)\} = A(t) e^{j \phi(t)}
\]

- Instantaneous Amplitude: \(A(t) = |z(t)|\).
- Instantaneous Frequency: \(f(t) = \frac{1}{2\pi} \frac{d\phi}{dt}\).
