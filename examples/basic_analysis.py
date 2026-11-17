"""Basic Signal Processing Analysis Example."""

from grid_disturbance.signals.filtering import detrend_signal, normalize_signal
from grid_disturbance.signals.spectral import analyze_spectrum
from grid_disturbance.synthetic.generator import SyntheticSignalGenerator


def main():
    print("=== Basic Signal Processing Example ===")
    gen = SyntheticSignalGenerator(random_seed=42)
    sig = gen.generate_forced_oscillation(
        frequency=1.20, amplitude=1.5, sampling_rate=30.0, duration_sec=15.0
    )

    print(f"Generated signal: {len(sig.data)} samples @ {sig.ground_truth.sampling_rate} Hz")
    print(f"Ground truth frequency: {sig.ground_truth.primary_frequency} Hz")

    # Preprocessing
    clean = detrend_signal(sig.data, method="linear")
    clean = normalize_signal(clean, method="standardize")

    # Spectral Analysis
    spec = analyze_spectrum(clean, sampling_rate=30.0)
    print(f"Estimated Dominant Frequency: {spec.dominant_frequency:.2f} Hz")
    print(f"Spectral Concentration: {spec.spectral_concentration:.2f}")
    print(f"Spectral Entropy: {spec.spectral_entropy:.2f}")
    print(f"Identified Peak Count: {len(spec.peaks)}")
    for i, p in enumerate(spec.peaks[:3]):
        print(
            f"  Peak {i + 1}: {p.frequency:.2f} Hz, Power: {p.power:.3f}, Prominence: {p.prominence:.3f}"
        )


if __name__ == "__main__":
    main()
