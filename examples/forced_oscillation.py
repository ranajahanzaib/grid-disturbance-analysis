"""Forced Oscillation Detection Example."""

from grid_disturbance.analysis.forced import analyze_forced_oscillation
from grid_disturbance.synthetic.generator import SyntheticSignalGenerator


def main():
    print("=== Forced Oscillation Detection Example ===")
    gen = SyntheticSignalGenerator(random_seed=42)

    # 1. Generate sustained forced oscillation
    forced_sig = gen.generate_forced_oscillation(
        frequency=0.85, amplitude=1.0, sampling_rate=30.0, duration_sec=20.0
    )
    ev_forced = analyze_forced_oscillation(forced_sig.data, sampling_rate=30.0)

    print("\n--- Forced Oscillation Analysis ---")
    print(f"Is Forced: {ev_forced.is_forced}")
    print(f"Likelihood: {ev_forced.likelihood * 100:.1f}%")
    print(f"Dominant Frequency: {ev_forced.dominant_frequency:.2f} Hz")
    print(f"Persistence Ratio: {ev_forced.persistence_ratio * 100:.1f}%")
    print(f"Frequency Drift: {ev_forced.frequency_drift_hz:.3f} Hz")
    print("Supporting Evidence:")
    for pt in ev_forced.supporting_points:
        print(f"  + {pt}")

    # 2. Compare against a natural ringdown
    ringdown_sig = gen.generate_damped_ringdown(
        frequency=0.85, damping_ratio=0.08, amplitude=1.5, sampling_rate=30.0
    )
    ev_ringdown = analyze_forced_oscillation(ringdown_sig.data, sampling_rate=30.0)

    print("\n--- Ringdown Tested Under Forced Detector ---")
    print(f"Is Forced: {ev_ringdown.is_forced} (Likelihood: {ev_ringdown.likelihood * 100:.1f}%)")
    print("Contradicting Points:")
    for pt in ev_ringdown.contradicting_points:
        print(f"  - {pt}")


if __name__ == "__main__":
    main()
