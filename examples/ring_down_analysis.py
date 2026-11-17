"""Ring-Down Oscillation and Damping Estimation Example."""

from grid_disturbance.analysis.ringdown import analyze_ringdown
from grid_disturbance.synthetic.generator import SyntheticSignalGenerator


def main():
    print("=== Ring-Down Damping Analysis Example ===")
    gen = SyntheticSignalGenerator(random_seed=42)

    # 1. Well-damped ringdown (8% damping ratio)
    sig_well = gen.generate_damped_ringdown(frequency=0.60, damping_ratio=0.08, amplitude=1.5)
    ev_well = analyze_ringdown(sig_well.data, sampling_rate=30.0)

    print("\n--- Well-Damped Ring-Down ---")
    print(f"Ground Truth Damping Ratio: {sig_well.ground_truth.damping_ratio * 100:.1f}%")
    print(f"Estimated Damping Ratio: {ev_well.damping_ratio * 100:.1f}%")
    print(f"Damping Classification: {ev_well.damping_classification}")
    print(f"Envelope Fit R²: {ev_well.envelope_r2:.2f}")

    # 2. Poorly-damped ringdown (1.5% damping ratio)
    sig_poor = gen.generate_poorly_damped(frequency=0.45, damping_ratio=0.015, amplitude=1.2)
    ev_poor = analyze_ringdown(sig_poor.data, sampling_rate=30.0)

    print("\n--- Poorly-Damped Ring-Down ---")
    print(f"Ground Truth Damping Ratio: {sig_poor.ground_truth.damping_ratio * 100:.1f}%")
    print(f"Estimated Damping Ratio: {ev_poor.damping_ratio * 100:.1f}%")
    print(f"Damping Classification: {ev_poor.damping_classification}")


if __name__ == "__main__":
    main()
