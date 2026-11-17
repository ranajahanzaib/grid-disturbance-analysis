"""Overlapping Multi-Mode Oscillation Decomposition Example."""

from grid_disturbance.analysis.overlapping import analyze_overlapping_events
from grid_disturbance.synthetic.generator import SyntheticSignalGenerator


def main():
    print("=== Overlapping Multi-Mode Oscillation Decomposition ===")
    gen = SyntheticSignalGenerator(random_seed=42)

    # Simultaneous 0.35 Hz inter-area mode + 1.25 Hz forced mode
    sig = gen.generate_overlapping_events(f1=0.35, f2=1.25, amp1=1.0, amp2=0.70)
    res = analyze_overlapping_events(sig.data, sampling_rate=30.0)

    print(f"\n{res.summary}")
    print(f"Has Multiple Events: {res.has_multiple_events}")
    print(f"Total Decomposed Components: {res.num_components}")

    for comp in res.components:
        print(f"\nComponent {comp.component_id}:")
        print(f"  Center Frequency: {comp.center_frequency:.2f} Hz")
        print(f"  RMS Amplitude: {comp.rms_amplitude:.3f}")
        print(f"  Energy Percentage: {comp.energy_percentage:.1f}%")
        print(f"  Active Window: {comp.start_time_sec:.1f}s -> {comp.end_time_sec:.1f}s")
        print(f"  Is Sustained: {comp.is_sustained}")


if __name__ == "__main__":
    main()
