"""Multi-Channel PMU Synchrophasor Cross-Spectral and Source Assessment Example."""

from grid_disturbance.analysis.source import assess_oscillation_source
from grid_disturbance.synthetic.scenarios import generate_multichannel_forced_oscillation_scenario


def main():
    print("=== Multi-Channel PMU Network Source Assessment Example ===")

    # 4-bus network scenario where Bus 1 is near the forced oscillation source
    scenario = generate_multichannel_forced_oscillation_scenario(frequency=0.80)
    print(f"Scenario: {scenario.scenario_name}")
    print(f"Channels: {scenario.channel_names}")
    print(f"True Source Bus: {scenario.true_source_channel}")
    print(f"True Disturbance Frequency: {scenario.true_source_frequency} Hz")

    # Assess source
    res = assess_oscillation_source(
        scenario.signals,
        scenario.channel_names,
        sampling_rate=scenario.sampling_rate,
        disturbance_frequency=scenario.true_source_frequency,
    )

    print("\n--- Assessment Outcome ---")
    print(f"Source Assessment Supported: {res.is_supported}")
    print(f"Capability Level: {res.capability_level}")
    print(f"Candidate Source Channel: {res.candidate_source_channel}")
    print(f"Candidate Source Region: {res.candidate_source_region}")
    print(f"Assessment Confidence: {res.confidence * 100:.1f}%")

    print("\nInter-Channel Cross-Spectral Relationships:")
    for rel in res.channel_relationships:
        print(f"  {rel.channel_a} -> {rel.channel_b}:")
        print(
            f"    Phase Diff: {rel.phase_difference_deg:+.1f}° | Coherence: {rel.magnitude_squared_coherence:.2f} | Time Delay: {rel.time_delay_ms:.1f} ms"
        )

    print("\nPhysical Limitations & Boundary:")
    for lim in res.limitations:
        print(f"  ! {lim}")


if __name__ == "__main__":
    main()
