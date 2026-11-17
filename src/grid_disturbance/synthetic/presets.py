"""Presets and dataset generation utilities for benchmarks and testing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from grid_disturbance.synthetic.generator import SyntheticSignal, SyntheticSignalGenerator


def generate_single_test_signal(
    disturbance_type: str,
    sampling_rate: float = 30.0,
    duration_sec: float = 20.0,
    random_seed: int = 42,
) -> SyntheticSignal:
    """Generate a single deterministic synthetic signal of a specified type."""
    gen = SyntheticSignalGenerator(random_seed=random_seed)

    if disturbance_type == "baseline_noise":
        return gen.generate_baseline(sampling_rate=sampling_rate, duration_sec=duration_sec)
    if disturbance_type == "forced_oscillation":
        return gen.generate_forced_oscillation(
            frequency=0.85, amplitude=1.2, sampling_rate=sampling_rate, duration_sec=duration_sec
        )
    if disturbance_type == "damped_ringdown":
        return gen.generate_damped_ringdown(
            frequency=0.60,
            damping_ratio=0.08,
            amplitude=1.5,
            sampling_rate=sampling_rate,
            duration_sec=duration_sec,
        )
    if disturbance_type == "poorly_damped":
        return gen.generate_poorly_damped(
            frequency=0.45,
            damping_ratio=0.015,
            amplitude=1.2,
            sampling_rate=sampling_rate,
            duration_sec=duration_sec,
        )
    if disturbance_type == "growing_oscillation":
        return gen.generate_growing_oscillation(
            frequency=0.70,
            growth_rate=0.08,
            amplitude=0.15,
            sampling_rate=sampling_rate,
            duration_sec=duration_sec,
        )
    if disturbance_type == "frequency_excursion":
        return gen.generate_frequency_excursion(
            initial_rocof=-0.25,
            nadir_delta_f=-0.40,
            sampling_rate=sampling_rate,
            duration_sec=duration_sec,
        )
    if disturbance_type == "frequency_drift":
        return gen.generate_frequency_drift(
            drift_rate=0.04, sampling_rate=sampling_rate, duration_sec=duration_sec
        )
    if disturbance_type == "overlapping_events":
        return gen.generate_overlapping_events(
            f1=0.35,
            f2=1.20,
            amp1=1.0,
            amp2=0.7,
            sampling_rate=sampling_rate,
            duration_sec=duration_sec,
        )

    raise ValueError(f"Unknown synthetic disturbance type: {disturbance_type}")


def generate_benchmark_dataset(
    samples_per_class: int = 40,
    sampling_rate: float = 30.0,
    duration_sec: float = 20.0,
    random_seed: int = 42,
) -> tuple[list[SyntheticSignal], pd.DataFrame]:
    """Generate a balanced benchmark dataset across standard disturbance classes.

    Returns:
        signals: list of SyntheticSignal objects.
        metadata_df: DataFrame with labels, event_ids (for grouped splitting), and ground truth parameters.
    """
    classes = [
        "baseline_noise",
        "forced_oscillation",
        "damped_ringdown",
        "poorly_damped",
        "frequency_excursion",
        "overlapping_events",
    ]

    signals: list[SyntheticSignal] = []
    metadata_rows: list[dict] = []

    rng = np.random.default_rng(random_seed)

    for cls_name in classes:
        # Simulate distinct physical event groups (e.g. 5 event families per class)
        num_events = max(1, samples_per_class // 5)
        for i in range(samples_per_class):
            event_id = f"event_{cls_name}_{i % num_events}"
            sub_seed = int(rng.integers(0, 1000000))
            gen = SyntheticSignalGenerator(random_seed=sub_seed)

            if cls_name == "baseline_noise":
                sig = gen.generate_baseline(
                    sampling_rate=sampling_rate,
                    duration_sec=duration_sec,
                    snr_db=float(rng.uniform(35, 45)),
                )
            elif cls_name == "forced_oscillation":
                f = float(rng.uniform(0.2, 3.5))
                amp = float(rng.uniform(0.5, 2.0))
                sig = gen.generate_forced_oscillation(
                    frequency=f,
                    amplitude=amp,
                    sampling_rate=sampling_rate,
                    duration_sec=duration_sec,
                    snr_db=float(rng.uniform(20, 32)),
                )
            elif cls_name == "damped_ringdown":
                f = float(rng.uniform(0.3, 1.8))
                zeta = float(rng.uniform(0.05, 0.15))
                sig = gen.generate_damped_ringdown(
                    frequency=f,
                    damping_ratio=zeta,
                    amplitude=float(rng.uniform(0.8, 2.5)),
                    sampling_rate=sampling_rate,
                    duration_sec=duration_sec,
                    snr_db=float(rng.uniform(20, 32)),
                )
            elif cls_name == "poorly_damped":
                f = float(rng.uniform(0.2, 1.5))
                zeta = float(rng.uniform(0.005, 0.028))
                sig = gen.generate_poorly_damped(
                    frequency=f,
                    damping_ratio=zeta,
                    amplitude=float(rng.uniform(0.8, 2.0)),
                    sampling_rate=sampling_rate,
                    duration_sec=duration_sec,
                    snr_db=float(rng.uniform(20, 32)),
                )
            elif cls_name == "frequency_excursion":
                rocof = float(rng.uniform(-0.40, -0.15))
                nadir = float(rng.uniform(-0.60, -0.25))
                sig = gen.generate_frequency_excursion(
                    initial_rocof=rocof,
                    nadir_delta_f=nadir,
                    sampling_rate=sampling_rate,
                    duration_sec=duration_sec,
                    snr_db=float(rng.uniform(25, 38)),
                )
            elif cls_name == "overlapping_events":
                f1 = float(rng.uniform(0.25, 0.65))
                f2 = float(rng.uniform(1.10, 2.80))
                sig = gen.generate_overlapping_events(
                    f1=f1,
                    f2=f2,
                    amp1=float(rng.uniform(0.6, 1.5)),
                    amp2=float(rng.uniform(0.4, 1.2)),
                    sampling_rate=sampling_rate,
                    duration_sec=duration_sec,
                    snr_db=float(rng.uniform(20, 30)),
                )

            record_id = f"{cls_name}_{i:04d}"
            signals.append(sig)
            metadata_rows.append(
                {
                    "record_id": record_id,
                    "event_id": event_id,
                    "disturbance_type": cls_name,
                    "sampling_rate": sampling_rate,
                    "duration_sec": duration_sec,
                    "num_samples": len(sig.data),
                    "primary_frequency": sig.ground_truth.primary_frequency,
                    "secondary_frequency": sig.ground_truth.secondary_frequency,
                    "true_damping_ratio": sig.ground_truth.damping_ratio,
                    "true_amplitude": sig.ground_truth.amplitude,
                    "true_snr_db": sig.ground_truth.snr_db,
                    "true_rocof": sig.ground_truth.rocof_hz_per_sec,
                }
            )

    df = pd.DataFrame(metadata_rows)
    return signals, df
