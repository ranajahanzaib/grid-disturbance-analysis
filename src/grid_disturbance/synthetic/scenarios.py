"""Multi-channel power grid disturbance scenarios for source localization testing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grid_disturbance.synthetic.generator import SyntheticSignalGenerator


@dataclass
class MultichannelScenario:
    """A simulated multi-channel grid recording across several PMU measurement stations."""

    scenario_name: str
    time: np.ndarray
    channel_names: list[str]
    signals: np.ndarray  # Shape: (num_channels, num_samples)
    sampling_rate: float
    true_source_channel: str
    true_source_frequency: float
    phase_lags_deg: dict[str, float]
    inter_area_mode: bool = False


def generate_multichannel_forced_oscillation_scenario(
    frequency: float = 0.80,
    sampling_rate: float = 30.0,
    duration_sec: float = 20.0,
    random_seed: int = 42,
) -> MultichannelScenario:
    """Simulate a 4-station PMU network observing a forced oscillation.

    Bus 1: Source generator bus (reference 0 deg phase, largest amplitude 1.0)
    Bus 2: Neighboring substation (-20 deg phase lag, amplitude 0.85)
    Bus 3: Intermediate transmission bus (-50 deg phase lag, amplitude 0.60)
    Bus 4: Remote receiving load center (-95 deg phase lag, amplitude 0.35)
    """
    gen = SyntheticSignalGenerator(random_seed=random_seed)

    n = int(sampling_rate * duration_sec)
    t = np.arange(n) / sampling_rate

    channels = ["PMU_BUS_1", "PMU_BUS_2", "PMU_BUS_3", "PMU_BUS_4"]
    amplitudes = [1.0, 0.85, 0.60, 0.35]
    phase_lags = [0.0, -20.0, -50.0, -95.0]

    signals = np.zeros((len(channels), n))
    phase_dict = {}

    for i, (ch, amp, phi_deg) in enumerate(zip(channels, amplitudes, phase_lags, strict=False)):
        phi_rad = np.radians(phi_deg)
        pure_osc = amp * np.cos(2 * np.pi * frequency * t + phi_rad)
        noise = gen._generate_ambient_noise(n, snr_db=28.0, signal_power=amp**2 / 2.0)
        signals[i, :] = pure_osc + noise
        phase_dict[ch] = phi_deg

    return MultichannelScenario(
        scenario_name="4_bus_forced_oscillation",
        time=t,
        channel_names=channels,
        signals=signals,
        sampling_rate=sampling_rate,
        true_source_channel="PMU_BUS_1",
        true_source_frequency=frequency,
        phase_lags_deg=phase_dict,
        inter_area_mode=False,
    )


def generate_two_area_inter_area_scenario(
    frequency: float = 0.40,  # 0.40 Hz inter-area electromechanical mode
    sampling_rate: float = 30.0,
    duration_sec: float = 25.0,
    damping_ratio: float = 0.035,
    random_seed: int = 42,
) -> MultichannelScenario:
    """Simulate classic 2-area inter-area oscillation (Area 1 swinging 180 deg out of phase with Area 2).

    Area 1 (PMU_1, PMU_2): phase ~ 0 deg
    Area 2 (PMU_3, PMU_4): phase ~ 180 deg (opposite swinging group)
    """
    gen = SyntheticSignalGenerator(random_seed=random_seed)
    n = int(sampling_rate * duration_sec)
    t = np.arange(n) / sampling_rate

    channels = ["AREA1_PMU_A", "AREA1_PMU_B", "AREA2_PMU_C", "AREA2_PMU_D"]
    amplitudes = [1.2, 1.0, 1.1, 0.9]
    phase_lags = [0.0, -10.0, 175.0, 185.0]

    omega = 2 * np.pi * frequency
    alpha = omega * (damping_ratio / np.sqrt(max(1e-4, 1.0 - damping_ratio**2)))
    decay = np.exp(-alpha * t)

    signals = np.zeros((len(channels), n))
    phase_dict = {}

    for i, (ch, amp, phi_deg) in enumerate(zip(channels, amplitudes, phase_lags, strict=False)):
        phi_rad = np.radians(phi_deg)
        osc = amp * decay * np.cos(omega * t + phi_rad)
        noise = gen._generate_ambient_noise(n, snr_db=26.0, signal_power=amp**2 / 4.0)
        signals[i, :] = osc + noise
        phase_dict[ch] = phi_deg

    return MultichannelScenario(
        scenario_name="two_area_inter_area_ringdown",
        time=t,
        channel_names=channels,
        signals=signals,
        sampling_rate=sampling_rate,
        true_source_channel="AREA1_PMU_A",
        true_source_frequency=frequency,
        phase_lags_deg=phase_dict,
        inter_area_mode=True,
    )
