"""Physically-grounded synthetic signal generation for power grid disturbances."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class DisturbanceGroundTruth:
    """Ground-truth metadata for a synthetic signal."""

    disturbance_type: str
    nominal_frequency: float = 60.0
    sampling_rate: float = 30.0
    duration_sec: float = 20.0
    num_samples: int = 600
    primary_frequency: float = 0.0
    secondary_frequency: float | None = None
    damping_ratio: float = 0.0
    damping_coefficient: float = 0.0
    amplitude: float = 1.0
    snr_db: float = 30.0
    onset_time_sec: float = 0.0
    event_duration_sec: float = 20.0
    rocof_hz_per_sec: float = 0.0
    modulation_frequency: float | None = None
    modulation_index: float | None = None
    channels: list[str] = field(default_factory=lambda: ["channel_1"])
    metadata: dict = field(default_factory=dict)


@dataclass
class SyntheticSignal:
    """Container for synthetic signal waveform and ground truth."""

    time: np.ndarray
    data: np.ndarray  # Shape: (num_samples,) or (num_channels, num_samples)
    ground_truth: DisturbanceGroundTruth


class SyntheticSignalGenerator:
    """Deterministic generator for power grid disturbance signatures."""

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)

    def _generate_ambient_noise(
        self,
        n: int,
        snr_db: float = 30.0,
        signal_power: float = 1.0,
        colored: bool = True,
    ) -> np.ndarray:
        """Generate ambient power grid noise (pink/colored or Gaussian)."""
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        white = self.rng.normal(0, np.sqrt(noise_power), n)

        if not colored or n < 10:
            return white

        # 1/f pink noise approximation via simple IIR filter
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1.0, -2.494956002, 2.017265875, -0.522189400]
        import scipy.signal as signal

        pink = signal.lfilter(b, a, white)
        # Rescale to match target noise power
        p_act = np.mean(pink**2)
        if p_act > 1e-12:
            pink = pink * np.sqrt(noise_power / p_act)
        return pink

    def generate_baseline(
        self,
        sampling_rate: float = 30.0,
        duration_sec: float = 20.0,
        snr_db: float = 40.0,
    ) -> SyntheticSignal:
        """1. Clean baseline signal (small ambient fluctuations around 0)."""
        n = int(sampling_rate * duration_sec)
        t = np.arange(n) / sampling_rate
        noise = self._generate_ambient_noise(n, snr_db=snr_db, signal_power=0.01)

        gt = DisturbanceGroundTruth(
            disturbance_type="baseline_noise",
            sampling_rate=sampling_rate,
            duration_sec=duration_sec,
            num_samples=n,
            primary_frequency=0.0,
            damping_ratio=0.0,
            amplitude=0.0,
            snr_db=snr_db,
        )
        return SyntheticSignal(time=t, data=noise, ground_truth=gt)

    def generate_forced_oscillation(
        self,
        frequency: float = 0.85,
        amplitude: float = 1.0,
        sampling_rate: float = 30.0,
        duration_sec: float = 20.0,
        onset_time_sec: float = 2.0,
        snr_db: float = 25.0,
        harmonics: list[float] | None = None,
    ) -> SyntheticSignal:
        """2. Forced oscillation (sustained external periodic excitation).

        x(t) = A * cos(2*pi*f*t + phi) + harmonics + noise
        """
        n = int(sampling_rate * duration_sec)
        t = np.arange(n) / sampling_rate
        phi = self.rng.uniform(0, 2 * np.pi)

        # Active mask
        active = t >= onset_time_sec
        waveform = np.zeros(n)
        waveform[active] = amplitude * np.cos(
            2 * np.pi * frequency * (t[active] - onset_time_sec) + phi
        )

        # Optional harmonics (e.g. 2nd or 3rd harmonic from non-linear actuator)
        if harmonics:
            for order, h_amp in enumerate(harmonics, start=2):
                waveform[active] += h_amp * np.cos(
                    2 * np.pi * frequency * order * (t[active] - onset_time_sec) + phi
                )

        noise = self._generate_ambient_noise(n, snr_db=snr_db, signal_power=amplitude**2 / 2.0)
        signal_out = waveform + noise

        gt = DisturbanceGroundTruth(
            disturbance_type="forced_oscillation",
            sampling_rate=sampling_rate,
            duration_sec=duration_sec,
            num_samples=n,
            primary_frequency=frequency,
            damping_ratio=0.0,  # Zero damping -> strictly sustained
            damping_coefficient=0.0,
            amplitude=amplitude,
            snr_db=snr_db,
            onset_time_sec=onset_time_sec,
            event_duration_sec=duration_sec - onset_time_sec,
        )
        return SyntheticSignal(time=t, data=signal_out, ground_truth=gt)

    def generate_damped_ringdown(
        self,
        frequency: float = 0.65,
        damping_ratio: float = 0.08,  # 8% damping (well damped)
        amplitude: float = 1.5,
        sampling_rate: float = 30.0,
        duration_sec: float = 20.0,
        onset_time_sec: float = 2.0,
        snr_db: float = 25.0,
    ) -> SyntheticSignal:
        """3. Natural ring-down oscillation following a disturbance.

        x(t) = A * exp(-alpha * t) * cos(2*pi*f*t + phi)
        where alpha = zeta * omega_n / sqrt(1 - zeta^2)
        """
        n = int(sampling_rate * duration_sec)
        t = np.arange(n) / sampling_rate
        phi = self.rng.uniform(0, 2 * np.pi)

        omega_d = 2 * np.pi * frequency
        # alpha = omega_d * (zeta / sqrt(1 - zeta^2))
        alpha = omega_d * (damping_ratio / np.sqrt(max(1e-4, 1.0 - damping_ratio**2)))

        active = t >= onset_time_sec
        waveform = np.zeros(n)
        tau = t[active] - onset_time_sec
        waveform[active] = amplitude * np.exp(-alpha * tau) * np.cos(omega_d * tau + phi)

        noise = self._generate_ambient_noise(n, snr_db=snr_db, signal_power=amplitude**2 / 4.0)
        signal_out = waveform + noise

        gt = DisturbanceGroundTruth(
            disturbance_type="damped_ringdown",
            sampling_rate=sampling_rate,
            duration_sec=duration_sec,
            num_samples=n,
            primary_frequency=frequency,
            damping_ratio=damping_ratio,
            damping_coefficient=alpha,
            amplitude=amplitude,
            snr_db=snr_db,
            onset_time_sec=onset_time_sec,
            event_duration_sec=duration_sec - onset_time_sec,
        )
        return SyntheticSignal(time=t, data=signal_out, ground_truth=gt)

    def generate_poorly_damped(
        self,
        frequency: float = 0.45,
        damping_ratio: float = 0.015,  # 1.5% damping (poorly damped, lingers long time)
        amplitude: float = 1.2,
        sampling_rate: float = 30.0,
        duration_sec: float = 20.0,
        onset_time_sec: float = 1.0,
        snr_db: float = 25.0,
    ) -> SyntheticSignal:
        """4. Poorly damped oscillation (damping ratio < 3%)."""
        sig = self.generate_damped_ringdown(
            frequency=frequency,
            damping_ratio=damping_ratio,
            amplitude=amplitude,
            sampling_rate=sampling_rate,
            duration_sec=duration_sec,
            onset_time_sec=onset_time_sec,
            snr_db=snr_db,
        )
        sig.ground_truth.disturbance_type = "poorly_damped"
        return sig

    def generate_growing_oscillation(
        self,
        frequency: float = 0.70,
        growth_rate: float = 0.08,  # alpha < 0
        amplitude: float = 0.10,
        sampling_rate: float = 30.0,
        duration_sec: float = 20.0,
        onset_time_sec: float = 2.0,
        snr_db: float = 30.0,
    ) -> SyntheticSignal:
        """5. Growing / unstable oscillation (negative damping, exponential expansion)."""
        n = int(sampling_rate * duration_sec)
        t = np.arange(n) / sampling_rate
        phi = self.rng.uniform(0, 2 * np.pi)

        active = t >= onset_time_sec
        waveform = np.zeros(n)
        tau = t[active] - onset_time_sec
        waveform[active] = (
            amplitude * np.exp(growth_rate * tau) * np.cos(2 * np.pi * frequency * tau + phi)
        )

        noise = self._generate_ambient_noise(n, snr_db=snr_db, signal_power=amplitude**2)
        signal_out = waveform + noise

        omega = 2 * np.pi * frequency
        zeta = -growth_rate / np.sqrt(growth_rate**2 + omega**2)

        gt = DisturbanceGroundTruth(
            disturbance_type="growing_oscillation",
            sampling_rate=sampling_rate,
            duration_sec=duration_sec,
            num_samples=n,
            primary_frequency=frequency,
            damping_ratio=zeta,
            damping_coefficient=-growth_rate,
            amplitude=amplitude,
            snr_db=snr_db,
            onset_time_sec=onset_time_sec,
            event_duration_sec=duration_sec - onset_time_sec,
        )
        return SyntheticSignal(time=t, data=signal_out, ground_truth=gt)

    def generate_frequency_excursion(
        self,
        initial_rocof: float = -0.30,  # Hz/s initial Rate of Change of Frequency
        nadir_delta_f: float = -0.45,  # Maximum frequency drop (Hz)
        recovery_time_sec: float = 12.0,
        sampling_rate: float = 30.0,
        duration_sec: float = 20.0,
        onset_time_sec: float = 2.0,
        snr_db: float = 35.0,
    ) -> SyntheticSignal:
        """6. Frequency excursion following generation loss (RoCoF, nadir, primary governor recovery)."""
        n = int(sampling_rate * duration_sec)
        t = np.arange(n) / sampling_rate

        delta_f = np.zeros(n)
        active = t >= onset_time_sec
        tau = t[active] - onset_time_sec

        # Standard power system frequency response curve model:
        # delta_f(t) = nadir * (t / t_nadir) * exp(1 - t / t_nadir)
        t_nadir = max(1.0, abs(nadir_delta_f / initial_rocof))
        curve = nadir_delta_f * (tau / t_nadir) * np.exp(1.0 - (tau / t_nadir))
        # Add partial governor stabilization settling
        settled_delta = nadir_delta_f * 0.40
        decay = np.exp(-tau / recovery_time_sec)
        combined = curve * decay + settled_delta * (1.0 - decay)
        delta_f[active] = combined

        noise = self._generate_ambient_noise(n, snr_db=snr_db, signal_power=abs(nadir_delta_f) ** 2)
        signal_out = delta_f + noise

        gt = DisturbanceGroundTruth(
            disturbance_type="frequency_excursion",
            sampling_rate=sampling_rate,
            duration_sec=duration_sec,
            num_samples=n,
            primary_frequency=0.0,
            amplitude=abs(nadir_delta_f),
            snr_db=snr_db,
            onset_time_sec=onset_time_sec,
            rocof_hz_per_sec=initial_rocof,
            metadata={"nadir_delta_f": nadir_delta_f, "t_nadir": t_nadir},
        )
        return SyntheticSignal(time=t, data=signal_out, ground_truth=gt)

    def generate_frequency_drift(
        self,
        drift_rate: float = 0.05,  # Hz / second ramp
        sampling_rate: float = 30.0,
        duration_sec: float = 20.0,
        snr_db: float = 35.0,
    ) -> SyntheticSignal:
        """7. Slow continuous frequency drift / ramp."""
        n = int(sampling_rate * duration_sec)
        t = np.arange(n) / sampling_rate
        drift = drift_rate * t
        noise = self._generate_ambient_noise(
            n, snr_db=snr_db, signal_power=float(np.mean(drift**2))
        )
        signal_out = drift + noise

        gt = DisturbanceGroundTruth(
            disturbance_type="frequency_drift",
            sampling_rate=sampling_rate,
            duration_sec=duration_sec,
            num_samples=n,
            amplitude=float(np.max(np.abs(drift))),
            snr_db=snr_db,
            rocof_hz_per_sec=drift_rate,
        )
        return SyntheticSignal(time=t, data=signal_out, ground_truth=gt)

    def generate_generator_electromechanical(
        self,
        mode_type: Literal["inter_area", "local_plant"] = "inter_area",
        damping_ratio: float = 0.04,
        amplitude: float = 1.0,
        sampling_rate: float = 30.0,
        duration_sec: float = 20.0,
        snr_db: float = 25.0,
    ) -> SyntheticSignal:
        """8. Generator electromechanical oscillation (inter-area 0.2-0.8Hz or local plant 1.0-1.8Hz)."""
        freq = (
            self.rng.uniform(0.25, 0.75)
            if mode_type == "inter_area"
            else self.rng.uniform(1.10, 1.70)
        )
        sig = self.generate_damped_ringdown(
            frequency=freq,
            damping_ratio=damping_ratio,
            amplitude=amplitude,
            sampling_rate=sampling_rate,
            duration_sec=duration_sec,
            onset_time_sec=1.5,
            snr_db=snr_db,
        )
        sig.ground_truth.disturbance_type = f"generator_{mode_type}"
        sig.ground_truth.metadata["mode_type"] = mode_type
        return sig

    def generate_overlapping_events(
        self,
        f1: float = 0.35,  # e.g. Inter-area mode
        f2: float = 1.25,  # e.g. Forced oscillation or plant mode
        amp1: float = 1.0,
        amp2: float = 0.70,
        damping1: float = 0.04,
        sampling_rate: float = 30.0,
        duration_sec: float = 20.0,
        snr_db: float = 25.0,
    ) -> SyntheticSignal:
        """9. Multiple overlapping simultaneous oscillations."""
        n = int(sampling_rate * duration_sec)
        t = np.arange(n) / sampling_rate

        # Event 1: Ring-down at f1
        omega1 = 2 * np.pi * f1
        alpha1 = omega1 * (damping1 / np.sqrt(max(1e-4, 1.0 - damping1**2)))
        e1 = amp1 * np.exp(-alpha1 * t) * np.cos(omega1 * t)

        # Event 2: Sustained forced oscillation at f2 starting at t = 3s
        active2 = t >= 3.0
        e2 = np.zeros(n)
        e2[active2] = amp2 * np.cos(2 * np.pi * f2 * (t[active2] - 3.0))

        waveform = e1 + e2
        noise = self._generate_ambient_noise(
            n, snr_db=snr_db, signal_power=float(np.mean(waveform**2))
        )
        signal_out = waveform + noise

        gt = DisturbanceGroundTruth(
            disturbance_type="overlapping_events",
            sampling_rate=sampling_rate,
            duration_sec=duration_sec,
            num_samples=n,
            primary_frequency=f1,
            secondary_frequency=f2,
            damping_ratio=damping1,
            amplitude=amp1 + amp2,
            snr_db=snr_db,
            metadata={"components": [{"freq": f1, "amp": amp1}, {"freq": f2, "amp": amp2}]},
        )
        return SyntheticSignal(time=t, data=signal_out, ground_truth=gt)

    def generate_amplitude_modulated(
        self,
        carrier_freq: float = 1.10,
        mod_freq: float = 0.15,
        mod_index: float = 0.60,
        amplitude: float = 1.0,
        sampling_rate: float = 30.0,
        duration_sec: float = 20.0,
        snr_db: float = 25.0,
    ) -> SyntheticSignal:
        """10. Amplitude-modulated oscillation: A(t) = A_0 * (1 + m * cos(2*pi*f_m*t)) * cos(2*pi*f_c*t)."""
        n = int(sampling_rate * duration_sec)
        t = np.arange(n) / sampling_rate
        envelope = amplitude * (1.0 + mod_index * np.cos(2 * np.pi * mod_freq * t))
        carrier = np.cos(2 * np.pi * carrier_freq * t)
        waveform = envelope * carrier
        noise = self._generate_ambient_noise(n, snr_db=snr_db, signal_power=amplitude**2 / 2.0)
        signal_out = waveform + noise

        gt = DisturbanceGroundTruth(
            disturbance_type="amplitude_modulated",
            sampling_rate=sampling_rate,
            duration_sec=duration_sec,
            num_samples=n,
            primary_frequency=carrier_freq,
            amplitude=amplitude,
            snr_db=snr_db,
            modulation_frequency=mod_freq,
            modulation_index=mod_index,
        )
        return SyntheticSignal(time=t, data=signal_out, ground_truth=gt)

    def generate_frequency_modulated(
        self,
        carrier_freq: float = 1.0,
        mod_freq: float = 0.10,
        freq_deviation: float = 0.20,
        amplitude: float = 1.0,
        sampling_rate: float = 30.0,
        duration_sec: float = 20.0,
        snr_db: float = 25.0,
    ) -> SyntheticSignal:
        """11. Frequency-modulated oscillation: cos(2*pi*f_c*t + (delta_f / f_m) * sin(2*pi*f_m*t))."""
        n = int(sampling_rate * duration_sec)
        t = np.arange(n) / sampling_rate
        beta = freq_deviation / mod_freq
        waveform = amplitude * np.cos(
            2 * np.pi * carrier_freq * t + beta * np.sin(2 * np.pi * mod_freq * t)
        )
        noise = self._generate_ambient_noise(n, snr_db=snr_db, signal_power=amplitude**2 / 2.0)
        signal_out = waveform + noise

        gt = DisturbanceGroundTruth(
            disturbance_type="frequency_modulated",
            sampling_rate=sampling_rate,
            duration_sec=duration_sec,
            num_samples=n,
            primary_frequency=carrier_freq,
            amplitude=amplitude,
            snr_db=snr_db,
            modulation_frequency=mod_freq,
            metadata={"freq_deviation": freq_deviation},
        )
        return SyntheticSignal(time=t, data=signal_out, ground_truth=gt)

    def generate_transient_disturbance(
        self,
        impulse_time_sec: float = 3.0,
        impulse_amplitude: float = 3.0,
        ringdown_freq: float = 0.80,
        damping_ratio: float = 0.12,
        sampling_rate: float = 30.0,
        duration_sec: float = 20.0,
        snr_db: float = 30.0,
    ) -> SyntheticSignal:
        """12. Transient disturbance: Fault spike followed by rapid ringdown."""
        sig = self.generate_damped_ringdown(
            frequency=ringdown_freq,
            damping_ratio=damping_ratio,
            amplitude=impulse_amplitude,
            sampling_rate=sampling_rate,
            duration_sec=duration_sec,
            onset_time_sec=impulse_time_sec,
            snr_db=snr_db,
        )
        # Add sharp impulse at impulse_time
        idx = int(impulse_time_sec * sampling_rate)
        if idx < len(sig.data):
            sig.data[idx] += impulse_amplitude * 1.5

        sig.ground_truth.disturbance_type = "transient_disturbance"
        return sig
