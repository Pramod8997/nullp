"""
Synthetic UK-DALE Data Generator — v2 (Production-Ready).

Generates realistic 1 Hz transient power signatures for 10 appliance classes
matching UK-DALE 1 Hz profiles described in Kelly & Knottenbelt (2015).

v2 improvements (addressing PRODUCTION_AUDIT_ARCHIVE §2.1 — Overfitting):
  - Phase jitter: transient onset varies ±30% of window
  - Harmonic distortion: adds mains harmonics (3rd, 5th, 7th) to motor loads
  - Variable steady-state: ±20% variation in operating power
  - Brownout noise: random voltage sag events that reduce amplitude
  - Multi-state appliances: washing machines, dishwashers cycle through modes
  - Cross-talk noise: background load from other appliances
  - Sensor noise: models ESP32 ADC quantization and thermal noise
"""
import numpy as np

# 10 appliance classes with (steady_W, transient_peak_W, noise_std, has_harmonics, is_multi_state)
APPLIANCE_PROFILES = {
    'fridge':          (150,  300,  10,  False, False),
    'hvac':            (1500, 2200, 80,  True,  False),
    'kettle':          (2200, 2400, 30,  False, False),
    'tv':              (120,  180,  8,   False, False),
    'washing_machine': (500,  2000, 100, True,  True),
    'dishwasher':      (1800, 2000, 60,  True,  True),
    'microwave':       (1200, 1300, 20,  False, False),
    'oven':            (2000, 2200, 50,  False, False),
    'ev_charger':      (3300, 3500, 40,  False, False),
    'laptop':          (60,   90,   5,   False, False),
}

SEQ_LEN = 128
SAMPLES_PER_CLASS = 500   # >= K_SHOT + Q_QUERY + calibration margin


class SyntheticUKDALE:
    """
    Generates realistic 1 Hz transient power signatures for 10 appliance classes.
    Each segment is SEQ_LEN samples centred on a simulated turn-on transient.

    v2: Incorporates realistic variance to prevent the ProtoNet from
    memorizing fixed synthetic profiles (PRODUCTION_AUDIT §2.1 remediation).
    """

    def __init__(self, seq_len=SEQ_LEN, n_samples=SAMPLES_PER_CLASS, seed=42):
        self.seq_len   = seq_len
        self.n_samples = n_samples
        self._rng      = np.random.default_rng(seed)

    def _add_harmonics(self, seg, amplitude_frac=0.05):
        """Add mains harmonics (3rd, 5th, 7th) typical of motor-driven loads."""
        t = np.arange(len(seg), dtype=np.float32)
        for harmonic in [3, 5, 7]:
            # Random phase offset per sample for realism
            phase = self._rng.uniform(0, 2 * np.pi)
            freq = harmonic * (50.0 / len(seg))  # Relative to 50Hz mains
            amplitude = amplitude_frac * np.mean(np.abs(seg)) / harmonic
            seg += amplitude * np.sin(2 * np.pi * freq * t + phase)
        return seg

    def _add_brownout(self, seg, probability=0.15):
        """Simulate random voltage sag events (brownouts) that reduce power."""
        if self._rng.random() < probability:
            sag_start = self._rng.integers(0, max(1, len(seg) - 10))
            sag_len = self._rng.integers(5, min(20, len(seg) - sag_start))
            sag_factor = self._rng.uniform(0.7, 0.9)  # 10-30% voltage reduction
            seg[sag_start:sag_start + sag_len] *= sag_factor
        return seg

    def _add_crosstalk(self, seg, max_watts=15.0):
        """Add background noise from other appliances on the same mains."""
        # Slowly varying background (other devices turning on/off)
        bg_freq = self._rng.uniform(0.01, 0.1)
        bg_phase = self._rng.uniform(0, 2 * np.pi)
        bg_amp = self._rng.uniform(3.0, max_watts)
        t = np.arange(len(seg), dtype=np.float32)
        seg += bg_amp * np.sin(2 * np.pi * bg_freq * t + bg_phase)
        # Add step-change noise (another appliance switching)
        if self._rng.random() < 0.15:
            step_pos = self._rng.integers(0, len(seg))
            step_val = self._rng.uniform(-10, 20)
            seg[step_pos:] += step_val
        return seg

    def _add_sensor_noise(self, seg):
        """Model ESP32 ADC quantization noise and thermal drift."""
        # ADC quantization: 12-bit over 3.3V range (resolution ~3W at 230V/30A)
        quant_step = 3.0
        seg = np.round(seg / quant_step) * quant_step
        # Thermal drift: slow baseline shift
        drift = self._rng.uniform(-2.0, 2.0)
        t = np.arange(len(seg), dtype=np.float32) / len(seg)
        seg += drift * t
        return seg

    def _make_multi_state_segment(self, steady_w, peak_w, noise_std):
        """Generate multi-state appliance (washing machine, dishwasher)."""
        seg = np.zeros(self.seq_len, dtype=np.float32)

        # Variable onset position (phase jitter: ±30%)
        onset = int(self.seq_len * self._rng.uniform(0.2, 0.5))

        # Pre-transient: baseline noise
        baseline = self._rng.uniform(2.0, 8.0)
        seg[:onset] = self._rng.normal(baseline, noise_std * 0.1, onset)

        # Multi-state: 2-4 distinct power levels after onset
        remaining = self.seq_len - onset
        n_states = self._rng.integers(2, 5)
        state_lengths = self._rng.dirichlet(np.ones(n_states)) * remaining
        state_lengths = np.round(state_lengths).astype(int)
        # Fix rounding to match exactly
        state_lengths[-1] = remaining - state_lengths[:-1].sum()

        pos = onset
        for i, slen in enumerate(state_lengths):
            if slen <= 0:
                continue
            # Clamp to actual remaining space to prevent shape mismatch
            actual_len = min(int(slen), self.seq_len - pos)
            if actual_len <= 0:
                break
            # Each state has a different power level
            if i == 0:
                # Initial surge (transient)
                state_power = self._rng.uniform(peak_w * 0.8, peak_w * 1.1)
            else:
                # Operating states: variable power
                state_power = self._rng.uniform(steady_w * 0.3, steady_w * 1.2)

            seg[pos:pos + actual_len] = self._rng.normal(state_power, noise_std, actual_len)
            pos += actual_len

        return seg

    def _make_segment(self, steady_w, peak_w, noise_std,
                      has_harmonics=False, is_multi_state=False):
        """Generate a single appliance power signature with realistic variance."""

        if is_multi_state:
            seg = self._make_multi_state_segment(steady_w, peak_w, noise_std)
        else:
            seg = np.zeros(self.seq_len, dtype=np.float32)

            # Phase jitter: onset varies ±30% from center
            onset = int(self.seq_len * self._rng.uniform(0.2, 0.5))

            # Variable steady-state power: ±20% of nominal
            actual_steady = steady_w * self._rng.uniform(0.85, 1.15)
            actual_peak = peak_w * self._rng.uniform(0.9, 1.1)

            # Variable decay constant
            decay_tau = self.seq_len * self._rng.uniform(0.08, 0.25)

            # Pre-transient: baseline noise (not always near-zero)
            baseline = self._rng.uniform(2.0, 15.0)
            seg[:onset] = self._rng.normal(baseline, noise_std * 0.2, onset)

            # Transient: exponential rise to steady state
            post_len = self.seq_len - onset
            t = np.arange(post_len, dtype=np.float32)
            decay = actual_peak * np.exp(-t / decay_tau)
            steady = np.full(post_len, actual_steady)
            seg[onset:] = (np.maximum(steady, decay)
                          + self._rng.normal(0, noise_std, post_len))

        # Apply augmentations
        if has_harmonics:
            seg = self._add_harmonics(seg)

        seg = self._add_brownout(seg)
        seg = self._add_crosstalk(seg)
        seg = self._add_sensor_noise(seg)

        return np.clip(seg, 0, 4000).astype(np.float32)

    def load_all_classes(self):
        """Returns dict {class_name: np.ndarray (n_samples, seq_len)}."""
        dataset = {}
        for name, (steady, peak, noise, harmonics, multi) in APPLIANCE_PROFILES.items():
            segs = np.stack([self._make_segment(steady, peak, noise,
                                                harmonics, multi)
                             for _ in range(self.n_samples)])
            dataset[name] = segs
        return dataset
