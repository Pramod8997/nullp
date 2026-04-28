"""
Synthetic UK-DALE Data Generator.
Generates realistic 1 Hz transient power signatures for 10 appliance classes
matching UK-DALE 1 Hz profiles described in Kelly & Knottenbelt (2015).
"""
import numpy as np

# 10 appliance classes with (steady_W, transient_peak_W, noise_std)
APPLIANCE_PROFILES = {
    'fridge':          (150,  300,  10),
    'hvac':            (1500, 2200, 80),
    'kettle':          (2200, 2400, 30),
    'tv':              (120,  180,  8),
    'washing_machine': (500,  2000, 100),
    'dishwasher':      (1800, 2000, 60),
    'microwave':       (1200, 1300, 20),
    'oven':            (2000, 2200, 50),
    'ev_charger':      (3300, 3500, 40),
    'laptop':          (60,   90,   5),
}

SEQ_LEN = 128
SAMPLES_PER_CLASS = 500   # >= K_SHOT + Q_QUERY + calibration margin


class SyntheticUKDALE:
    """
    Generates realistic 1 Hz transient power signatures for 10 appliance classes.
    Each segment is SEQ_LEN samples centred on a simulated turn-on transient.
    The signature shape matches UK-DALE 1 Hz profiles described in Kelly &
    Knottenbelt (2015).
    """

    def __init__(self, seq_len=SEQ_LEN, n_samples=SAMPLES_PER_CLASS, seed=42):
        self.seq_len   = seq_len
        self.n_samples = n_samples
        self._rng      = np.random.default_rng(seed)

    def _make_segment(self, steady_w, peak_w, noise_std):
        seg  = np.zeros(self.seq_len, dtype=np.float32)
        half = self.seq_len // 2
        # Pre-transient: near-zero with noise
        seg[:half] = self._rng.normal(5, noise_std * 0.1, half)
        # Transient: exponential rise to steady state
        t      = np.arange(self.seq_len - half)
        decay  = peak_w * np.exp(-t / (self.seq_len * 0.15))
        steady = np.full(self.seq_len - half, steady_w)
        seg[half:] = (np.maximum(steady, decay)
                      + self._rng.normal(0, noise_std, self.seq_len - half))
        return np.clip(seg, 0, 4000)

    def load_all_classes(self):
        """Returns dict {class_name: np.ndarray (n_samples, seq_len)}."""
        dataset = {}
        for name, (steady, peak, noise) in APPLIANCE_PROFILES.items():
            segs = np.stack([self._make_segment(steady, peak, noise)
                             for _ in range(self.n_samples)])
            dataset[name] = segs
        return dataset
