"""
Heuristic Appliance Classifier — deterministic fallback for ML failure.

Purpose
-------
The deployed pipeline identifies appliances with ProtoNet + OpenMax. That path
can go dark in the field for reasons that have nothing to do with a bug:

  * weight files missing, truncated, or from an incompatible architecture
  * deep-learning framework import/allocation failure on a memory-constrained host
  * prototype registry empty (no classes enrolled yet)
  * inference raising on malformed input

`run_pipeline.py` already fails soft in those cases, returning "pending" or
"error" instead of crashing — and edge safety is entirely independent of ML
(ESP32 Core 0 opens the relay locally; FleetDiagnosticsMonitor is pure
threshold logic). So a model failure is never a *safety* failure.

What it does cost is the product: with no classifier, every event is
unattributed and the dashboard shows nothing useful. This module keeps
appliance identification alive in a coarser, fully deterministic form using
steady-state power band, transient overshoot, duty cycle and shape — the
classic pre-ML NILM feature set. Zero external ML libraries, no weights, no training.

Accuracy is materially lower than the trained ProtoNet; results are labelled
`degraded=True` and carry low confidence so the UI can badge them and so they
never silently masquerade as model output.

Usage:
    clf = HeuristicApplianceClassifier()
    result = clf.classify(window_128_samples)
    result.appliance    # 'kettle'
    result.confidence   # 0.0 - MAX_HEURISTIC_CONFIDENCE
    result.degraded     # always True
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Heuristic output is never allowed to reach the ProtoNet confidence gate
# (0.90) — a rule-based guess must not be actionable as if it were calibrated.
MAX_HEURISTIC_CONFIDENCE = 0.75
ON_THRESHOLD_W = 20.0
UNKNOWN = "unknown"


@dataclass
class HeuristicResult:
    """Outcome of a rule-based classification."""
    appliance: str
    confidence: float
    degraded: bool = True
    source: str = "heuristic_fallback"
    features: Dict[str, float] = field(default_factory=dict)
    runner_up: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "appliance": self.appliance,
            "confidence": round(self.confidence, 4),
            "degraded": self.degraded,
            "source": self.source,
            "runner_up": self.runner_up,
            "features": {k: round(v, 3) for k, v in self.features.items()},
        }


@dataclass
class ApplianceRule:
    """
    Power-signature envelope for one appliance class.

    steady_w      : (min, max) plausible steady-state operating watts
    peak_w        : (min, max) plausible peak watts
    duty          : (min, max) fraction of the window spent above threshold
    overshoot     : (min, max) peak / steady ratio — inrush signature
    volatility    : (min, max) std/mean over the on-portion — cycling vs flat
    """
    name: str
    steady_w: Tuple[float, float]
    peak_w: Tuple[float, float]
    duty: Tuple[float, float] = (0.05, 1.0)
    overshoot: Tuple[float, float] = (1.0, 12.0)
    volatility: Tuple[float, float] = (0.0, 3.0)
    weight: float = 1.0


# ── Feature centroids fitted on real UK-DALE + REDD windows ──────────────────
# Hand-drawn power bands overlap so heavily that band-matching alone scored only
# 0.22 accuracy on real data — barely above the 0.11 chance rate for 9 classes.
# So each class is also summarised as a centroid in a robust feature space and
# classified by nearest scaled distance. Still fully deterministic, still needs
# neither external ML libraries nor a weights file.
#
# Features: [log10(steady_w), log10(peak_w), duty, log10(overshoot), volatility]
# Regenerate with: python scripts/fit_heuristic_centroids.py
FEATURE_NAMES = ("log_steady", "log_peak", "duty", "log_overshoot", "volatility")

# Per-feature robust scale, used to normalise distances across features.
FEATURE_SCALES: Tuple[float, ...] = (0.2063, 0.2233, 1.0000, 0.0272, 0.0474)

# {class_name: centroid vector} — written by scripts/fit_heuristic_centroids.py.
# Empty means "not fitted yet"; the classifier then falls back to band rules
# alone, which still works but scores materially worse.
CLASS_CENTROIDS: Dict[str, Tuple[float, ...]] = {
    'dishwasher': (2.2810, 2.3802, 0.7500, 0.0132, 0.0265),
    'fridge': (2.0934, 2.1732, 0.7500, 0.0367, 0.0441),
    'hvac': (2.0043, 2.0374, 0.7500, 0.0445, 0.0426),
    'kettle': (3.4600, 3.4658, 0.7500, 0.0058, 0.0083),
    'laptop': (1.9823, 2.0645, 0.7500, 0.0289, 0.0504),
    'microwave': (3.1858, 3.1976, 0.6328, 0.0056, 0.3695),
    'oven': (2.9899, 3.1532, 0.7500, 0.0104, 0.0179),
    'tv': (1.9685, 1.9868, 0.7500, 0.0382, 0.1267),
    'washing_machine': (2.8228, 2.9668, 0.7500, 0.0612, 0.1696),
}


# Envelopes derived from the measured UK-DALE / REDD windows extracted by
# data/nilmtk_reader.py (per-class mean and p95), widened for tolerance.
DEFAULT_RULES: List[ApplianceRule] = [
    # Consumer electronics (demo profile & low-power appliances)
    ApplianceRule("phone_charger",    steady_w=(5, 125),     peak_w=(8, 145),
                  duty=(0.10, 1.0), overshoot=(1.0, 3.0),  volatility=(0.0, 0.8)),
    ApplianceRule("router",           steady_w=(5, 35),      peak_w=(8, 45),
                  duty=(0.40, 1.0), overshoot=(1.0, 2.0),  volatility=(0.0, 0.4)),
    ApplianceRule("monitor",          steady_w=(15, 80),     peak_w=(20, 100),
                  duty=(0.20, 1.0), overshoot=(1.0, 2.5),  volatility=(0.0, 0.5)),
    ApplianceRule("laptop",           steady_w=(15, 220),    peak_w=(20, 450),
                  duty=(0.20, 1.0), overshoot=(1.0, 4.0),  volatility=(0.0, 0.9)),
    ApplianceRule("desktop_computer", steady_w=(50, 450),    peak_w=(70, 600),
                  duty=(0.20, 1.0), overshoot=(1.0, 3.5),  volatility=(0.0, 0.9)),
    ApplianceRule("projector",        steady_w=(30, 450),    peak_w=(40, 550),
                  duty=(0.20, 1.0), overshoot=(1.0, 3.0),  volatility=(0.0, 0.7)),
    ApplianceRule("tv",               steady_w=(30, 250),    peak_w=(40, 600),
                  duty=(0.25, 1.0), overshoot=(1.0, 3.5),  volatility=(0.0, 0.8)),
    # Household appliances (standard profile)
    ApplianceRule("fridge",           steady_w=(50, 300),    peak_w=(80, 1200),
                  duty=(0.10, 1.0), overshoot=(1.2, 8.0),  volatility=(0.05, 1.4)),
    ApplianceRule("hvac",             steady_w=(200, 3000),  peak_w=(300, 5500),
                  duty=(0.20, 1.0), overshoot=(1.0, 4.0),  volatility=(0.0, 1.2)),
    ApplianceRule("microwave",        steady_w=(600, 1700),  peak_w=(700, 2200),
                  duty=(0.10, 0.85), overshoot=(1.0, 2.5), volatility=(0.0, 1.0)),
    ApplianceRule("dishwasher",       steady_w=(150, 2400),  peak_w=(400, 3200),
                  duty=(0.20, 1.0), overshoot=(1.1, 6.0),  volatility=(0.15, 2.0)),
    ApplianceRule("washing_machine",  steady_w=(100, 2200),  peak_w=(300, 3800),
                  duty=(0.15, 1.0), overshoot=(1.2, 9.0),  volatility=(0.20, 2.5)),
    ApplianceRule("oven",             steady_w=(800, 3000),  peak_w=(1000, 3300),
                  duty=(0.25, 1.0), overshoot=(1.0, 2.2),  volatility=(0.0, 1.0)),
    ApplianceRule("kettle",           steady_w=(1600, 3300), peak_w=(1800, 4200),
                  duty=(0.08, 0.80), overshoot=(1.0, 2.0), volatility=(0.0, 0.9)),
    ApplianceRule("ev_charger",       steady_w=(2800, 7500), peak_w=(3000, 8000),
                  duty=(0.50, 1.0), overshoot=(1.0, 1.6),  volatility=(0.0, 0.4)),
]


class HeuristicApplianceClassifier:
    """
    Deterministic power-signature classifier used when ProtoNet is unavailable.

    Scores a window against each appliance envelope and returns the best match.
    Thread-safe and allocation-light: safe to call on the MQTT ingest path.
    """

    def __init__(self, rules: Optional[Sequence[ApplianceRule]] = None,
                 on_threshold_w: float = ON_THRESHOLD_W,
                 max_confidence: float = MAX_HEURISTIC_CONFIDENCE,
                 centroids: Optional[Dict[str, Sequence[float]]] = None,
                 feature_scales: Optional[Sequence[float]] = None):
        self.rules = list(rules) if rules else list(DEFAULT_RULES)
        self.on_threshold_w = on_threshold_w
        self.max_confidence = max_confidence

        src = centroids if centroids is not None else CLASS_CENTROIDS
        self.centroids = {k: np.asarray(v, dtype=np.float64)
                          for k, v in (src or {}).items()}
        self.feature_scales = np.asarray(
            feature_scales if feature_scales is not None else FEATURE_SCALES,
            dtype=np.float64)
        self.feature_scales = np.where(self.feature_scales > 1e-9,
                                       self.feature_scales, 1.0)

    # ── Feature extraction ───────────────────────────────────────────────────
    def feature_vector(self, f: Dict[str, float]) -> np.ndarray:
        """Pack the summary features into the centroid feature space."""
        eps = 1e-6
        return np.array([
            np.log10(max(f.get("steady_w", 0.0), eps)),
            np.log10(max(f.get("peak_w", 0.0), eps)),
            f.get("duty", 0.0),
            np.log10(max(f.get("overshoot", 1.0), eps)),
            f.get("volatility", 0.0),
        ], dtype=np.float64)

    # ── Feature extraction ───────────────────────────────────────────────────
    def extract_features(self, window: Sequence[float]) -> Dict[str, float]:
        """Summarise a power window into the features the rules score against."""
        w = np.asarray(window, dtype=np.float64).ravel()
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        if w.size == 0:
            return {}

        on = w[w > self.on_threshold_w]
        duty = float(on.size) / float(w.size)
        if on.size == 0:
            return {"peak_w": float(w.max()), "steady_w": 0.0, "duty": 0.0,
                    "overshoot": 1.0, "volatility": 0.0, "mean_w": float(w.mean())}

        steady = float(np.median(on))
        peak = float(w.max())
        overshoot = peak / steady if steady > 1e-6 else 1.0
        volatility = float(on.std() / on.mean()) if on.mean() > 1e-6 else 0.0

        return {
            "peak_w": peak,
            "steady_w": steady,
            "duty": duty,
            "overshoot": overshoot,
            "volatility": volatility,
            "mean_w": float(w.mean()),
        }

    # ── Scoring ──────────────────────────────────────────────────────────────
    @staticmethod
    def _band_score(value: float, lo: float, hi: float) -> float:
        """
        1.0 inside [lo, hi], decaying smoothly outside so a near-miss still
        scores above an unrelated class instead of falling straight to zero.
        """
        if lo <= value <= hi:
            return 1.0
        span = max(hi - lo, 1e-6)
        dist = (lo - value) if value < lo else (value - hi)
        return float(max(0.0, 1.0 - (dist / span)))

    def _score_rule(self, rule: ApplianceRule, f: Dict[str, float]) -> float:
        # Power band dominates: it is the most reliable discriminator at 1 Hz.
        terms = [
            (3.0, self._band_score(f["steady_w"], *rule.steady_w)),
            (2.0, self._band_score(f["peak_w"], *rule.peak_w)),
            (1.0, self._band_score(f["duty"], *rule.duty)),
            (1.0, self._band_score(f["overshoot"], *rule.overshoot)),
            (1.0, self._band_score(f["volatility"], *rule.volatility)),
        ]
        total_w = sum(w for w, _ in terms)
        return rule.weight * sum(w * s for w, s in terms) / total_w

    # ── Public API ───────────────────────────────────────────────────────────
    def classify(self, window: Sequence[float]) -> HeuristicResult:
        """Classify a power window. Never raises."""
        try:
            f = self.extract_features(window)
        except Exception as e:                      # defensive: last line of defence
            logger.warning(f"Heuristic feature extraction failed: {e}")
            return HeuristicResult(UNKNOWN, 0.0)

        if not f or f.get("peak_w", 0.0) <= 0.0:
            return HeuristicResult(UNKNOWN, 0.0, features=f)

        # Preferred path: nearest fitted centroid in the robust feature space.
        if self.centroids:
            return self._classify_by_centroid(f)

        if f.get("steady_w", 0.0) <= 0.0:
            return HeuristicResult(UNKNOWN, 0.0, features=f)

        return self._classify_by_rules(f)

    def _classify_by_centroid(self, f: Dict[str, float]) -> HeuristicResult:
        v = self.feature_vector(f)
        names, dists = [], []
        for name, c in self.centroids.items():
            if c.shape != v.shape:
                continue
            d = float(np.linalg.norm((v - c) / self.feature_scales))
            names.append(name)
            dists.append(d)

        if not names:
            return self._classify_by_rules(f)

        order = np.argsort(dists)
        best, second = order[0], (order[1] if len(order) > 1 else None)
        best_name = names[best]
        runner_up = names[second] if second is not None else None

        # Convert distance to confidence: near the centroid is confident, and a
        # clear margin over the runner-up raises it further.
        d0 = dists[best]
        d1 = dists[second] if second is not None else d0 * 2.0
        closeness = 1.0 / (1.0 + d0)
        margin = (d1 - d0) / max(d1 + d0, 1e-6)
        confidence = min(self.max_confidence,
                         self.max_confidence * closeness * (0.55 + 0.45 * min(1.0, margin * 3)))

        return HeuristicResult(appliance=best_name, confidence=float(confidence),
                               features=f, runner_up=runner_up)

    def _classify_by_rules(self, f: Dict[str, float]) -> HeuristicResult:
        scored = sorted(((self._score_rule(r, f), r.name) for r in self.rules),
                        reverse=True)
        best_score, best_name = scored[0]
        runner_up = scored[1][1] if len(scored) > 1 else None

        if best_score <= 0.0:
            return HeuristicResult(UNKNOWN, 0.0, features=f, runner_up=runner_up)

        # Margin over the runner-up damps confidence when classes overlap
        # (e.g. oven vs kettle both sit high on the power band).
        margin = best_score - (scored[1][0] if len(scored) > 1 else 0.0)
        confidence = min(self.max_confidence,
                         best_score * self.max_confidence * (0.6 + 0.4 * min(1.0, margin * 4)))

        return HeuristicResult(
            appliance=best_name,
            confidence=float(confidence),
            features=f,
            runner_up=runner_up,
        )

    def classify_batch(self, windows: Sequence[Sequence[float]]
                       ) -> List[HeuristicResult]:
        return [self.classify(w) for w in windows]
