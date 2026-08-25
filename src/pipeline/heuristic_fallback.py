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

# Nearest-centroid distance beyond which the window is called UNKNOWN rather
# than assigned to the closest class.
#
# Without this the centroid path was a pure argmin: it returned the nearest
# class at ANY distance, so a load no class describes still came back with a
# label and a non-trivial confidence. Measured before the gate: a 65 W laptop
# window classified as `hvac` (band 200-3000 W) and a 55 W monitor as `laptop`.
#
# Units are scaled-feature sigmas (the distance is divided by FEATURE_SCALES),
# so 6.0 means "six robust within-class deviations from every known centroid".
# Chosen from the real UK-DALE/REDD window distances: in-distribution windows
# sit below ~3, so 6.0 rejects genuine novelty without discarding real matches.
CENTROID_REJECT_RADIUS = 6.0

# Fractional slack applied to a rule's power envelope by `plausible_classes`.
# Envelopes in DEFAULT_RULES are already widened for tolerance; this is only to
# keep a window that sits exactly on a boundary from being excluded by float
# noise or by Indian mains voltage swing (P scales as V^2, so +-6% mains is
# ~+-12% power).
ENVELOPE_SLACK = 0.15


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
#
# 🔴 This dict is MERGED across profiles, and must stay that way. It holds the
# household class set (fitted from the default window cache) AND the
# consumer-electronics set (fitted from the `_demo` cache) side by side, because
# there is more than one deployment profile:
#
#   config/config.yaml           -> household, 3.5 kW ceiling
#   config/config.demo.yaml      -> simulated consumer-electronics fleet, 600 W
#   config/config.hardware.yaml  -> the physical rig, 250 W, laptop + charger
#
# When only the household classes were present, the centroid path -- which
# `classify()` prefers over the band rules -- could not emit `phone_charger` at
# all, so the physical rig's second appliance was unclassifiable in the very
# degraded mode that HARDWARE_FINAL_SPEC.md §7 requires to work (protonet.pt
# deleted). fit_heuristic_centroids.py therefore merges its results in rather
# than overwriting the dict. Restrict what a given deployment may emit with
# `allowed_classes=`, not by deleting centroids.
CLASS_CENTROIDS: Dict[str, Tuple[float, ...]] = {
    'desktop_computer': (1.8633, 1.9777, 0.7500, 0.1098, 0.1392),
    'dishwasher': (2.2810, 2.3802, 0.7500, 0.0132, 0.0265),
    'fridge': (2.0934, 2.1732, 0.7500, 0.0367, 0.0441),
    'hvac': (2.0043, 2.0374, 0.7500, 0.0445, 0.0426),
    'kettle': (3.4600, 3.4658, 0.7500, 0.0058, 0.0083),
    'laptop': (1.9823, 2.0645, 0.7500, 0.0289, 0.0504),
    'microwave': (3.1858, 3.1976, 0.6328, 0.0056, 0.3695),
    'monitor': (1.7709, 1.7853, 0.7500, 0.0071, 0.0570),
    'oven': (2.9899, 3.1532, 0.7500, 0.0104, 0.0179),
    'projector': (2.3181, 2.4249, 0.7500, 0.0643, 0.3530),
    'tv': (1.9685, 1.9868, 0.7500, 0.0382, 0.1267),
    'washing_machine': (2.8228, 2.9668, 0.7500, 0.0612, 0.1696),
}

# 🔴 Why `phone_charger` and `router` have no centroid, deliberately.
#
# `fit_heuristic_centroids.py --cache-tag _demo` does fit them, but from the
# only real windows that exist for those classes: UK-DALE b1/m27,32,34 (charger)
# and b1/m18 + b2/m18 (router). Both are sub-10 W loads — the fitted centroids
# come out at log_steady 0.699 (5.0 W) and 0.778 (6.0 W).
#
# `extract_features` only counts samples above `on_threshold_w` (20 W default).
# A 5 W window therefore yields `on.size == 0` -> `steady_w = 0.0`, and
# `classify()` returns UNKNOWN before the centroid path is ever consulted. The
# centroids would be permanently unreachable, so storing them would only make
# the coverage look better than it is.
#
# This is the documented division of labour, not an oversight: CLAUDE.md §1.7
# routes 3-10 W trickle/standby loads to `PhantomTracker`, and only 18-120 W
# fast-charge / USB-PD loads cross `TRANSIENT_THRESHOLD_W` into NILM. There is
# no real USB-PD-era charger data in UK-DALE (2013 vintage) at all — the highest
# `phone_charger` window in the cache peaks at 33 W and the median is 5 W. A
# modern 45-120 W charger has to be enrolled from the user's own hardware
# through the LABEL_REQUEST loop; no fitted constant can stand in for it.


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
    # Purely resistive filament lamp — the physical rig's calibration reference
    # and the ballast that makes the 312 W CRITICAL trip reachable
    # (HARDWARE_FINAL_SPEC.md BOM item 19, D8, bring-up stages 5 and 7).
    #
    # It is separable from a laptop at the same wattage by shape alone, not by
    # power: PF = 1, a dead-flat steady state, and no SMPS soft-start, so both
    # overshoot and volatility sit far tighter than any switch-mode load. The
    # cold-filament inrush is ~10x for a few milliseconds and is invisible at
    # the PZEM's ~500 ms register refresh, which is why overshoot stays ~1.0.
    # Band spans 60 W and 100 W lamps plus Indian mains swing (P scales as V^2).
    ApplianceRule("incandescent_lamp", steady_w=(45, 140),   peak_w=(50, 165),
                  duty=(0.30, 1.0), overshoot=(1.0, 1.25), volatility=(0.0, 0.10)),
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


# ── Physical plausibility gate ───────────────────────────────────────────────

def plausible_classes(features: Dict[str, float],
                      rules: Optional[Sequence[ApplianceRule]] = None,
                      slack: float = ENVELOPE_SLACK) -> set:
    """
    Return the classes whose measured power envelope can contain this window.

    Why this exists as a separate, model-free gate
    ----------------------------------------------
    The learned embedding is **power-scale-blind**. Measured on the shipped
    `backend/models/weights_demo` artefacts, squared prototype distances for
    genuinely novel loads land inside the known-class range rather than outside
    it, so neither the OpenMax Weibull tail nor a plain distance threshold can
    separate known from novel:

        known windows (real UK-DALE)  median d2 = 0.27, p99 = 5.10
        kettle    2000 W  -> `tv`               d2 = 5.89
        oven      2500 W  -> `tv`               d2 = 10.39
        washing    500 W  -> `router`           d2 = 1.02   (router is a 5 W load)
        heater     800 W  -> `phone_charger`    d2 = 1.35

    A 500 W load landing 1.02 from a 5 W router's prototype is not a threshold
    that needs tuning — the distance carries no scale information to threshold.
    Absolute watts, however, are measured directly by the PZEM and are the one
    thing about a load that cannot be confused. So the reject decision is made
    here, on physics, and the network is only ever allowed to choose *among*
    classes that the wattage already permits.

    Args:
        features: output of `HeuristicApplianceClassifier.extract_features`.
        rules:    envelopes to test. Defaults to DEFAULT_RULES.
        slack:    fractional widening of each band (see ENVELOPE_SLACK).

    Returns:
        Set of class names. Empty means no known class can draw this power —
        the caller should report UNKNOWN and request a label.
    """
    src = list(rules) if rules is not None else DEFAULT_RULES
    steady = float(features.get("steady_w", 0.0) or 0.0)
    peak = float(features.get("peak_w", 0.0) or 0.0)
    if steady <= 0.0 and peak <= 0.0:
        return set()

    out = set()
    for r in src:
        s_lo, s_hi = r.steady_w[0] * (1.0 - slack), r.steady_w[1] * (1.0 + slack)
        p_lo, p_hi = r.peak_w[0] * (1.0 - slack), r.peak_w[1] * (1.0 + slack)
        # steady_w is the discriminator; peak only has to not contradict it.
        # A window captured mid-transient can peak well above the rule's band
        # without the class being wrong, so peak is tested one-sided (a load
        # cannot peak *below* its own floor).
        if s_lo <= steady <= s_hi and peak >= p_lo * 0.5:
            out.add(r.name)
        elif steady <= 0.0 and p_lo <= peak <= p_hi:
            out.add(r.name)
    return out


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
                 feature_scales: Optional[Sequence[float]] = None,
                 allowed_classes: Optional[Sequence[str]] = None,
                 reject_radius: float = CENTROID_REJECT_RADIUS):
        """
        Args:
            allowed_classes: Restrict output to this class set, e.g. the
                `appliances:` list from config/config.hardware.yaml. Classes
                outside it are dropped from BOTH the centroid and the band-rule
                path before scoring, so a class no socket on the rig can present
                is never the answer. None (default) allows every class.

                Filtering at construction rather than post-hoc matters: the
                runner-up and the confidence margin are both computed over the
                surviving classes, so a suppressed class cannot silently damp
                the confidence of the one that is actually plugged in.
            reject_radius: Scaled-feature distance beyond which the nearest
                centroid is not trusted and the window is reported UNKNOWN
                rather than assigned. See CENTROID_REJECT_RADIUS.
        """
        self.allowed_classes = set(allowed_classes) if allowed_classes else None
        self.reject_radius = float(reject_radius)

        rule_src = list(rules) if rules else list(DEFAULT_RULES)
        if self.allowed_classes is not None:
            kept = [r for r in rule_src if r.name in self.allowed_classes]
            if kept:
                rule_src = kept
            else:
                logger.warning(
                    "allowed_classes=%s matched no rule; keeping the full rule "
                    "set rather than leaving the classifier with nothing to "
                    "score.", sorted(self.allowed_classes))
        self.rules = rule_src

        self.on_threshold_w = on_threshold_w
        self.max_confidence = max_confidence

        src = centroids if centroids is not None else CLASS_CENTROIDS
        self.centroids = {k: np.asarray(v, dtype=np.float64)
                          for k, v in (src or {}).items()}
        if self.allowed_classes is not None:
            kept_c = {k: v for k, v in self.centroids.items()
                      if k in self.allowed_classes}
            # An empty result means this profile's classes were never fitted.
            # Keeping the unfiltered centroids would let the preferred centroid
            # path answer with a class the profile forbids, so drop to the band
            # rules (already filtered) instead.
            self.centroids = kept_c

        # Centroids for classes that have no band rule. The physical gate is
        # built from the rule set, so it has nothing to say about these — they
        # must bypass it rather than be vetoed by it. Empty for the default
        # centroid set, non-empty only when a caller supplies its own.
        self._extra_centroids = {k for k in self.centroids
                                 if k not in {r.name for r in self.rules}}

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

        # Physical gate first: only classes whose measured power envelope can
        # contain this window are eligible. A load outside every envelope is
        # novel by construction and no amount of feature distance should be
        # allowed to label it.
        eligible = plausible_classes(f, self.rules)
        if not eligible and not self._extra_centroids:
            return HeuristicResult(UNKNOWN, 0.0, features=f)

        # Preferred path: nearest fitted centroid in the robust feature space.
        if self.centroids:
            result = self._classify_by_centroid(f, eligible)
            if result is not None:
                return result
            # No eligible class has a fitted centroid, or the nearest one was
            # beyond the reject radius — fall through to the band rules, which
            # cover classes the centroid fit could not reach (see the
            # phone_charger / router note above CLASS_CENTROIDS).

        if f.get("steady_w", 0.0) <= 0.0:
            return HeuristicResult(UNKNOWN, 0.0, features=f)

        if not eligible:
            return HeuristicResult(UNKNOWN, 0.0, features=f)

        return self._classify_by_rules(f, eligible)

    def _classify_by_centroid(self, f: Dict[str, float],
                              eligible: Optional[set] = None
                              ) -> Optional[HeuristicResult]:
        """
        Nearest-centroid classification restricted to `eligible` classes.

        A centroid whose class has no band rule is exempt from the restriction.
        `eligible` is derived from the rule set, so it carries no opinion about
        such a class — vetoing it would be the gate ruling on something it has
        no knowledge of, and would silently disable any caller-supplied centroid
        set that does not mirror DEFAULT_RULES.

        Returns None when the caller should fall through to the band rules:
        either no eligible class has a fitted centroid, or the nearest centroid
        is further than CENTROID_REJECT_RADIUS.
        """
        v = self.feature_vector(f)
        rule_names = {r.name for r in self.rules}
        names, dists = [], []
        for name, c in self.centroids.items():
            if c.shape != v.shape:
                continue
            if eligible is not None and name in rule_names and name not in eligible:
                continue
            d = float(np.linalg.norm((v - c) / self.feature_scales))
            names.append(name)
            dists.append(d)

        if not names:
            return None

        order = np.argsort(dists)
        best, second = order[0], (order[1] if len(order) > 1 else None)
        best_name = names[best]
        runner_up = names[second] if second is not None else None

        if dists[best] > self.reject_radius:
            return None

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

    def _classify_by_rules(self, f: Dict[str, float],
                           eligible: Optional[set] = None) -> HeuristicResult:
        pool = [r for r in self.rules
                if eligible is None or r.name in eligible]
        if not pool:
            return HeuristicResult(UNKNOWN, 0.0, features=f)

        scored = sorted(((self._score_rule(r, f), r.name) for r in pool),
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
