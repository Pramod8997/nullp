"""
Fit the heuristic fallback's class centroids from real UK-DALE / REDD windows.

The fallback in src/pipeline/heuristic_fallback.py classifies by nearest
centroid in a 5-feature space. This script computes those centroids (and the
per-feature robust scales) from the real windows cached by
data/nilmtk_reader.py, then rewrites the CLASS_CENTROIDS / FEATURE_SCALES
constants in that module in place.

Constants are embedded in source rather than loaded from a file on purpose:
the fallback exists precisely for the case where model artefacts are missing or
unreadable, so it must not depend on an artefact of its own.

Usage:
  source .venv/bin/activate
  python scripts/fit_heuristic_centroids.py                    # default classes
  python scripts/fit_heuristic_centroids.py --cache-tag _demo  # demo classes
  python scripts/fit_heuristic_centroids.py --dry-run          # print only
"""
import os
import re
import sys
import glob
import logging
import argparse

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.heuristic_fallback import HeuristicApplianceClassifier

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger(__name__)

MODULE_PATH = 'src/pipeline/heuristic_fallback.py'
CACHE_GLOB = 'data/real/cache/*_windows{tag}.npz'
GROUP_PREFIX = '__groups__'


def load_cached_real(cache_tag: str = '') -> dict:
    """Merge every cached real-window npz into {class: (N, 128)}."""
    out = {}
    paths = sorted(glob.glob(CACHE_GLOB.format(tag=cache_tag)))
    if not paths:
        logger.error(f"No caches matching {CACHE_GLOB.format(tag=cache_tag)}. "
                     f"Run training (or the loader) once to populate them.")
        return {}
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            for k in z.files:
                if k.startswith(GROUP_PREFIX):
                    continue
                arr = np.asarray(z[k], dtype=np.float32)
                if arr.ndim == 2 and arr.shape[0] > 0:
                    out.setdefault(k, []).append(arr)
        logger.info(f"read {p}")
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def fit(data: dict, on_threshold_w: float):
    """Return (centroids, scales) in the classifier's feature space."""
    clf = HeuristicApplianceClassifier(on_threshold_w=on_threshold_w,
                                       centroids={})
    per_class_vecs, all_vecs = {}, []

    for cls, arr in sorted(data.items()):
        vecs = []
        for w in arr:
            f = clf.extract_features(w)
            if not f or f.get('steady_w', 0.0) <= 0.0:
                continue
            vecs.append(clf.feature_vector(f))
        if len(vecs) < 5:
            logger.warning(f"{cls}: only {len(vecs)} usable windows — skipping")
            continue
        V = np.stack(vecs)
        per_class_vecs[cls] = V
        all_vecs.append(V)
        logger.info(f"{cls:18s} n={len(V):5d}")

    if not per_class_vecs:
        return {}, None

    # Median centroid — resistant to the outlier windows real meters produce.
    centroids = {cls: np.median(V, axis=0) for cls, V in per_class_vecs.items()}

    # Robust per-feature scale: median absolute deviation of the pooled
    # within-class residuals, so a feature that varies a lot inside a class
    # carries less weight in the distance.
    resid = np.concatenate([V - centroids[cls] for cls, V in per_class_vecs.items()])
    mad = np.median(np.abs(resid), axis=0) * 1.4826
    scales = np.where(mad > 1e-3, mad, 1.0)

    return centroids, scales


def patch_module(centroids: dict, scales: np.ndarray) -> None:
    """Rewrite FEATURE_SCALES and CLASS_CENTROIDS in the fallback module."""
    src = open(MODULE_PATH).read()

    scales_txt = ("FEATURE_SCALES: Tuple[float, ...] = ("
                  + ", ".join(f"{v:.4f}" for v in scales) + ")")
    src, n1 = re.subn(r"FEATURE_SCALES: Tuple\[float, \.\.\.\] = \([^)]*\)",
                      scales_txt, src, count=1)

    lines = ["CLASS_CENTROIDS: Dict[str, Tuple[float, ...]] = {"]
    for cls in sorted(centroids):
        vals = ", ".join(f"{v:.4f}" for v in centroids[cls])
        lines.append(f"    {cls!r}: ({vals}),")
    lines.append("}")
    cent_txt = "\n".join(lines)

    src, n2 = re.subn(
        r"CLASS_CENTROIDS: Dict\[str, Tuple\[float, \.\.\.\]\] = \{[^}]*\}",
        cent_txt, src, count=1, flags=re.S)

    if not (n1 and n2):
        logger.error(f"Could not locate constants to patch "
                     f"(FEATURE_SCALES={n1}, CLASS_CENTROIDS={n2})")
        return

    open(MODULE_PATH, 'w').write(src)
    logger.info(f"Patched {MODULE_PATH}: {len(centroids)} centroids")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-tag', default='',
                    help="'' for the default class set, '_demo' for demo classes")
    ap.add_argument('--on-threshold-w', type=float, default=None,
                    help='Defaults to 20 W, or 3 W when --cache-tag=_demo')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    thr = args.on_threshold_w
    if thr is None:
        thr = 3.0 if args.cache_tag == '_demo' else 20.0

    data = load_cached_real(args.cache_tag)
    if not data:
        sys.exit(1)

    centroids, scales = fit(data, thr)
    if not centroids:
        logger.error("Nothing fitted.")
        sys.exit(1)

    print("\nFEATURE_SCALES =", ", ".join(f"{v:.4f}" for v in scales))
    print("CLASS_CENTROIDS:")
    for cls in sorted(centroids):
        print(f"  {cls:18s}", ", ".join(f"{v:8.4f}" for v in centroids[cls]))

    if args.dry_run:
        logger.info("--dry-run: module not modified")
    else:
        patch_module(centroids, scales)


if __name__ == '__main__':
    main()
