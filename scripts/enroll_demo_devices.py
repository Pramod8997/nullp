#!/usr/bin/env python3
"""
Enrol this deployment's OWN devices into a prototype registry.

Why this exists
---------------
The shipped demo artifact (`backend/models/weights_demo/prototype_registry.pt`)
is fitted on UK-DALE/REDD. Measured on the cached demo windows
(`data/real/cache/ukdale_windows_demo.npz`):

    class            steady p10 / p50 / p90 W
    laptop                0 /  21 /  45      <- a 2012 netbook, not a 120 W laptop
    monitor              26 /  58 /  72
    projector           168 / 191 / 212
    phone_charger         0 /   0 /   0      <- ZERO windows above the 20 W floor
    router                0 /   0 /   0      <- likewise

So the shipped prototypes physically cannot represent a modern 45-120 W USB-PD
charger or a 120 W laptop, and no amount of pipeline tuning changes that — see
claude_debug/ML_PIPELINE_FIX_2026-08-25.md §2.5. The remedy already in the
codebase is enrolment: `PrototypeRegistry.add_class` records both a prototype
and the power envelope observed on the enrolled segments, and
`EMSOrchestrator._classify_device` treats an enrolled class as self-confirming
(its envelope IS the independent absolute-watts channel, so the deterministic
centroid vote is not consulted) and lets it outrank the shipped classes.

This script performs that enrolment ahead of time for the four demo classes,
using the SAME generator the demo fleet publishes from
(`backend/scripts/simulate_esp32.py:DEMO_DEVICES`) so there is one source of
truth for the operating points. That is the offline equivalent of the operator
labelling each device once through the dashboard.

Honesty note
------------
`--source sim` enrols on the simulator's own distribution. It makes the DEMO
work and is verified on held-out seeds, but it is NOT evidence that the model
generalises to unseen physical hardware. For the physical rig use
`--source capture` with real PZEM windows; until such a capture exists, physical
four-class recognition is NOT PHYSICALLY VERIFIED.

Usage
-----
    python scripts/enroll_demo_devices.py                    # demo, from sim profiles
    python scripts/enroll_demo_devices.py --capture bench.npz  # from real windows
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.protonet import ProtoNet, PrototypeRegistry  # noqa: E402

# The four classes in scope (FAST_FIX_SCOPE.md). Names are the existing wire
# labels — `phone_charger`, not `phone` — because src/api/main.py, the frontend
# and 54 test assertions key on them.
REQUIRED_CLASSES = ("phone_charger", "monitor", "laptop", "projector")

# `desktop_computer` is not in the required four, but node_desktop IS in the demo
# fleet at 250 W, and enrolling the required four alone makes it WORSE: the
# enrolled projector envelope (297-302 W) pads out to 252-347 W, an enrolled
# class outranks a shipped one, and a 250 W desktop window then gets called
# projector. Measured over 24 windows per class:
#
#   variant                    four-class   desktop     novel rejected
#   shipped registry              0/96      12/24            8/8
#   enrol required four only     96/96       8/24            6/8
#   enrol all five fleet nodes   96/96      21/24            7/8
#
# So enrolling every node the fleet actually presents is both the scope-correct
# answer and the strictly better one. This is not taxonomy expansion —
# desktop_computer is already a shipped class and already a demo node.
TARGET_CLASSES = REQUIRED_CLASSES + ("desktop_computer",)

# Which simulated node presents which class.
NODE_FOR_CLASS = {
    "phone_charger":    "node_charger",
    "monitor":          "node_monitor",
    "laptop":           "node_laptop",
    "projector":        "node_projector",
    "desktop_computer": "node_desktop",
}

SEQ_LEN = 128
K_ENROLL = 10          # matches train_demo_models.py's K_SHOT * 2
ENROLL_SEED_BASE = 100  # held out from the seeds used by the regression test


def sim_windows(rated: float, var: float, k: int, seed_base: int,
                seq_len: int = SEQ_LEN, on_fraction: float = 0.8) -> np.ndarray:
    """
    k windows of the gaussian profile `simulate_esp32.simulate_device` publishes
    at 1 Hz: an off-portion, then N(rated, var) clipped at zero.
    """
    out = np.zeros((k, seq_len), dtype=np.float32)
    off = int(seq_len * (1.0 - on_fraction))
    for i in range(k):
        rng = np.random.default_rng(seed_base + i)
        out[i, off:] = np.maximum(0.0, rng.normal(rated, var, seq_len - off))
    return out


def load_sim_profiles() -> dict:
    """{class_name: (rated, var)} read from the demo simulator, not duplicated."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "backend", "scripts"))
    from simulate_esp32 import DEMO_DEVICES  # noqa: E402

    by_id = {d["id"]: d for d in DEMO_DEVICES}
    profiles = {}
    for cls, node in NODE_FOR_CLASS.items():
        cfg = by_id.get(node)
        if cfg is None:
            raise SystemExit(f"{node} missing from simulate_esp32.DEMO_DEVICES")
        profiles[cls] = (float(cfg["rated"]), float(cfg["var"]))
    return profiles


def load_capture(path: str) -> dict:
    """
    {class_name: (K, 128) watts} from a real capture.

    Expects an .npz with one array per class name, each (K, 128) of watts as the
    PZEM reported them. Only TARGET_CLASSES are read.
    """
    z = np.load(path, allow_pickle=False)
    out = {}
    for cls in TARGET_CLASSES:
        if cls not in z:
            print(f"  ⚠ {cls}: absent from {path} — not enrolled")
            continue
        a = np.asarray(z[cls], dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != SEQ_LEN:
            raise SystemExit(f"{cls}: expected (K, {SEQ_LEN}) watts, got {a.shape}")
        if not np.all(np.isfinite(a)):
            raise SystemExit(f"{cls}: capture contains non-finite samples")
        out[cls] = a
    if not out:
        raise SystemExit(f"no target classes found in {path}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="backend/models/weights_demo/protonet.pt",
                    help="encoder checkpoint to embed the enrolment windows with")
    ap.add_argument("--base", default="backend/models/weights_demo/prototype_registry.pt",
                    help="shipped registry to start from ('' for an empty one)")
    ap.add_argument("--out", default="backend/models/weights_demo/prototype_registry_enrolled.pt",
                    help="where to write the enrolled registry")
    ap.add_argument("--capture", default=None, metavar="NPZ",
                    help="real captured windows instead of the simulator profiles")
    ap.add_argument("--k", type=int, default=K_ENROLL,
                    help="windows per class when generating from the simulator")
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        print(f"❌ encoder weights not found: {args.weights}")
        return 1

    encoder = ProtoNet(seq_len=SEQ_LEN)
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing:
        print(f"  ⚠ encoder missing keys: {sorted(missing)[:4]}")
    encoder.eval()

    registry = PrototypeRegistry(encoder)
    if args.base and os.path.exists(args.base):
        registry.load(args.base)
        print(f"base registry: {sorted(registry.class_names())}")
    else:
        print("base registry: (empty)")

    if args.capture:
        segments = load_capture(args.capture)
        print(f"source: capture {args.capture}  ⚠ physical provenance is the "
              f"operator's responsibility")
    else:
        profiles = load_sim_profiles()
        segments = {cls: sim_windows(rated, var, args.k, ENROLL_SEED_BASE)
                    for cls, (rated, var) in profiles.items()}
        print("source: simulate_esp32.DEMO_DEVICES  "
              "⚠ simulated — NOT physical validation")

    for cls in TARGET_CLASSES:
        segs = segments.get(cls)
        if segs is None:
            continue
        # add_class OVERWRITES the shipped prototype for this name and MERGES the
        # envelope, which is the intent: the deployment's own device replaces the
        # UK-DALE stand-in that could not represent it.
        registry.prototypes.pop(cls, None)
        registry.envelopes.pop(cls, None)
        registry.add_class(cls, segs)
        env = registry.power_envelope(cls)
        if env is None:
            print(f"  ⚠ {cls}: no envelope recorded — every enrolment window sat "
                  f"below the 20 W on-threshold, so this class stays "
                  f"unconstrained and will NOT be recognised")
        else:
            print(f"  ✅ {cls:14s} envelope = {env[0]:7.1f} - {env[1]:7.1f} W  "
                  f"(K={len(segs)})")

    registry.save(args.out)
    print(f"\nwrote {args.out}  ({len(registry.class_names())} classes, "
          f"{len(registry.envelopes)} with envelopes)")
    print("Point config.demo.yaml protonet.registry_path at it to use it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
