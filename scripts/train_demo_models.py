"""
Demo-profile ProtoNet training — consumer-electronics loads.

Why a separate profile
----------------------
The default 10-class model spans 3 W (standby) to 3.5 kW (EV charger). That
dynamic range is what makes its low-power classes weakest — on unseen houses
`laptop` scores F1 = 0.16. A bench demo whose loads are a laptop, a projector
and a phone charger sits entirely inside that weak band, so the general model
is the wrong tool for it.

This script trains a dedicated model over `DEMO_CLASSES`, all within roughly
one decade of power (3 W - 400 W), extracted from real UK-DALE meters:

  laptop            <- b1/m4, b2/m2, b2/m11, b3/m4
  desktop_computer  <- b1/m51, b5/m3, b5/m14, b1/m9 (HTPC), b5/m9
  monitor           <- b1/m14, b2/m3, b5/m6, b5/m10
  projector         <- b3/m5            (single instance — see caveat below)
  tv                <- b1/m7, b4/m2, b5/m5
  router            <- b1/m18, b2/m18, b1/m21, b5/m8
  phone_charger     <- b1/m34, b1/m32, b1/m27

The extraction floor drops to 3 W (from 20 W) so charger-class loads survive
the quality gates.

Caveats worth stating out loud
------------------------------
* `projector` appears on exactly one meter in one house, so its accuracy cannot
  be validated on an unseen house. Treat its score as optimistic.
* A phone charger at 3-10 W is below `TRANSIENT_THRESHOLD_W` (20 W) in
  `src/pipeline/aggregate_nilm.py`, and below the 100 A CT's 4.6 W start
  current. Even a perfect classifier will not be handed an event for it unless
  you fit the 10 A PZEM variant and lower the detector threshold.

Usage:
  source .venv/bin/activate
  python scripts/train_demo_models.py                 # full run
  python scripts/train_demo_models.py --episodes 2000 # quick check
"""
import os
import sys
import json
import time
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.protonet    import ProtoNet, OpenMaxWeibull, PrototypeRegistry
from src.models.calibration import TemperatureScaler
from data.unified_loader    import (UnifiedNILMDataset, DEMO_CLASSES,
                                    DEMO_EXTRA_MAP)
from scripts.train_models   import (sample_episode, evaluate, evaluate_per_class,
                                    K_SHOT, Q_QUERY, SEQ_LEN)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
WEIGHTS_DIR = 'backend/models/weights_demo'
# Consumer electronics start conducting far below the kitchen-appliance floor.
DEMO_ON_THRESHOLD_W = 3.0


def train_demo(args):
    t0 = time.time()
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    dataset = UnifiedNILMDataset(
        seq_len=SEQ_LEN,
        sources=['ukdale', 'redd'],          # synthetic has no projector/monitor/router
        min_samples_per_class=args.min_samples,
        max_samples_per_class=args.max_samples,
        target_classes=DEMO_CLASSES,
        extra_canonical_map=DEMO_EXTRA_MAP,
        on_threshold_w=DEMO_ON_THRESHOLD_W,
        cache_tag='_demo',
    )

    data = dataset.load_all_classes()
    if len(data) < 2:
        logger.error(f"Only {len(data)} demo classes have data — cannot train. "
                     f"Check that data/real/ukdale.h5 exists and hdf5plugin is installed.")
        return

    house_train, house_val, house_info = dataset.get_house_holdout_split(
        holdout_frac=args.house_holdout)

    if house_val and len(house_val) >= 2 and not args.no_house_holdout:
        train_data, val_data = house_train, house_val
        protocol = 'unseen_house_holdout'
    else:
        train_data, val_data = dataset.get_train_val_split(val_fraction=0.2)
        protocol = 'random_split'
        logger.warning("Unseen-house holdout not viable for the demo class set "
                       "(too few houses per class) — using a random split. "
                       "Scores will be optimistic.")

    logger.info(f"Protocol: {protocol}")
    for cls in sorted(train_data):
        logger.info(f"  {cls:18s} train={len(train_data[cls]):5d} "
                    f"val={len(val_data.get(cls, [])):5d}")

    n_way = min(args.n_way, len(train_data))
    if n_way < 2:
        logger.error("Need >= 2 classes with data. Aborting.")
        return

    model = ProtoNet(seq_len=SEQ_LEN).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.episodes)
    nll = nn.NLLLoss()
    rng = np.random.default_rng(42)

    logger.info(f"Training {args.episodes} episodes ({n_way}-way {K_SHOT}-shot) on {DEVICE}")
    best_val, best_ep, patience, accs = 0.0, 0, 0, []
    ckpt = f'{WEIGHTS_DIR}/protonet.pt'

    for ep in range(1, args.episodes + 1):
        model.train()
        sup, qry, lbl = sample_episode(train_data, n_way, K_SHOT, Q_QUERY, rng)
        sup = torch.tensor(sup, dtype=torch.float32).to(DEVICE)
        qry = torch.tensor(qry, dtype=torch.float32).to(DEVICE)
        lbl = torch.tensor(lbl, dtype=torch.long).to(DEVICE)

        log_probs, _ = model(sup, qry)
        loss = nll(log_probs, lbl)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optim.step()
        sched.step()
        accs.append((log_probs.argmax(dim=1) == lbl).float().mean().item())

        if ep % 500 == 0 or ep == args.episodes:
            va = evaluate(model, val_data, n_episodes=100,
                          rng=np.random.default_rng(ep))
            logger.info(f"  Episode {ep:5d} | loss {loss.item():.4f} | "
                        f"train_acc {np.mean(accs[-500:]):.3f} | val_acc {va:.3f}")
            if va > best_val:
                best_val, best_ep, patience = va, ep, 0
                torch.save(model.state_dict(), ckpt)
                logger.info(f"  ✅ New best val_acc={va:.3f} — saved")
            else:
                patience += 1
                if patience >= args.patience:
                    logger.info(f"  ⏹️ Early stop at {ep} (best {best_val:.3f} @ {best_ep})")
                    break

    if not os.path.exists(ckpt):
        torch.save(model.state_dict(), ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    model.eval()

    # ── Prototype registry over every demo class ──
    all_data = {cls: segs for cls, segs in train_data.items()}
    for cls, segs in val_data.items():
        all_data[cls] = (np.concatenate([all_data[cls], segs], axis=0)
                         if cls in all_data else segs)

    registry = PrototypeRegistry(model, device=DEVICE)
    for cls, segs in all_data.items():
        registry.add_class(cls, segs[:K_SHOT * 2])
    registry.save(f'{WEIGHTS_DIR}/prototype_registry.pt')
    logger.info(f"Prototype Registry saved ({len(registry.class_names())} classes)")

    # ── OpenMax Weibull tails ──
    openmax = OpenMaxWeibull(num_classes=len(all_data))
    with torch.no_grad():
        for idx, (cls, segs) in enumerate(all_data.items()):
            x = torch.tensor(np.asarray(segs[:50], dtype=np.float32)).to(DEVICE)
            emb = model.embed(x)
            proto, _ = registry.prototypes[cls]
            openmax.fit(idx, torch.sum((emb - proto) ** 2, dim=1).cpu().numpy())
    openmax.save(f'{WEIGHTS_DIR}/openmax_weibull.pkl')

    # ── Temperature calibration ──
    scaler = TemperatureScaler()
    logits, labels = [], []
    with torch.no_grad():
        for _ in range(200):
            try:
                s, q, l = sample_episode(val_data, min(n_way, len(val_data)),
                                         K_SHOT, Q_QUERY,
                                         rng=np.random.default_rng())
                _, d = model(torch.tensor(s, dtype=torch.float32).to(DEVICE),
                             torch.tensor(q, dtype=torch.float32).to(DEVICE))
                logits.append((-d).cpu().numpy())
                labels.append(l)
            except Exception:
                continue
    if logits:
        scaler.calibrate(np.concatenate(logits), np.concatenate(labels))
        scaler.save(f'{WEIGHTS_DIR}/temperature_scaler.pt')
        logger.info(f"Temperature T = {scaler.temperature.item():.4f}")

    # ── Honest closed-set report ──
    closed = evaluate_per_class(model, registry, val_data)
    if closed:
        logger.info(f"\nClosed-set eval ({closed['n_samples']} windows, {protocol}):")
        logger.info(f"  overall_accuracy = {closed['overall_accuracy']:.4f}")
        logger.info(f"  macro_f1         = {closed['macro_f1']:.4f}")
        for cls, f1 in closed['per_class_f1'].items():
            logger.info(f"    {cls:18s} F1 = {f1:.4f}"
                        f"{'  <-- WEAK' if f1 < 0.60 else ''}")

    report = {
        'profile': 'demo_consumer_electronics',
        'trained_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'classes': sorted(all_data.keys()),
        'evaluation_protocol': protocol,
        'house_split': house_info,
        'on_threshold_w': DEMO_ON_THRESHOLD_W,
        'episodic_val_accuracy': round(float(best_val), 4),
        'closed_set_eval': closed,
        'class_provenance': dataset.provenance,
        'weights_dir': WEIGHTS_DIR,
        'caveats': [
            "projector has a single meter in a single house — its score is optimistic",
            "phone_charger (3-10W) is below TRANSIENT_THRESHOLD_W=20W in "
            "src/pipeline/aggregate_nilm.py and below the 100A CT's 4.6W start "
            "current; needs the 10A PZEM variant plus a lowered detector threshold",
        ],
    }
    os.makedirs('training_results', exist_ok=True)
    with open('training_results/demo_training_report.json', 'w') as fh:
        json.dump(report, fh, indent=2)
    logger.info(f"Report -> training_results/demo_training_report.json")
    logger.info(f"Weights -> {WEIGHTS_DIR}/  ({time.time()-t0:.1f}s)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Demo-profile ProtoNet training')
    ap.add_argument('--episodes', type=int, default=8000)
    ap.add_argument('--n-way', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--max-samples', type=int, default=800)
    ap.add_argument('--min-samples', type=int, default=20)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--house-holdout', type=float, default=0.3)
    ap.add_argument('--no-house-holdout', action='store_true')
    train_demo(ap.parse_args())
