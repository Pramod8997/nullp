"""
Episodic N-way K-shot meta-training for ProtoNet — v2 (Production).

Trains on unified real + synthetic NILM data with proper train/val evaluation
to prevent overfitting. Supports UK-DALE, REDD, and improved synthetic data.

Episode structure (matches report Section 1.5 and FR2):
  N = 5   (number of classes per episode)
  K = 5   (support samples per class, "5-shot")
  Q = 10  (query samples per class per episode)

After training:
  1. Build Prototype Registry on all classes
  2. Fit Weibull tails via OpenMaxWeibull.fit()
  3. Calibrate temperature T via TemperatureScaler.calibrate()
  4. Save all weights to backend/models/weights/

Usage:
  source .venv/bin/activate
  python scripts/train_models.py                    # Full training (all sources)
  python scripts/train_models.py --sources synd     # Synthetic only
  python scripts/train_models.py --episodes 1000    # Quick validation
"""
import os
import sys
import argparse
import logging
import json
import time
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is importable regardless of cwd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.protonet    import ProtoNet, OpenMaxWeibull, PrototypeRegistry
from src.models.calibration import TemperatureScaler
from data.unified_loader    import UnifiedNILMDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
N_WAY       = 5
K_SHOT      = 5
Q_QUERY     = 10
N_EPISODES  = 10_000
LR          = 1e-3
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
WEIGHTS_DIR = 'backend/models/weights'
SEQ_LEN     = 128
os.makedirs(WEIGHTS_DIR, exist_ok=True)


# ── Episode Sampler ──────────────────────────────────────────────────────────
def sample_episode(dataset: dict, n_way: int, k_shot: int, q_query: int,
                   rng: np.random.Generator = None):
    """
    Args:
        dataset: {class_name: np.ndarray (M, SEQ_LEN)} — M >= k_shot + q_query

    Returns:
        support: (n_way, k_shot, SEQ_LEN) numpy
        query:   (n_way * q_query, SEQ_LEN) numpy
        labels:  (n_way * q_query,) numpy int
    """
    if rng is None:
        rng = np.random.default_rng()

    classes = list(dataset.keys())
    selected = rng.choice(classes,
                          size=min(n_way, len(classes)),
                          replace=False)
    sup_list, qry_list, lbl_list = [], [], []

    for idx, cls in enumerate(selected):
        samples = dataset[cls]
        n       = len(samples)
        need    = k_shot + q_query
        chosen  = (rng.choice(n, need, replace=True)
                   if n < need
                   else rng.choice(n, need, replace=False))
        sup_list.append(samples[chosen[:k_shot]])
        qry_list.append(samples[chosen[k_shot:]])
        lbl_list.extend([idx] * q_query)

    support = np.stack(sup_list)         # (N, K, SEQ_LEN)
    query   = np.concatenate(qry_list)   # (N*Q, SEQ_LEN)
    labels  = np.array(lbl_list)         # (N*Q,)
    return support, query, labels


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_data, n_episodes=200, rng=None):
    """Evaluate model on validation set. Returns mean accuracy."""
    model.eval()
    if rng is None:
        rng = np.random.default_rng(999)

    accs = []
    n_way = min(N_WAY, len(val_data))
    if n_way < 2:
        return 0.0

    for _ in range(n_episodes):
        try:
            sup_np, q_np, lbl_np = sample_episode(val_data, n_way, K_SHOT, Q_QUERY, rng)
            support = torch.tensor(sup_np, dtype=torch.float32).to(DEVICE)
            query   = torch.tensor(q_np,   dtype=torch.float32).to(DEVICE)
            labels  = torch.tensor(lbl_np, dtype=torch.long).to(DEVICE)
            log_probs, _ = model(support, query)
            acc = (log_probs.argmax(dim=1) == labels).float().mean().item()
            accs.append(acc)
        except Exception as e:
            logger.debug(f"Eval episode failed: {e}")
            continue

    return np.mean(accs) if accs else 0.0


# ── Real-domain evaluation ────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_per_class(model, registry, eval_data: dict) -> dict:
    """
    Nearest-prototype classification over a *fixed* class set, scored per class.

    Unlike episodic accuracy (which samples N classes at a time and therefore
    reports an easier task), this is the closed-set metric that matches how the
    deployed pipeline actually classifies: one embedding vs all prototypes.
    """
    model.eval()
    class_names = registry.class_names()
    if len(class_names) < 2:
        return {}

    protos = torch.stack([registry.prototypes[c][0] for c in class_names]).to(DEVICE)
    name_to_idx = {c: i for i, c in enumerate(class_names)}

    y_true, y_pred = [], []
    for cls, segs in eval_data.items():
        if cls not in name_to_idx or len(segs) == 0:
            continue
        x = torch.tensor(np.asarray(segs, dtype=np.float32)).to(DEVICE)
        emb = model.embed(x)                                   # (B, D)
        d = torch.cdist(emb, protos, p=2) ** 2                 # (B, C)
        pred = d.argmin(dim=1).cpu().numpy()
        y_true.extend([name_to_idx[cls]] * len(pred))
        y_pred.extend(pred.tolist())

    if not y_true:
        return {}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    overall = float((y_true == y_pred).mean())

    per_class_f1 = {}
    for cls, i in name_to_idx.items():
        tp = int(((y_pred == i) & (y_true == i)).sum())
        fp = int(((y_pred == i) & (y_true != i)).sum())
        fn = int(((y_pred != i) & (y_true == i)).sum())
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        per_class_f1[cls] = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    return {
        'overall_accuracy': overall,
        'macro_f1': float(np.mean(list(per_class_f1.values()))) if per_class_f1 else 0.0,
        'per_class_f1': {k: round(v, 4) for k, v in sorted(per_class_f1.items())},
        'n_samples': int(len(y_true)),
    }


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    t0 = time.time()

    # ── Load Data ──
    logger.info(f"Loading data from sources: {args.sources}")
    dataset = UnifiedNILMDataset(
        seq_len=SEQ_LEN,
        sources=args.sources,
        min_samples_per_class=20,
        max_samples_per_class=args.max_samples,
    )

    train_data, val_data = dataset.get_train_val_split(val_fraction=0.2)

    # Unseen-house holdout: the metric that actually predicts field behaviour.
    house_train, house_val, house_info = dataset.get_house_holdout_split(
        holdout_frac=args.house_holdout)
    if house_val and not args.no_house_holdout:
        # Train only on the training houses so the holdout stays untouched.
        # ev_charger has no real coverage in either dataset, so its synthetic
        # windows are carried over from the merged split.
        for cls, segs in train_data.items():
            if dataset.provenance.get(cls) == 'synthetic':
                house_train.setdefault(cls, segs)
        train_data, val_data = house_train, house_val
        logger.info(f"Using UNSEEN-HOUSE holdout: train={house_info['train_houses']} "
                    f"val={house_info['holdout_houses']}")
    else:
        logger.warning("Unseen-house holdout unavailable — falling back to a random "
                       "split. Reported accuracy will be optimistic.")

    logger.info(f"Train classes: {list(train_data.keys())}")
    logger.info(f"Val classes:   {list(val_data.keys())}")
    for cls in sorted(train_data.keys()):
        logger.info(f"  {cls}: train={len(train_data[cls])}, "
                    f"val={len(val_data.get(cls, []))}")

    if len(train_data) < 2:
        logger.error("Need at least 2 classes to train ProtoNet. Aborting.")
        return

    # ── Model ──
    model = ProtoNet(seq_len=SEQ_LEN).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.episodes)
    nll = nn.NLLLoss()

    rng = np.random.default_rng(42)
    n_way = min(N_WAY, len(train_data))

    logger.info(f"Starting {args.episodes} episodes ({n_way}-way {K_SHOT}-shot) on {DEVICE}...")

    best_val_acc = 0.0
    best_epoch = 0
    train_accs = []
    patience_counter = 0

    for ep in range(1, args.episodes + 1):
        model.train()
        sup_np, q_np, lbl_np = sample_episode(train_data, n_way, K_SHOT, Q_QUERY, rng)

        support = torch.tensor(sup_np, dtype=torch.float32).to(DEVICE)
        query   = torch.tensor(q_np,   dtype=torch.float32).to(DEVICE)
        labels  = torch.tensor(lbl_np, dtype=torch.long).to(DEVICE)

        log_probs, _ = model(support, query)
        loss = nll(log_probs, labels)

        optim.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optim.step()
        scheduler.step()

        acc = (log_probs.argmax(dim=1) == labels).float().mean().item()
        train_accs.append(acc)

        # ── Periodic Evaluation ──
        if ep % 500 == 0 or ep == args.episodes:
            avg_train_acc = np.mean(train_accs[-500:])
            val_acc = evaluate(model, val_data, n_episodes=100, rng=np.random.default_rng(ep))

            logger.info(f"  Episode {ep:5d} | loss {loss.item():.4f} | "
                       f"train_acc {avg_train_acc:.3f} | val_acc {val_acc:.3f} | "
                       f"lr {scheduler.get_last_lr()[0]:.2e}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = ep
                patience_counter = 0
                torch.save(model.state_dict(), f'{WEIGHTS_DIR}/protonet.pt')
                logger.info(f"  ✅ New best val_acc={val_acc:.3f} — saved!")
            else:
                patience_counter += 1

            # Early stopping with patience
            if patience_counter >= args.patience:
                logger.info(f"  ⏹️ Early stopping at episode {ep} "
                           f"(best val_acc={best_val_acc:.3f} at episode {best_epoch})")
                break

            # Overfitting detection: train >> val
            if avg_train_acc > 0.95 and val_acc < 0.7:
                logger.warning(f"  ⚠️ OVERFITTING DETECTED: train={avg_train_acc:.3f} >> val={val_acc:.3f}")

    # ── Load best model for post-training steps ──
    model.load_state_dict(torch.load(f'{WEIGHTS_DIR}/protonet.pt',
                                      map_location=DEVICE, weights_only=False))
    model.eval()
    logger.info(f"Loaded best model from episode {best_epoch} (val_acc={best_val_acc:.3f})")

    # ── Final evaluation ──
    final_val_acc = evaluate(model, val_data, n_episodes=500, rng=np.random.default_rng(9999))
    logger.info(f"Final validation accuracy (500 episodes): {final_val_acc:.3f}")

    # ── Build Prototype Registry on ALL classes ───────────────────────────
    logger.info("Building Prototype Registry...")
    all_data = {**train_data}
    for cls, segs in val_data.items():
        if cls in all_data:
            all_data[cls] = np.concatenate([all_data[cls], segs], axis=0)
        else:
            all_data[cls] = segs

    registry = PrototypeRegistry(model, device=DEVICE)
    for cls_name, segments in all_data.items():
        registry.add_class(cls_name, segments[:K_SHOT * 2])  # Use 10 support samples
    registry.save(f'{WEIGHTS_DIR}/prototype_registry.pt')
    logger.info(f"Prototype Registry saved ({len(registry.class_names())} classes).")

    # ── Fit Weibull tails (OpenMax) ──────────────────────────────────────────
    logger.info("Fitting Weibull distributions (OpenMax)...")
    openmax = OpenMaxWeibull(num_classes=len(all_data))
    model.eval()
    with torch.no_grad():
        for idx, (cls_name, segments) in enumerate(all_data.items()):
            x = torch.tensor(segments[:50], dtype=torch.float32).to(DEVICE)
            emb = model.embed(x)
            proto_tensor, _ = registry.prototypes[cls_name]
            dists = torch.sum((emb - proto_tensor) ** 2, dim=1).cpu().numpy()
            openmax.fit(idx, dists)
    openmax.save(f'{WEIGHTS_DIR}/openmax_weibull.pkl')
    logger.info("Weibull models saved.")

    # ── Temperature Scaling calibration ──────────────────────────────────────
    logger.info("Calibrating temperature scaling...")
    scaler = TemperatureScaler()
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for _ in range(200):   # 200 calibration episodes
            try:
                sup_np, q_np, lbl_np = sample_episode(val_data, n_way, K_SHOT, Q_QUERY,
                                                       rng=np.random.default_rng())
                sup  = torch.tensor(sup_np, dtype=torch.float32).to(DEVICE)
                q    = torch.tensor(q_np,   dtype=torch.float32).to(DEVICE)
                _, dists = model(sup, q)
                logits_ep = (-dists).cpu().numpy()
                all_logits.append(logits_ep)
                all_labels.append(lbl_np)
            except Exception:
                continue

    if all_logits:
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        scaler.calibrate(all_logits, all_labels)
        scaler.save(f'{WEIGHTS_DIR}/temperature_scaler.pt')
        logger.info(f"Temperature T = {scaler.temperature.item():.4f} saved.")
    else:
        logger.warning("No calibration data available — using T=1.0")

    elapsed = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE in {elapsed:.1f}s")
    logger.info(f"  Best validation accuracy: {best_val_acc:.3f} (episode {best_epoch})")
    logger.info(f"  Final validation accuracy: {final_val_acc:.3f}")
    logger.info(f"  Weights saved to: {WEIGHTS_DIR}/")
    logger.info("  Files: protonet.pt, prototype_registry.pt, openmax_weibull.pkl, temperature_scaler.pt")
    logger.info(f"{'='*60}")

    # ── Closed-set evaluation on UNSEEN houses (the deployment-relevant score) ──
    closed = evaluate_per_class(model, registry, val_data)
    if closed:
        logger.info(f"\nClosed-set nearest-prototype eval on held-out data "
                    f"({closed['n_samples']} windows):")
        logger.info(f"  overall_accuracy = {closed['overall_accuracy']:.4f}")
        logger.info(f"  macro_f1         = {closed['macro_f1']:.4f}")
        for cls, f1 in closed['per_class_f1'].items():
            flag = '  <-- WEAK' if f1 < 0.60 else ''
            logger.info(f"    {cls:18s} F1 = {f1:.4f}{flag}")

    # ── Persist an honest training report ──
    report = {
        'trained_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'sources': args.sources,
        'episodes_run': ep,
        'device': DEVICE,
        'episodic_val_accuracy': round(float(final_val_acc), 4),
        'best_episodic_val_accuracy': round(float(best_val_acc), 4),
        'evaluation_protocol': ('unseen_house_holdout'
                                if (house_val and not args.no_house_holdout)
                                else 'random_split'),
        'house_split': house_info,
        'class_provenance': dataset.provenance,
        'real_windows_total': int(sum(len(v) for v in dataset.real_data.values())),
        'closed_set_eval': closed,
    }
    try:
        os.makedirs('training_results', exist_ok=True)
        with open('training_results/training_report.json', 'w') as fh:
            json.dump(report, fh, indent=2)
        logger.info("Report written to training_results/training_report.json")
    except Exception as e:
        logger.warning(f"Could not write training report: {e}")

    # Report verdict — judged on the unseen-house closed-set macro F1 when
    # available, since episodic N-way accuracy overstates deployed performance.
    verdict_metric = closed.get('macro_f1', final_val_acc) if closed else final_val_acc
    metric_name = 'unseen-house macro F1' if closed else 'episodic val acc'
    if verdict_metric >= 0.80:
        logger.info(f"✅ VERDICT: Model generalizes well ({metric_name}={verdict_metric:.3f})")
    elif verdict_metric >= 0.60:
        logger.warning(f"⚠️ VERDICT: Moderate generalization ({metric_name}={verdict_metric:.3f}) "
                       f"— usable with the confidence gate at 0.90; weak classes listed above")
    else:
        logger.error(f"❌ VERDICT: Poor generalization ({metric_name}={verdict_metric:.3f}) "
                     f"— the heuristic fallback matters; do not trust per-appliance billing")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ProtoNet Meta-Training for NILM')
    parser.add_argument('--sources', nargs='+', default=['ukdale', 'redd', 'synd'],
                        help='Data sources to use: ukdale, redd, synd')
    parser.add_argument('--episodes', type=int, default=N_EPISODES,
                        help='Number of training episodes')
    parser.add_argument('--max-samples', type=int, default=500,
                        help='Max samples per class')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (in 500-episode eval intervals)')
    parser.add_argument('--house-holdout', type=float, default=0.3,
                        help='Fraction of real windows reserved as unseen houses')
    parser.add_argument('--no-house-holdout', action='store_true',
                        help='Use a random split instead (optimistic; not for reporting)')
    args = parser.parse_args()
    train(args)
