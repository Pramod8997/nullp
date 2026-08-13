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
import random
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    logger.info(f"Train classes: {list(train_data.keys())}")
    logger.info(f"Val classes:   {list(val_data.keys())}")
    for cls in sorted(train_data.keys()):
        logger.info(f"  {cls}: train={len(train_data[cls])}, val={len(val_data[cls])}")

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
    logger.info(f"  Files: protonet.pt, prototype_registry.pt, openmax_weibull.pkl, temperature_scaler.pt")
    logger.info(f"{'='*60}")

    # Report verdict
    if final_val_acc >= 0.80:
        logger.info("✅ VERDICT: Model generalizes well (val_acc >= 80%)")
    elif final_val_acc >= 0.60:
        logger.warning("⚠️ VERDICT: Moderate generalization (60-80%) — consider more data")
    else:
        logger.error("❌ VERDICT: Poor generalization (< 60%) — needs custom dataset or architecture changes")


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
    args = parser.parse_args()
    train(args)
