"""
Episodic N-way K-shot meta-training for ProtoNet on synthetic UK-DALE data.

Episode structure (matches report Section 1.5 and FR2):
  N = 5   (number of classes per episode)
  K = 5   (support samples per class, "5-shot")
  Q = 10  (query samples per class per episode)
  Total episodes: 10,000 for full training, 1,000 for quick validation

After training:
  1. Build Prototype Registry on all 10 classes
  2. Fit Weibull tails via OpenMaxWeibull.fit()
  3. Calibrate temperature T via TemperatureScaler.calibrate()
  4. Save all weights to backend/models/weights/
"""
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is importable regardless of cwd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.protonet    import ProtoNet, OpenMaxWeibull, PrototypeRegistry
from src.models.calibration import TemperatureScaler
from data.synd              import SyntheticUKDALE

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
def sample_episode(dataset: dict, n_way: int, k_shot: int, q_query: int):
    """
    Args:
        dataset: {class_name: np.ndarray (M, SEQ_LEN)} — M >= k_shot + q_query

    Returns:
        support: (n_way, k_shot, SEQ_LEN) numpy
        query:   (n_way * q_query, SEQ_LEN) numpy
        labels:  (n_way * q_query,) numpy int
    """
    classes  = random.sample(list(dataset.keys()), n_way)
    sup_list, qry_list, lbl_list = [], [], []

    for idx, cls in enumerate(classes):
        samples = dataset[cls]
        n       = len(samples)
        need    = k_shot + q_query
        chosen  = (np.random.choice(n, need, replace=True)
                   if n < need
                   else np.random.choice(n, need, replace=False))
        sup_list.append(samples[chosen[:k_shot]])
        qry_list.append(samples[chosen[k_shot:]])
        lbl_list.extend([idx] * q_query)

    support = np.stack(sup_list)         # (N, K, SEQ_LEN)
    query   = np.concatenate(qry_list)   # (N*Q, SEQ_LEN)
    labels  = np.array(lbl_list)         # (N*Q,)
    return support, query, labels


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    print("[Meta-Training] Loading synthetic dataset...")
    dataset = SyntheticUKDALE(seq_len=SEQ_LEN).load_all_classes()
    print(f"[Meta-Training] Classes: {list(dataset.keys())}")
    print(f"[Meta-Training] Samples per class: {next(iter(dataset.values())).shape[0]}")

    model = ProtoNet(seq_len=SEQ_LEN).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    nll   = nn.NLLLoss()

    print(f"[Meta-Training] Starting {N_EPISODES} episodes ({N_WAY}-way {K_SHOT}-shot) on {DEVICE}...")

    for ep in range(1, N_EPISODES + 1):
        sup_np, q_np, lbl_np = sample_episode(dataset, N_WAY, K_SHOT, Q_QUERY)

        support = torch.tensor(sup_np, dtype=torch.float32).to(DEVICE)
        query   = torch.tensor(q_np,   dtype=torch.float32).to(DEVICE)
        labels  = torch.tensor(lbl_np, dtype=torch.long).to(DEVICE)

        log_probs, _ = model(support, query)
        loss         = nll(log_probs, labels)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if ep % 500 == 0:
            acc = (log_probs.argmax(dim=1) == labels).float().mean().item()
            print(f"  Episode {ep:5d} | loss {loss.item():.4f} | acc {acc:.3f}")

    # ── Save encoder ──────────────────────────────────────────────────────────
    torch.save(model.state_dict(), f'{WEIGHTS_DIR}/protonet.pt')
    print("[Meta-Training] ProtoNet saved.")

    # ── Build Prototype Registry on ALL 10 classes ───────────────────────────
    print("[Meta-Training] Building Prototype Registry...")
    registry = PrototypeRegistry(model, device=DEVICE)
    for cls_name, segments in dataset.items():
        registry.add_class(cls_name, segments[:K_SHOT])
    registry.save(f'{WEIGHTS_DIR}/prototype_registry.pt')
    print("[Meta-Training] Prototype Registry saved.")

    # ── Fit Weibull tails (OpenMax) ──────────────────────────────────────────
    print("[Meta-Training] Fitting Weibull distributions (OpenMax)...")
    openmax = OpenMaxWeibull(num_classes=len(dataset))
    model.eval()
    with torch.no_grad():
        for idx, (cls_name, segments) in enumerate(dataset.items()):
            x   = torch.tensor(segments[:50], dtype=torch.float32).to(DEVICE)
            emb = model.embed(x)
            proto_tensor, _ = registry.prototypes[cls_name]
            dists = torch.sum((emb - proto_tensor) ** 2, dim=1).cpu().numpy()
            openmax.fit(idx, dists)
    openmax.save(f'{WEIGHTS_DIR}/openmax_weibull.pkl')
    print("[Meta-Training] Weibull models saved.")

    # ── Temperature Scaling calibration ──────────────────────────────────────
    print("[Meta-Training] Calibrating temperature scaling...")
    scaler      = TemperatureScaler()
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for _ in range(200):   # 200 calibration episodes
            sup_np, q_np, lbl_np = sample_episode(dataset, N_WAY, K_SHOT, Q_QUERY)
            sup  = torch.tensor(sup_np, dtype=torch.float32).to(DEVICE)
            q    = torch.tensor(q_np,   dtype=torch.float32).to(DEVICE)
            _, dists = model(sup, q)
            logits_ep = (-dists).cpu().numpy()   # logits = negative distances
            all_logits.append(logits_ep)
            all_labels.append(lbl_np)

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    scaler.calibrate(all_logits, all_labels)
    scaler.save(f'{WEIGHTS_DIR}/temperature_scaler.pt')
    print(f"[Meta-Training] Temperature T = {scaler.temperature.item():.4f} saved.")

    print("\n[Meta-Training] ALL DONE.")
    print(f"  Weights saved to: {WEIGHTS_DIR}/")
    print(f"  Files: protonet.pt, prototype_registry.pt, openmax_weibull.pkl, temperature_scaler.pt")


if __name__ == '__main__':
    train()
