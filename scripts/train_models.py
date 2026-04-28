"""
Episodic Meta-Learning Trainer for ProtoNet CNN
================================================
Supports:
  - Episodic N-way K-shot training on synthetic mock data
  - --datasets flag for future UK-DALE / REDD real dataset support
  - Temperature Scaling calibration after training
  - Weibull OpenMax fitting on support set
  - Evaluation report: accuracy, ECE, unknown rejection rate
"""

import argparse
import os
import sys
import logging
import time

import numpy as np
import h5py
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.protonet import (
    CNN1DEncoder, TemperatureScaler, WEibullOpenMax,
    SupportSetManager, EpisodicDataset
)
from src.pipeline.calibration import compute_ece

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_mock_data(data_path: str) -> dict:
    """Load synthetic UK-DALE-like data from HDF5."""
    windows_per_class = {}
    with h5py.File(data_path, 'r') as hf:
        appliances = hf['appliances']
        for class_name in appliances:
            grp = appliances[class_name]
            windows = np.array(grp['windows'])
            windows_per_class[class_name] = windows
    logger.info(f"Loaded {len(windows_per_class)} classes from {data_path}")
    for cls, w in windows_per_class.items():
        logger.info(f"  {cls}: {w.shape[0]} windows, shape {w.shape[1:]}")
    return windows_per_class


def train_protonet_episodic(encoder, dataset, n_way, k_shot, n_query,
                             n_episodes, lr, device):
    """
    Episodic training loop for ProtoNet.
    Each episode:
      1. Sample n_way classes
      2. For each class: k_shot support + n_query query samples
      3. Compute prototype = mean of support embeddings per class
      4. Compute squared Euclidean distance from each query to each prototype
      5. Loss = cross-entropy over softmax(-distances)
      6. Backprop and update encoder
    """
    encoder = encoder.to(device)
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    losses = []
    accs = []

    for episode in range(n_episodes):
        support, query, labels = dataset.sample_episode(n_way, k_shot, n_query)
        support = support.to(device)
        query = query.to(device)
        labels = labels.to(device)

        support_emb = encoder(support)   # [n_way*k_shot, 128]
        query_emb = encoder(query)       # [n_way*n_query, 128]

        # Prototypes: mean of support embeddings per class
        n_way_actual = min(n_way, len(dataset.class_names))
        prototypes = support_emb.view(n_way_actual, k_shot, -1).mean(dim=1)  # [n_way, 128]

        # Squared Euclidean distances: [n_query_total, n_way]
        dists = torch.cdist(query_emb, prototypes, p=2).pow(2)

        # Loss: cross-entropy over negative distances
        log_probs = F.log_softmax(-dists, dim=1)
        loss = F.nll_loss(log_probs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (log_probs.argmax(1) == labels).float().mean().item()
        losses.append(loss.item())
        accs.append(acc)

        if episode % 100 == 0:
            avg_loss = np.mean(losses[-100:]) if losses else 0
            avg_acc = np.mean(accs[-100:]) if accs else 0
            logger.info(f"Episode {episode}/{n_episodes}: loss={avg_loss:.4f}, acc={avg_acc:.3f}")

    final_acc = np.mean(accs[-100:]) if accs else 0
    logger.info(f"Training complete. Final 100-episode avg accuracy: {final_acc:.3f}")
    return encoder


def calibrate_temperature(encoder, dataset, n_way, k_shot, n_query,
                          n_episodes, device) -> TemperatureScaler:
    """Run calibration episodes to fit temperature scalar."""
    encoder.eval()
    scaler = TemperatureScaler()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for _ in range(n_episodes):
            support, query, labels = dataset.sample_episode(n_way, k_shot, n_query)
            support = support.to(device)
            query = query.to(device)

            support_emb = encoder(support)
            query_emb = encoder(query)

            n_way_actual = min(n_way, len(dataset.class_names))
            prototypes = support_emb.view(n_way_actual, k_shot, -1).mean(dim=1)
            dists = torch.cdist(query_emb, prototypes, p=2).pow(2)
            logits = -dists  # [n_query_total, n_way]

            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Fit temperature
    scaler.fit(all_logits, all_labels)
    logger.info(f"Temperature scaling fitted: T = {scaler.temperature.item():.4f}")

    # Compute ECE before and after
    probs_before = F.softmax(all_logits, dim=1)
    max_probs_before, preds_before = probs_before.max(dim=1)
    correct_before = (preds_before == all_labels).numpy().tolist()
    ece_before = compute_ece(max_probs_before.numpy().tolist(), correct_before)

    scaled_logits = scaler(all_logits)
    probs_after = F.softmax(scaled_logits, dim=1)
    max_probs_after, preds_after = probs_after.max(dim=1)
    correct_after = (preds_after.detach() == all_labels).numpy().tolist()
    ece_after = compute_ece(max_probs_after.detach().numpy().tolist(), correct_after)

    logger.info(f"ECE before scaling: {ece_before:.4f}")
    logger.info(f"ECE after scaling:  {ece_after:.4f}")

    return scaler


def evaluate_unknown_rejection(encoder, weibull, support_manager, scaler,
                                known_windows, unknown_windows, confidence_threshold, device):
    """Evaluate unknown rejection rate on held-out data."""
    encoder.eval()
    
    # Test on known samples
    known_correct = 0
    known_total = 0
    for cls, windows in known_windows.items():
        for w in windows[:20]:  # Test first 20 per class
            pred_cls, conf, _ = support_manager.classify(
                w, encoder, weibull, scaler, confidence_threshold, device
            )
            if pred_cls == cls:
                known_correct += 1
            known_total += 1

    known_acc = known_correct / max(known_total, 1)

    # Test on unknown samples (should be rejected)
    unknown_rejected = 0
    unknown_total = len(unknown_windows)
    for w in unknown_windows:
        pred_cls, conf, _ = support_manager.classify(
            w, encoder, weibull, scaler, confidence_threshold, device
        )
        if pred_cls == "unknown":
            unknown_rejected += 1

    rejection_rate = unknown_rejected / max(unknown_total, 1)

    logger.info(f"Known class accuracy: {known_acc:.3f} ({known_correct}/{known_total})")
    logger.info(f"Unknown rejection rate: {rejection_rate:.3f} ({unknown_rejected}/{unknown_total})")
    return known_acc, rejection_rate


def main():
    parser = argparse.ArgumentParser(description="EMS ProtoNet Episodic Trainer")
    parser.add_argument("--datasets", nargs="+", default=["synthetic"],
                        choices=["synthetic", "ukdale", "redd"],
                        help="Datasets to train on. 'synthetic' uses mock_ukdale.h5. "
                             "'ukdale' and 'redd' require real data (Phase 2).")
    parser.add_argument("--data-path", type=str, default="backend/data/mock_ukdale.h5",
                        help="Path to HDF5 data file")
    parser.add_argument("--n-way", type=int, default=5, help="N-way for episodic training")
    parser.add_argument("--k-shot", type=int, default=5, help="K-shot for support set")
    parser.add_argument("--n-query", type=int, default=15, help="Query samples per class per episode")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--embedding-size", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--window-size", type=int, default=60, help="Input window size")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--weights-dir", type=str, default="backend/models/weights",
                        help="Directory to save model weights")
    args = parser.parse_args()

    # Check dataset support
    for ds in args.datasets:
        if ds in ["ukdale", "redd"]:
            logger.warning(f"Dataset '{ds}' requires real data loading (Phase 2). "
                          f"Falling back to synthetic data.")

    # Device selection
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        logger.info("Run 'python scripts/generate_mock_ukdale.py' first to generate synthetic data.")
        sys.exit(1)

    windows_per_class = load_mock_data(args.data_path)

    # Split into train and held-out classes for unknown evaluation
    all_classes = list(windows_per_class.keys())
    if len(all_classes) >= 8:
        train_classes = all_classes[:8]
        unknown_classes = all_classes[8:]
    else:
        train_classes = all_classes
        unknown_classes = []

    train_data = {k: windows_per_class[k] for k in train_classes}
    unknown_data = {k: windows_per_class[k] for k in unknown_classes}

    logger.info(f"Train classes ({len(train_classes)}): {train_classes}")
    logger.info(f"Unknown/held-out classes ({len(unknown_classes)}): {unknown_classes}")

    # Create episodic dataset
    # Convert to list format for EpisodicDataset
    train_windows_list = {k: [w for w in v] for k, v in train_data.items()}
    dataset = EpisodicDataset(train_windows_list)

    # Initialize encoder
    encoder = CNN1DEncoder(input_size=args.window_size, embedding_size=args.embedding_size)

    # ═══ Phase 1: Episodic Training ═══
    logger.info("=" * 60)
    logger.info("  Phase 1: Episodic Meta-Learning Training")
    logger.info("=" * 60)

    start_time = time.time()
    encoder = train_protonet_episodic(
        encoder, dataset,
        n_way=args.n_way, k_shot=args.k_shot, n_query=args.n_query,
        n_episodes=args.episodes, lr=args.lr, device=device
    )
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f}s")

    # ═══ Phase 2: Generate Support Set & Prototypes ═══
    logger.info("=" * 60)
    logger.info("  Phase 2: Support Set & Prototype Generation")
    logger.info("=" * 60)

    support_manager = SupportSetManager()
    for cls, windows in train_data.items():
        # Use first k_shot windows as support set
        for w in windows[:args.k_shot]:
            support_manager.add_support(cls, w)

    prototypes = support_manager.compute_prototypes(encoder, device)
    logger.info(f"Generated prototypes for {len(prototypes)} classes")

    # ═══ Phase 3: Weibull OpenMax Fitting ═══
    logger.info("=" * 60)
    logger.info("  Phase 3: Weibull OpenMax Fitting")
    logger.info("=" * 60)

    weibull = WEibullOpenMax(tail_size=20, alpha=3)
    support_manager.fit_openmax(encoder, weibull, device)
    logger.info(f"Weibull models fitted for {len(weibull.weibull_models)} classes")

    # ═══ Phase 4: Temperature Scaling Calibration ═══
    logger.info("=" * 60)
    logger.info("  Phase 4: Temperature Scaling Calibration")
    logger.info("=" * 60)

    scaler = calibrate_temperature(
        encoder, dataset,
        n_way=args.n_way, k_shot=args.k_shot, n_query=args.n_query,
        n_episodes=100, device=device
    )

    # ═══ Phase 5: Unknown Rejection Evaluation ═══
    if unknown_classes:
        logger.info("=" * 60)
        logger.info("  Phase 5: Unknown Rejection Evaluation")
        logger.info("=" * 60)

        # Combine all unknown windows
        unknown_windows_flat = []
        for cls, windows in unknown_data.items():
            unknown_windows_flat.extend(windows[:50])

        evaluate_unknown_rejection(
            encoder, weibull, support_manager, scaler,
            train_data, unknown_windows_flat,
            confidence_threshold=0.90, device=device
        )

    # ═══ Save Everything ═══
    logger.info("=" * 60)
    logger.info("  Saving Artifacts")
    logger.info("=" * 60)

    os.makedirs(args.weights_dir, exist_ok=True)

    weights_path = os.path.join(args.weights_dir, "cnn_weights.pth")
    torch.save(encoder.state_dict(), weights_path)
    logger.info(f"Saved encoder weights: {weights_path}")

    anchors_path = os.path.join(args.weights_dir, "protonet_anchors.pt")
    support_manager.save_registry(anchors_path)
    logger.info(f"Saved support registry: {anchors_path}")

    scaler_path = os.path.join(args.weights_dir, "temperature_scaler.pth")
    scaler.save(scaler_path)
    logger.info(f"Saved temperature scaler: {scaler_path}")

    import pickle
    weibull_path = os.path.join(args.weights_dir, "weibull_openmax.pkl")
    with open(weibull_path, 'wb') as f:
        pickle.dump(weibull, f)
    logger.info(f"Saved Weibull OpenMax: {weibull_path}")

    logger.info("=" * 60)
    logger.info("  ✅ Training pipeline complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
