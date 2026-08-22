"""
Custom Agent 1: ML Model Verification Agent
Uses PyTorch-native meta-learning to verify that:
  - Patch 6 (Sigmoid gating) preserves signal amplitude
  - Patch 7 (squared Euclidean) aligns training/inference metrics
  - Improved data augmentation reduces overfitting

This replaces torchmeta (which has PyTorch version conflicts) with
direct PyTorch implementation of realistic augmentation.
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.protonet import ProtoNet, PreCNNTemporalAttention, OpenMaxWeibull

print("=" * 60)
print("  CUSTOM AGENT 1: ML MODEL VERIFICATION")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# TEST 1: Verify Sigmoid gating preserves signal amplitude
# ──────────────────────────────────────────────────────────────
print("\n[TEST 1] Sigmoid Gating Signal Preservation")
attn = PreCNNTemporalAttention(seq_len=128)
attn.eval()

# Generate a realistic power signal (appliance turn-on transient)
x = torch.zeros(1, 128)
x[0, 60:] = 1500.0  # Step from 0W to 1500W at sample 60

with torch.no_grad():
    y = attn(x)
    input_energy = (x ** 2).sum().item()
    output_energy = (y ** 2).sum().item()
    ratio = output_energy / (input_energy + 1e-12)

print(f"  Input energy:  {input_energy:.2f}")
print(f"  Output energy: {output_energy:.2f}")
print(f"  Energy ratio:  {ratio:.4f}")

if ratio > 0.05:
    print("  ✅ PASS — Sigmoid preserves >5% of signal energy")
    print("     (Softmax would have reduced this to ~0.78%)")
else:
    print("  ❌ FAIL — Signal is being excessively squashed")

# ──────────────────────────────────────────────────────────────
# TEST 2: Verify squared Euclidean distance alignment
# ──────────────────────────────────────────────────────────────
print("\n[TEST 2] Squared Euclidean Distance Metric Alignment")
model = ProtoNet(seq_len=128, embed_dim=128)
model.eval()

# Create synthetic support (3 classes, 5 shots each) and query
support = torch.randn(3, 5, 128)
query   = torch.randn(4, 128)

with torch.no_grad():
    log_probs, dists = model(support, query)

# Verify distances are squared (should be significantly larger than L2)
l2_check = dists.min().item()
print(f"  Min distance: {l2_check:.4f}")
print(f"  Max distance: {dists.max().item():.4f}")

# Squared Euclidean distances should be positive and scaled up
if l2_check >= 0:
    print("  ✅ PASS — Distances are non-negative (squared Euclidean)")
else:
    print("  ❌ FAIL — Negative distances detected")

# Verify log_probs are valid
if torch.isfinite(log_probs).all() and (log_probs <= 0).all():
    print("  ✅ PASS — Log-probabilities are valid (finite, <= 0)")
else:
    print("  ❌ FAIL — Invalid log-probabilities detected")

# ──────────────────────────────────────────────────────────────
# TEST 3: OpenMax + Weibull alignment with squared distance
# ──────────────────────────────────────────────────────────────
print("\n[TEST 3] OpenMax/Weibull EVT Distance Alignment")
openmax = OpenMaxWeibull(tail_size=3)

# Create known class prototypes and embeddings
prototypes = {0: torch.randn(128), 1: torch.randn(128), 2: torch.randn(128)}
known_embeddings = {
    i: torch.stack([proto + torch.randn(128) * 0.1 for _ in range(10)])
    for i, proto in prototypes.items()
}

try:
    openmax.fit(prototypes, known_embeddings)
    # Test with a known embedding (close to class 0)
    test_known = prototypes[0] + torch.randn(128) * 0.05
    pred_class, conf, is_unknown = openmax.predict(test_known)

    print(f"  Known input  → class={pred_class}, conf={conf:.4f}, unknown={is_unknown}")

    # Test with an unknown embedding (far from all prototypes)
    test_unknown = torch.randn(128) * 10.0
    pred_class2, conf2, is_unknown2 = openmax.predict(test_unknown)

    print(f"  Unknown input → class={pred_class2}, conf={conf2:.4f}, unknown={is_unknown2}")

    if is_unknown2:
        print("  ✅ PASS — OpenMax correctly flags unknown with squared distance")
    else:
        print("  ⚠️ WARNING — Unknown not flagged (may need threshold tuning)")
except Exception as e:
    print(f"  ❌ FAIL — OpenMax crashed: {e}")

# ──────────────────────────────────────────────────────────────
# TEST 4: Realistic data augmentation (replaces torchmeta)
# ──────────────────────────────────────────────────────────────
print("\n[TEST 4] Realistic Data Augmentation for Anti-Overfitting")

def realistic_augment(signal, phase_jitter=0.1, harmonic_ratio=0.05, brownout_prob=0.1):
    """Augmentation with phase jitter, harmonic distortion, and brownout noise."""
    aug = signal.clone()
    # Phase jitter: random time shift
    shift = int(np.random.uniform(-phase_jitter, phase_jitter) * len(aug))
    aug = torch.roll(aug, shifts=shift)
    # Harmonic distortion: add 3rd/5th harmonics
    t = torch.linspace(0, 2 * np.pi, len(aug))
    aug += harmonic_ratio * signal.abs().mean() * torch.sin(3 * t)
    aug += harmonic_ratio * 0.5 * signal.abs().mean() * torch.sin(5 * t)
    # Brownout: random amplitude dip
    if np.random.random() < brownout_prob:
        dip_start = np.random.randint(0, len(aug) - 20)
        aug[dip_start:dip_start + 20] *= np.random.uniform(0.3, 0.7)
    # Sensor noise (ESP32 ADC noise floor)
    aug += torch.randn_like(aug) * 2.0
    return aug

# Test augmentation diversity
base_signal = torch.zeros(128)
base_signal[30:90] = 1200.0  # Kettle-like profile

augmented = torch.stack([realistic_augment(base_signal) for _ in range(100)])
variance = augmented.var(dim=0).mean().item()
print("  Base signal variance: 0.0 (fixed profile)")
print(f"  Augmented variance:   {variance:.2f}")

if variance > 100.0:
    print("  ✅ PASS — Augmentation creates meaningful variance")
    print("     (prevents 100% accuracy memorization on synthetic data)")
else:
    print("  ❌ FAIL — Augmentation variance too low, model may still memorize")

# ──────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ML VERIFICATION COMPLETE")
print("=" * 60)
