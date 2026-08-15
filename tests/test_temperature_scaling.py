import pytest
import torch
from src.models.calibration import temperature_scale, confidence_gate

try:
    from hypothesis import assume
except ImportError:
    def assume(condition: bool):
        assert condition


# TEST 2D-1: T=1 leaves logits unchanged
def test_temp_scaling_t1_identity():
    logits = torch.tensor([[2.0, 1.0, 0.5]])
    probs_raw = torch.softmax(logits, dim=1)
    probs_scaled = temperature_scale(logits, T=1.0)
    assert torch.allclose(probs_raw, probs_scaled, atol=1e-5)

# TEST 2D-2: T>1 makes distribution softer (lower max probability)
def test_temp_scaling_high_t_softens():
    logits = torch.tensor([[5.0, 0.0, -5.0]])
    probs_t1 = temperature_scale(logits, T=1.0)
    probs_t5 = temperature_scale(logits, T=5.0)
    assert probs_t5[0].max().item() < probs_t1[0].max().item()

# TEST 2D-3: T<1 makes distribution sharper (higher max probability)
def test_temp_scaling_low_t_sharpens():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    probs_t1 = temperature_scale(logits, T=1.0)
    probs_t05 = temperature_scale(logits, T=0.5)
    assert probs_t05[0].max().item() > probs_t1[0].max().item()

# TEST 2D-4: Confidence gate — confidence < 0.90 must NEVER reach RL agent
def test_confidence_gate_blocks_rl_below_threshold():
    # Construct logits that produce max softmax < 0.90
    logits = torch.tensor([[2.0, 1.8, 1.6, 1.4]])  # Spread logits → low max softmax
    T = 1.0
    probs = temperature_scale(logits, T=T)
    confidence = probs.max().item()
    assume(confidence < 0.90)
    result = confidence_gate(confidence, threshold=0.90)
    assert result.action == "SKIP_RL"
    assert result.event_type == "LOW_CONFIDENCE"

# TEST 2D-5: Confidence gate — confidence ≥ 0.90 passes through to RL
def test_confidence_gate_passes_rl_above_threshold():
    # Construct logits that produce max softmax ≥ 0.90
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    probs = temperature_scale(logits, T=1.0)
    confidence = probs.max().item()
    assert confidence >= 0.90
    result = confidence_gate(confidence, threshold=0.90)
    assert result.action == "PASS_RL"

# TEST 2D-6: Boundary value — exactly 0.90 must pass (not block)
def test_confidence_gate_exact_boundary_passes():
    result = confidence_gate(confidence=0.90, threshold=0.90)
    assert result.action == "PASS_RL"

# TEST 2D-7: Boundary value — 0.8999 must be blocked
def test_confidence_gate_just_below_boundary_blocked():
    result = confidence_gate(confidence=0.8999, threshold=0.90)
    assert result.action == "SKIP_RL"
