"""
Temperature Scaling — post-hoc confidence calibration.
Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
"""
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import numpy as np


@dataclass
class ConfidenceGateResult:
    action: str
    event_type: Optional[str] = None
    confidence: float = 0.0
    threshold: float = 0.90


def temperature_scale(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """
    Applies temperature scaling to logits and returns softmax probabilities.
    p_calibrated = softmax(logits / T)
    """
    return torch.softmax(logits / T, dim=-1)


def confidence_gate(confidence: float, threshold: float = 0.90) -> ConfidenceGateResult:
    """
    Confidence gate for gating predictions to RL agent.
    If confidence < threshold: returns action='SKIP_RL', event_type='LOW_CONFIDENCE'.
    If confidence >= threshold: returns action='PASS_RL', event_type=None.
    """
    if confidence >= threshold:
        return ConfidenceGateResult(
            action="PASS_RL",
            event_type=None,
            confidence=confidence,
            threshold=threshold,
        )
    return ConfidenceGateResult(
        action="SKIP_RL",
        event_type="LOW_CONFIDENCE",
        confidence=confidence,
        threshold=threshold,
    )


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling (Guo et al., ICML 2017).
    Trains a single scalar T on a held-out calibration set.
    At inference: calibrated_prob = softmax(logits / T).

    Usage:
        scaler = TemperatureScaler()
        scaler.calibrate(logits_np, labels_np)          # fit T
        probs, conf = scaler.calibrated_confidence(logits_np)  # infer
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Return calibrated softmax probabilities."""
        return torch.softmax(logits / self.temperature.clamp(min=0.05), dim=-1)

    def calibrate(self, logits: np.ndarray, labels: np.ndarray,
                  lr: float = 0.01, max_iter: int = 500):
        """
        Fit T by minimising NLL on the calibration set.

        Args:
            logits: (N, C) uncalibrated logits
            labels: (N,)   integer class indices
        """
        self.train()
        opt = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        lg  = torch.tensor(logits, dtype=torch.float32)
        lb  = torch.tensor(labels, dtype=torch.long)
        nll = nn.CrossEntropyLoss()

        def _eval():
            opt.zero_grad()
            loss = nll(lg / self.temperature.clamp(min=0.05), lb)
            loss.backward()
            return loss

        opt.step(_eval)
        self.eval()
        print(f"[TemperatureScaler] T = {self.temperature.item():.4f}")

    def calibrated_confidence(self, logits: np.ndarray) -> tuple:
        """
        Returns (calibrated_prob_array, max_confidence_float).

        Args:
            logits: (C,) or (1, C) numpy array
        """
        with torch.no_grad():
            lg    = torch.tensor(logits, dtype=torch.float32)
            if lg.dim() == 1:
                lg = lg.unsqueeze(0)
            probs = self(lg).squeeze(0).numpy()
        return probs, float(probs.max())

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        sd = torch.load(path, map_location='cpu', weights_only=False)
        # Colab script saved param as 'T'; local model expects 'temperature'
        if 'T' in sd and 'temperature' not in sd:
            sd = {'temperature': sd['T']}
        self.load_state_dict(sd)
        self.eval()
