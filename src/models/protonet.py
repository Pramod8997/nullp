"""
ProtoNet: Full implementation for Phase-1.

Components:
  1. CNN1DEncoder       — 5-layer 1D-CNN mapping (batch, 128) → (batch, 128)
  2. TemporalAttention  — soft weight over 128 time-steps (Mishra et al., ICLR 2018)
  3. ProtoNet           — Encoder + Attention, episodic forward pass
  4. OpenMaxWeibull     — Weibull EVT open-set detection (Bendale & Boult, CVPR 2016)
  5. PrototypeRegistry  — Incremental per-class prototype store (encoder frozen)

Legacy compatibility classes are preserved below the new implementations.
"""
import os
import pickle
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import exponweib

# ── Constants ───────────────────────────────────────────────────────────────
EMBED_DIM     = 128
CNN_CHANNELS  = [32, 64, 128, 128, 128]   # 5 conv layers
KERNEL_SIZE   = 3
DROPOUT       = 0.1
WEIBULL_TAIL  = 20      # top-k distances used to fit Weibull tail
OPENMAX_ALPHA = 10      # number of top classes revised by OpenMax


# ── 1. 1D-CNN Encoder ───────────────────────────────────────────────────────

class CNN1DEncoder(nn.Module):
    """
    5-layer 1D-CNN mapping (batch, 1, seq_len) → (batch, EMBED_DIM).
    Each block: Conv1d → BatchNorm → ReLU → MaxPool → Dropout.

    Also supports the legacy API: CNN1DEncoder(input_size, embedding_size)
    for backward compatibility with existing test code.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = EMBED_DIM,
                 # Legacy positional arg names kept for backward compat
                 input_size: Optional[int] = None,
                 embedding_size: Optional[int] = None):
        super().__init__()
        # Resolve legacy keyword args
        if embedding_size is not None:
            embed_dim = embedding_size
        # input_size is ignored (handled by AdaptiveAvgPool)

        layers = []
        ch_in  = in_channels
        for ch_out in CNN_CHANNELS:
            layers += [
                nn.Conv1d(ch_in, ch_out, KERNEL_SIZE, padding=1),
                nn.BatchNorm1d(ch_out),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(DROPOUT),
            ]
            ch_in = ch_out
        self.cnn     = nn.Sequential(*layers)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.project = nn.Linear(CNN_CHANNELS[-1], embed_dim)

        # Legacy attention (kept for backward compat with old test that uses encoder directly)
        self._legacy_attention = _LegacyTemporalAttention(hidden_size=CNN_CHANNELS[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Supports two input shapes:
          - (batch, 1, seq_len)   → standard
          - (batch, seq_len)      → unsqueeze channel dim
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)   # (batch, 1, seq_len)
        h = self.cnn(x)                  # (batch, 128, L')
        h = self.pool(h).squeeze(-1)     # (batch, 128)
        return self.project(h)           # (batch, EMBED_DIM)


class _LegacyTemporalAttention(nn.Module):
    """Legacy attention used by old SupportSetManager tests."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_p = x.permute(0, 2, 1)
        w   = F.softmax(self.attention(x_p), dim=1)
        return (x_p * w).sum(dim=1)


# Keep module alias for legacy code
TemporalAttention = _LegacyTemporalAttention


# ── 2. Temporal Attention Module (pre-CNN) ──────────────────────────────────

class PreCNNTemporalAttention(nn.Module):
    """
    Computes a soft weight vector over the 128 raw time-steps.
    High-variance (informative) timesteps receive higher weight.
    Applied BEFORE CNN encoding on the raw segment.
    Reference: Mishra et al., ICLR 2018 — Simple Neural Attentive Meta-Learner.
    """

    def __init__(self, seq_len: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(seq_len, seq_len // 4),
            nn.Tanh(),
            nn.Linear(seq_len // 4, seq_len),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 128) raw segment → (batch, 128) weighted."""
        weights = self.attn(x)
        return x * weights


# ── 3. Full ProtoNet (Encoder + Attention) ──────────────────────────────────

class ProtoNet(nn.Module):
    """
    ProtoNet with temporal attention applied before the CNN encoder.

    embed(x) applies attention → CNN → projection → (batch, EMBED_DIM)
    forward(support, query) runs a full episodic pass.
    """

    def __init__(self, seq_len: int = 128, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.attention = PreCNNTemporalAttention(seq_len)
        self.encoder   = CNN1DEncoder(embed_dim=embed_dim)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 128) → (batch, EMBED_DIM)."""
        x = self.attention(x)
        return self.encoder(x)

    def forward(self, support: torch.Tensor, query: torch.Tensor):
        """
        Episodic forward pass.

        Args:
            support: (N, K, 128)  N classes, K support samples each
            query:   (Q, 128)     Q query samples

        Returns:
            log_probs: (Q, N) log-probabilities over classes
            dists:     (Q, N) squared Euclidean distances
        """
        N, K, L = support.shape
        sup_emb    = self.embed(support.view(N * K, L)).view(N, K, -1)
        prototypes = sup_emb.mean(dim=1)                         # (N, EMBED_DIM)
        q_emb      = self.embed(query)                           # (Q, EMBED_DIM)
        dists      = torch.cdist(q_emb.unsqueeze(0),
                                 prototypes.unsqueeze(0)).squeeze(0)  # (Q, N)
        return F.log_softmax(-dists, dim=1), dists


# ── 4. OpenMax + Weibull EVT ────────────────────────────────────────────────

class OpenMaxWeibull:
    """
    Post-training calibration: fit a Weibull tail model to each class's
    prototype distance distribution, then at inference compute P(unknown).

    Reference: Bendale & Boult, CVPR 2016 — Towards Open Set Deep Networks.

    Usage:
        omw = OpenMaxWeibull(num_classes=10, tail_size=WEIBULL_TAIL)
        omw.fit(class_idx, distances_to_prototype)   # call for each class
        probs, is_unknown = omw.predict(dists, softmax_probs)
    """

    def __init__(self, num_classes: int = 10,
                 tail_size: int       = WEIBULL_TAIL,
                 unknown_threshold: float = 0.5,
                 alpha: int           = OPENMAX_ALPHA):
        self.num_classes       = num_classes
        self.tail_size         = tail_size
        self.unknown_threshold = unknown_threshold
        self.alpha             = alpha
        self._weibull: Dict[int, tuple] = {}      # class_idx → params

        # Legacy support: per-class embedding storage used by fit(embeddings_per_class)
        self._class_embeddings: Dict[str, np.ndarray] = {}

    # ── New API (index-based, matches phase-1 spec) ──────────────────────────

    def fit(self, class_idx_or_embeddings, distances: Optional[np.ndarray] = None):
        """
        Two call signatures:
          fit(class_idx: int, distances: np.ndarray)   ← new spec API
          fit(embeddings_per_class: dict)               ← legacy API
        """
        if isinstance(class_idx_or_embeddings, dict):
            self._fit_legacy(class_idx_or_embeddings)
        else:
            self._fit_indexed(int(class_idx_or_embeddings), distances)

    def _fit_indexed(self, class_idx: int, distances: np.ndarray):
        tail   = np.sort(distances)[-self.tail_size:]
        if len(tail) < 2:
            tail = np.append(tail, tail[-1] + 1e-5)
        params = exponweib.fit(tail, floc=0)
        self._weibull[class_idx] = params

    def _fit_legacy(self, embeddings_per_class: Dict[str, np.ndarray]):
        """Legacy fit: stores embeddings and fits Weibull from distances to centroid."""
        import scipy.stats
        self._class_embeddings = embeddings_per_class
        for idx, (cls_name, embs) in enumerate(embeddings_per_class.items()):
            proto     = embs.mean(axis=0)
            distances = np.linalg.norm(embs - proto, axis=1)
            tail      = np.sort(distances)[-self.tail_size:]
            if len(tail) < 2:
                tail = np.append(tail, tail[-1] + 1e-5)
            shape, loc, scale = scipy.stats.weibull_min.fit(tail, floc=0)
            self._weibull[idx]       = (1, shape, loc, scale)   # exponweib compat shape
            # Also store by name for legacy compute_open_set_prob
            if not hasattr(self, '_weibull_by_name'):
                self._weibull_by_name: Dict[str, Any] = {}
            self._weibull_by_name[cls_name] = {
                'shape': shape, 'loc': loc, 'scale': scale,
                'prototype': proto
            }

    def _weibull_cdf(self, class_idx: int, distance: float) -> float:
        if class_idx not in self._weibull:
            return 0.0
        return float(exponweib.cdf(distance, *self._weibull[class_idx]))

    def predict(self, distances: np.ndarray, softmax_probs: np.ndarray):
        """
        Args:
            distances:     (N,) distance from query to each prototype
            softmax_probs: (N,) raw softmax over known classes

        Returns:
            revised_probs: (N+1,) where index N = unknown class
            is_unknown:    bool — True if P(unknown) > unknown_threshold
        """
        N      = len(softmax_probs)
        ranked = np.argsort(-softmax_probs)

        psi = np.zeros(N)
        for rank, cls in enumerate(ranked[:self.alpha]):
            if cls < N:
                psi[cls] = self._weibull_cdf(cls, distances[cls])

        revised   = softmax_probs * (1.0 - psi)
        p_unknown = float(np.sum(softmax_probs * psi))

        full  = np.append(revised, p_unknown)
        full /= full.sum() + 1e-12

        is_unknown = bool(full[-1] > self.unknown_threshold)
        return full, is_unknown

    # ── Legacy API (used by old SupportSetManager tests) ────────────────────

    def compute_open_set_prob(self, embedding: np.ndarray,
                              class_names: List[str],
                              distances: List[float]) -> float:
        """Legacy API: returns scalar open-set probability."""
        if not hasattr(self, '_weibull_by_name') or not self._weibull_by_name:
            return 0.0

        import scipy.stats
        model_distances = []
        for cls in class_names:
            if cls in self._weibull_by_name:
                proto = self._weibull_by_name[cls]['prototype']
                d     = np.linalg.norm(embedding - proto)
                model_distances.append((d, cls))

        if not model_distances:
            return 1.0

        model_distances.sort(key=lambda x: x[0])
        top_alpha = model_distances[:self.alpha]

        unknown_probs = []
        for d, cls in top_alpha:
            p = self._weibull_by_name[cls]
            cdf = scipy.stats.weibull_min.cdf(d, p['shape'], loc=p['loc'], scale=p['scale'])
            unknown_probs.append(cdf)

        return float(np.max(unknown_probs)) if unknown_probs else 0.0

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'weibull': self._weibull,
                         'weibull_by_name': getattr(self, '_weibull_by_name', {})}, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict) and 'weibull' in data:
            self._weibull = data['weibull']
            self._weibull_by_name = data.get('weibull_by_name', {})
        else:
            # Backward compat: plain dict of params
            self._weibull = data


# Alias used by existing tests
WEibullOpenMax = OpenMaxWeibull


# ── 5. Prototype Registry (incremental, no encoder retraining) ───────────────

class PrototypeRegistry:
    """
    Stores per-class prototype vectors.
    Supports incremental addition of new classes from as few as 5 samples
    without retraining the CNN encoder — the encoder is frozen.
    """

    def __init__(self, encoder: ProtoNet, device: str = 'cpu'):
        self.encoder    = encoder
        self.device     = device
        self.prototypes: Dict[str, Tuple[torch.Tensor, int]] = {}
        self.encoder.eval()

    @torch.no_grad()
    def add_class(self, class_name: str, support_segments: np.ndarray):
        """
        Args:
            class_name:       string label
            support_segments: (K, 128) numpy array, K >= 1
        """
        x     = torch.tensor(support_segments, dtype=torch.float32).to(self.device)
        emb   = self.encoder.embed(x)              # (K, EMBED_DIM)
        proto = emb.mean(dim=0)                     # (EMBED_DIM,)

        if class_name in self.prototypes:
            old_proto, old_n = self.prototypes[class_name]
            new_n    = old_n + len(support_segments)
            new_proto = (old_proto * old_n + proto * len(support_segments)) / new_n
            self.prototypes[class_name] = (new_proto, new_n)
        else:
            self.prototypes[class_name] = (proto, len(support_segments))

    @torch.no_grad()
    def classify(self, segment: np.ndarray):
        """
        Args:
            segment: (128,) numpy array

        Returns:
            (class_name, distance, distances_to_all_dict)
        """
        if not self.prototypes:
            return None, float('inf'), {}

        x   = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(self.device)
        emb = self.encoder.embed(x).squeeze(0)

        dists = {}
        for name, (proto, _) in self.prototypes.items():
            dists[name] = float(torch.dist(emb, proto).item() ** 2)

        best = min(dists, key=dists.get)
        return best, dists[best], dists

    def class_names(self) -> List[str]:
        return list(self.prototypes.keys())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({k: (v[0].cpu(), v[1]) for k, v in self.prototypes.items()}, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.prototypes = {k: (v[0].to(self.device), v[1]) for k, v in data.items()}


# ── 6. Legacy: SupportSetManager (for backward compat with existing tests) ───

class SupportSetManager:
    """
    Legacy wrapper kept for backward compatibility with existing test suite.
    New code should use PrototypeRegistry instead.
    """

    def __init__(self):
        self.raw_windows: Dict[str, List[np.ndarray]] = {}

    def add_support(self, class_name: str, window: np.ndarray) -> None:
        if class_name not in self.raw_windows:
            self.raw_windows[class_name] = []
        self.raw_windows[class_name].append(window)

    def compute_prototypes(self, encoder: nn.Module,
                           device: torch.device = torch.device('cpu')) -> Dict[str, np.ndarray]:
        prototypes = {}
        encoder.eval()
        with torch.no_grad():
            for cls, windows in self.raw_windows.items():
                t   = torch.tensor(np.array(windows), dtype=torch.float32).unsqueeze(1).to(device)
                emb = encoder(t)
                prototypes[cls] = emb.mean(dim=0).cpu().numpy()
        return prototypes

    def fit_openmax(self, encoder: nn.Module, weibull_model: OpenMaxWeibull,
                    device: torch.device = torch.device('cpu')) -> None:
        encoder.eval()
        embs_per_cls: Dict[str, np.ndarray] = {}
        with torch.no_grad():
            for cls, windows in self.raw_windows.items():
                t   = torch.tensor(np.array(windows), dtype=torch.float32).unsqueeze(1).to(device)
                embs_per_cls[cls] = encoder(t).cpu().numpy()
        weibull_model.fit(embs_per_cls)

    def classify(self, window: np.ndarray, encoder: nn.Module,
                 weibull: OpenMaxWeibull, scaler,
                 confidence_threshold: float,
                 device: torch.device = torch.device('cpu')) -> Tuple[str, float, Dict[str, float]]:
        encoder.eval()
        scaler.eval()
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = encoder(x).squeeze(0)
            prototypes = self.compute_prototypes(encoder, device)
            if not prototypes:
                return "unknown", 0.0, {}
            class_names    = list(prototypes.keys())
            proto_tensors  = torch.tensor(
                np.array([prototypes[c] for c in class_names]),
                dtype=torch.float32).to(device)
            dists    = torch.cdist(embedding.unsqueeze(0), proto_tensors, p=2).pow(2).squeeze(0)
            logits   = -dists
            scaled   = scaler(logits.unsqueeze(0)).squeeze(0)
            probs    = F.softmax(scaled, dim=0)
            max_prob, max_idx = torch.max(probs, dim=0)
            predicted = class_names[max_idx.item()]
            confidence = max_prob.item()
            dist_dict  = {n: dists[i].item() for i, n in enumerate(class_names)}
            emb_np     = embedding.cpu().numpy()
            open_set   = weibull.compute_open_set_prob(emb_np, class_names, list(dist_dict.values()))
            if open_set > (1.0 - confidence_threshold):
                return "unknown", open_set, dist_dict
        return predicted, confidence, dist_dict

    def save_registry(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.raw_windows, f)

    def load_registry(self, path: str) -> None:
        with open(path, 'rb') as f:
            self.raw_windows = pickle.load(f)

    def incremental_update(self, class_name: str, new_window: np.ndarray,
                           encoder: nn.Module) -> None:
        self.add_support(class_name, new_window)


# ── 7. Legacy: EpisodicDataset (for backward compat with existing tests) ─────

class EpisodicDataset:
    """Legacy episodic sampler kept for backward compat with existing tests."""

    def __init__(self, raw_windows_per_class: Dict[str, List[np.ndarray]]):
        self.windows     = {k: np.array(v) for k, v in raw_windows_per_class.items()}
        self.class_names = list(self.windows.keys())

    def sample_episode(self, n_way: int, k_shot: int,
                       n_query: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected  = np.random.choice(self.class_names,
                                     size=min(n_way, len(self.class_names)),
                                     replace=False)
        n_way_act = len(selected)
        sup_list, qry_list, lbl_list = [], [], []
        for i, cls in enumerate(selected):
            cw   = self.windows[cls]
            need = k_shot + n_query
            idx  = (np.random.choice(len(cw), need, replace=True)
                    if len(cw) < need
                    else np.random.choice(len(cw), need, replace=False))
            sup_list.append(torch.tensor(cw[idx[:k_shot]],   dtype=torch.float32).unsqueeze(1))
            qry_list.append(torch.tensor(cw[idx[k_shot:]],   dtype=torch.float32).unsqueeze(1))
            lbl_list.extend([i] * n_query)
        support = torch.cat(sup_list, dim=0)
        query   = torch.cat(qry_list, dim=0)
        labels  = torch.tensor(lbl_list, dtype=torch.long)
        return support, query, labels


# Legacy: old TemperatureScaler (thin wrapper around src.models.calibration version)
# Kept here so test imports like `from src.models.protonet import TemperatureScaler` still work.
class TemperatureScaler(nn.Module):
    """
    Legacy TemperatureScaler kept for backward compat.
    Prefer importing from src.models.calibration for new code.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        import scipy.optimize
        def eval_nll(t):
            sl  = logits / t[0]
            lp  = F.log_softmax(sl, dim=1)
            return F.nll_loss(lp, labels).item()
        res = scipy.optimize.minimize(eval_nll, x0=np.array([1.0]),
                                      bounds=[(0.01, 100.0)], method='L-BFGS-B')
        with torch.no_grad():
            self.temperature.copy_(torch.tensor([res.x[0]]))

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location='cpu'))
