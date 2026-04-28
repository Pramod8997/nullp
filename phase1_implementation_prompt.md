# Phase-1 Complete Implementation Prompt
## Project: Confidence-Aware Digital Twin-Driven Meta-RL for Few-Shot Smart Home Energy Management

---

## 0. How to use this prompt

Paste this entire document into a new Claude conversation (or give it to a developer).
It is self-contained. Every gap is described with: what exists, what is missing, exact
file paths, technical specification, and integration instructions.
Work through the gaps in the order listed — later gaps depend on earlier ones.

---

## 1. Project overview and context

**Title:** Confidence-Aware Digital Twin-Driven Meta-RL for Few-Shot Smart Home Energy Management
**Repo:** https://github.com/Pramod8997/nullp
**Stack:** Python 3.10, PyTorch, FastAPI, React (Vite), aiomqtt (amqtt broker), aiosqlite

**What this system does (full pipeline, 9 steps):**

```
Raw power (1 Hz, 10 appliance classes)
  → Step 1  Safety Monitor         [parallel, highest priority]
  → Step 2  Watchdog (z-score)     [sensor drift detection]
  → Step 3  SG filter + transient  [±50 W derivative threshold]
  → Step 4  1D-CNN encoder         [5 conv layers → 128-dim]
  → Step 5  Temporal Attention     [weight high-variance timestamps]
  → Step 6  ProtoNet + OpenMax     [known vs unknown classification]
  → Step 7  Temperature Scaling    [calibrate confidence score]
  → Step 8  Confidence Gate ≥0.90  [gate RL; route unknowns to delta stability]
  → Step 9a (known)  Digital Twin → Policy Promotion Gate → Q-Learning action
  → Step 9b (unknown, stable) → Label Request Router → Prototype Registry update
  → Step 9c (unknown, transient) → Temporary Anomaly Logger
  → Dashboard broadcast via WebSocket
```

**Current repo structure:**
```
nullp/
├── config/config.yaml
├── scripts/
│   ├── run_pipeline.py          ← main 9-step orchestrator (MODIFY)
│   ├── start_broker.py
│   ├── train_models.py          ← REPLACE with episodic meta-training
│   └── generate_mock_ukdale.py  ← EXTEND to 10 appliance classes
├── src/
│   ├── api/main.py              ← ADD label-route endpoint
│   ├── database/session.py
│   ├── hardware/mqtt.py
│   ├── models/
│   │   ├── protonet.py          ← MAJOR REWRITE
│   │   └── thermodynamics.py   ← VERIFY / FIX full PMV
│   ├── pipeline/
│   │   ├── safety.py
│   │   ├── watchdog.py
│   │   ├── phantom_tracker.py
│   │   ├── analytics.py
│   │   ├── classifier.py
│   │   ├── aggregate_nilm.py    ← VERIFY SG + derivative
│   │   └── failure_matrix.py
│   └── rl/agent.py              ← ADD policy promotion gate + confidence gate
├── backend/
│   ├── scripts/simulate_esp32.py ← EXTEND to 10 appliances
│   └── models/weights/
├── frontend/src/components/
│   └── DigitalTwin.jsx          ← ENSURE unknown-device labeling UI works
├── data/
│   ├── uk_dale.py               ← IMPLEMENT (or realistic synthetic)
│   ├── redd.py
│   └── synd.py
└── tests/
    ├── test_pipeline.py         ← ADD new component tests
    └── test_api.py
```

**Phase scope:** Phase 1 = 100% software. No real ESP32 hardware. The Tier-0 safety
relay is simulated in software (safety.py already exists). Phase 2 will deploy to
real ESP32 — make the code ESP32-deployment-ready but do not require it now.

---

## 2. Gap inventory and implementation specs

### GAP 1 — Savitzky-Golay filter + derivative transient detection
**File:** `src/pipeline/aggregate_nilm.py`
**Problem:** The report requires SG smoothing followed by a derivative-based transient
detector with ±50 W threshold over a 5-second window. The current file description
is "NILM step-change event detector" with no detail on the algorithm used.

**Required implementation:**

```python
# src/pipeline/aggregate_nilm.py  — REPLACE CONTENTS with:

import numpy as np
from scipy.signal import savgol_filter

WINDOW_SIZE = 5          # seconds (matches 1 Hz data → 5 samples)
SG_WINDOW   = 7          # Savitzky-Golay window (must be odd, > WINDOW_SIZE)
SG_POLYORD  = 2          # polynomial order
TRANSIENT_THRESHOLD_W = 50.0  # ±50 W triggers event

class NILMTransientDetector:
    """
    Step 1: smooth signal with Savitzky-Golay filter.
    Step 2: compute first derivative (np.diff).
    Step 3: flag transient if |derivative| >= TRANSIENT_THRESHOLD_W within
            any 5-sample rolling window.
    Returns the windowed signal segment centred on the transient peak for
    downstream CNN encoding.
    """

    def __init__(self, window_size=WINDOW_SIZE, sg_window=SG_WINDOW,
                 sg_polyord=SG_POLYORD, threshold=TRANSIENT_THRESHOLD_W,
                 embed_window=128):
        self.window_size = window_size
        self.sg_window   = sg_window
        self.sg_polyord  = sg_polyord
        self.threshold   = threshold
        self.embed_window = embed_window   # samples fed to CNN
        self._buffer     = []              # rolling 1 Hz power samples

    def push(self, power_w: float):
        """
        Push one 1 Hz sample. Returns (is_transient, segment_array | None).
        segment_array is shape (embed_window,), zero-padded if near buffer edge.
        """
        self._buffer.append(power_w)
        # Keep only what we need; trim to 3× embed_window for efficiency
        if len(self._buffer) > self.embed_window * 3:
            self._buffer = self._buffer[-(self.embed_window * 3):]

        if len(self._buffer) < self.sg_window:
            return False, None

        arr = np.array(self._buffer, dtype=np.float32)
        smoothed = savgol_filter(arr, self.sg_window, self.sg_polyord)
        deriv    = np.diff(smoothed)

        # Check last WINDOW_SIZE derivatives
        recent = deriv[-self.window_size:]
        if np.any(np.abs(recent) >= self.threshold):
            peak_idx = len(self._buffer) - 1
            half     = self.embed_window // 2
            start    = max(0, peak_idx - half)
            end      = start + self.embed_window
            segment  = arr[start:end]
            # zero-pad if near edges
            if len(segment) < self.embed_window:
                segment = np.pad(segment,
                                 (0, self.embed_window - len(segment)),
                                 mode='constant')
            return True, segment

        return False, None

    def reset(self):
        self._buffer.clear()
```

**Integration:** In `scripts/run_pipeline.py`, replace any existing step-change call
with `NILMTransientDetector.push(power_w)`. Feed returned segment to CNN encoder.

---

### GAP 2 — ProtoNet: Temporal Attention + OpenMax + Weibull EVT + Prototype Registry
**File:** `src/models/protonet.py`
**Problem:** The current file has a CNN encoder and SupportSetManager but is missing:
- Temporal Attention Module (Mishra et al., ICLR 2018)
- OpenMax Weibull EVT for open-set probability
- Proper incremental Prototype Registry

**Full replacement spec:**

```python
# src/models/protonet.py  — FULL REWRITE

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import exponweib   # Weibull fit

EMBED_DIM    = 128
CNN_CHANNELS = [32, 64, 128, 128, 128]   # 5 conv layers
KERNEL_SIZE  = 3
DROPOUT      = 0.1
WEIBULL_TAIL = 20      # top-k distances used to fit Weibull tail
OPENMAX_ALPHA= 10      # number of top classes revised by OpenMax

# ── 1. 1D-CNN Encoder ──────────────────────────────────────────────────────

class CNN1DEncoder(nn.Module):
    """
    5-layer 1D-CNN mapping (batch, 1, 128) → (batch, EMBED_DIM).
    Each block: Conv1d → BatchNorm → ReLU → MaxPool.
    """
    def __init__(self, in_channels=1, embed_dim=EMBED_DIM):
        super().__init__()
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
        self.cnn = nn.Sequential(*layers)
        # adaptive pool to fixed size then project to embed_dim
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.project = nn.Linear(CNN_CHANNELS[-1], embed_dim)

    def forward(self, x):
        # x: (batch, 128) → unsqueeze channel dim
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (batch, 1, 128)
        h = self.cnn(x)                 # (batch, 128, L')
        h = self.pool(h).squeeze(-1)    # (batch, 128)
        return self.project(h)          # (batch, EMBED_DIM)


# ── 2. Temporal Attention Module ───────────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    Computes a soft weight vector over the 128 time-steps.
    High-variance (informative) timesteps receive higher weight.
    Applied BEFORE CNN encoding on the raw segment.
    Reference: Mishra et al., ICLR 2018 — Simple Neural Attentive Meta-Learner.
    """
    def __init__(self, seq_len=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(seq_len, seq_len // 4),
            nn.Tanh(),
            nn.Linear(seq_len // 4, seq_len),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        """x: (batch, 128) raw segment"""
        weights = self.attn(x)          # (batch, 128)
        return x * weights              # element-wise weighted signal


# ── 3. Full ProtoNet (Encoder + Attention) ─────────────────────────────────

class ProtoNet(nn.Module):
    def __init__(self, seq_len=128, embed_dim=EMBED_DIM):
        super().__init__()
        self.attention = TemporalAttention(seq_len)
        self.encoder   = CNN1DEncoder(embed_dim=embed_dim)

    def embed(self, x):
        """x: (batch, 128) → (batch, EMBED_DIM)"""
        x = self.attention(x)
        return self.encoder(x)

    def forward(self, support, query):
        """
        Episodic forward pass.
        support: (N, K, 128)  N classes, K support samples each
        query:   (Q, 128)     Q query samples
        Returns: (Q, N) log-probabilities over classes
        """
        N, K, L = support.shape
        Q        = query.shape[0]

        sup_emb  = self.embed(support.view(N * K, L)).view(N, K, -1)
        prototypes = sup_emb.mean(dim=1)          # (N, EMBED_DIM)

        q_emb    = self.embed(query)              # (Q, EMBED_DIM)

        # Squared Euclidean distances
        dists = torch.cdist(q_emb.unsqueeze(0),
                            prototypes.unsqueeze(0)).squeeze(0)   # (Q, N)
        return F.log_softmax(-dists, dim=1), dists


# ── 4. OpenMax + Weibull EVT ───────────────────────────────────────────────

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

    def __init__(self, num_classes: int, tail_size: int = WEIBULL_TAIL,
                 unknown_threshold: float = 0.5, alpha: int = OPENMAX_ALPHA):
        self.num_classes       = num_classes
        self.tail_size         = tail_size
        self.unknown_threshold = unknown_threshold
        self.alpha             = alpha
        # weibull params per class: (shape, loc, scale)
        self._weibull = {}

    def fit(self, class_idx: int, distances: np.ndarray):
        """
        Fit Weibull to the tail (largest distances) of a class.
        Call once per class after episodic training is complete.
        distances: 1-D array of dist(query_embedding, prototype_c) for correct
                   support samples seen during training.
        """
        tail = np.sort(distances)[-self.tail_size:]
        # exponweib: generalized Weibull; fix loc at 0 for EVT tail
        params = exponweib.fit(tail, floc=0)
        self._weibull[class_idx] = params

    def _weibull_cdf(self, class_idx: int, distance: float) -> float:
        if class_idx not in self._weibull:
            return 0.0
        return float(exponweib.cdf(distance, *self._weibull[class_idx]))

    def predict(self, distances: np.ndarray, softmax_probs: np.ndarray):
        """
        distances:     (N,) distance from query to each prototype
        softmax_probs: (N,) raw softmax over known classes

        Returns:
            revised_probs: (N+1,) where index N = unknown class
            is_unknown: bool — True if P(unknown) > unknown_threshold
        """
        N = self.num_classes
        # Rank classes by softmax probability (descending), apply OpenMax to top-alpha
        ranked = np.argsort(-softmax_probs)

        # Compute Weibull CDF for each of the top-alpha classes
        psi = np.zeros(N)
        for rank, cls in enumerate(ranked[:self.alpha]):
            psi[cls] = self._weibull_cdf(cls, distances[cls])

        # Revise softmax probs: p̂_c = p_c × (1 - psi_c)
        revised = softmax_probs * (1.0 - psi)

        # Unknown probability = sum of redistributed mass
        p_unknown = float(np.sum(softmax_probs * psi))

        # Normalise all N+1 probs
        full = np.append(revised, p_unknown)
        full = full / (full.sum() + 1e-12)

        is_unknown = full[-1] > self.unknown_threshold
        return full, is_unknown

    def save(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self._weibull, f)

    def load(self, path: str):
        import pickle
        with open(path, 'rb') as f:
            self._weibull = pickle.load(f)


# ── 5. Prototype Registry (incremental, no encoder retraining) ────────────

class PrototypeRegistry:
    """
    Stores per-class prototype vectors.
    Supports incremental addition of new classes from as few as 5 samples
    without retraining the CNN encoder — the encoder is frozen.
    """

    def __init__(self, encoder: ProtoNet, device='cpu'):
        self.encoder    = encoder
        self.device     = device
        self.prototypes = {}    # {class_name: (prototype_tensor, sample_count)}
        self.encoder.eval()

    @torch.no_grad()
    def add_class(self, class_name: str, support_segments: np.ndarray):
        """
        support_segments: (K, 128) numpy array, K ≥ 1
        Computes prototype mean and stores/updates it.
        """
        x   = torch.tensor(support_segments, dtype=torch.float32).to(self.device)
        emb = self.encoder.embed(x)              # (K, EMBED_DIM)
        proto = emb.mean(dim=0)                  # (EMBED_DIM,)

        if class_name in self.prototypes:
            # Running mean update
            old_proto, old_n = self.prototypes[class_name]
            new_n    = old_n + len(support_segments)
            new_proto = (old_proto * old_n + proto * len(support_segments)) / new_n
            self.prototypes[class_name] = (new_proto, new_n)
        else:
            self.prototypes[class_name] = (proto, len(support_segments))

    @torch.no_grad()
    def classify(self, segment: np.ndarray):
        """
        segment: (128,) numpy array
        Returns: (class_name, distance, distances_to_all)
        """
        if not self.prototypes:
            return None, float('inf'), {}

        x   = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(self.device)
        emb = self.encoder.embed(x).squeeze(0)  # (EMBED_DIM,)

        dists = {}
        for name, (proto, _) in self.prototypes.items():
            dists[name] = float(torch.dist(emb, proto).item() ** 2)

        best = min(dists, key=dists.get)
        return best, dists[best], dists

    def class_names(self):
        return list(self.prototypes.keys())

    def save(self, path: str):
        torch.save({k: (v[0].cpu(), v[1]) for k, v in self.prototypes.items()}, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.prototypes = {k: (v[0].to(self.device), v[1]) for k, v in data.items()}
```

---

### GAP 3 — Temperature Scaling (confidence calibration)
**New file:** `src/models/calibration.py`
**Problem:** Missing entirely. Without it the "≥0.90 confidence" threshold has no
mathematical meaning (raw softmax is systematically overconfident).

```python
# src/models/calibration.py  — NEW FILE

import torch
import torch.nn as nn
import numpy as np

class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling (Guo et al., ICML 2017).
    Trains a single scalar T on a held-out calibration set.
    At inference: calibrated_prob = softmax(logits / T).
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits / self.temperature.clamp(min=0.05), dim=-1)

    def calibrate(self, logits: np.ndarray, labels: np.ndarray,
                  lr=0.01, max_iter=500):
        """
        Fit T using NLL on calibration set.
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
        """
        with torch.no_grad():
            lg     = torch.tensor(logits, dtype=torch.float32)
            probs  = self(lg).numpy()
        return probs, float(probs.max())

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location='cpu'))
        self.eval()
```

**Integration notes:**
- After episodic training is complete, hold out 15% of UK-DALE / synthetic data as
  calibration set. Collect (logits, true_labels) on this set. Call `scaler.calibrate()`.
- In the pipeline (Step 7), after ProtoNet inference: call
  `probs, conf = scaler.calibrated_confidence(logits)`.
- The confidence gate (Step 8) then checks `conf >= 0.90`.

---

### GAP 4 — Confidence Gate wired into RL + Delta Stability Analyzer
**Files:** `src/pipeline/delta_stability.py` (NEW), `scripts/run_pipeline.py` (MODIFY),
           `src/rl/agent.py` (MODIFY)

**4a. Delta Stability Analyzer (new file):**

```python
# src/pipeline/delta_stability.py  — NEW FILE

import numpy as np
from collections import deque

STABILITY_WINDOW  = 10      # number of recent unknown signatures to compare
STABILITY_THRESH  = 15.0    # squared Euclidean distance for "same device"
MIN_STABLE_COUNT  = 3       # must appear ≥ 3 times to be flagged as stable unknown

class DeltaStabilityAnalyzer:
    """
    Implements DFD Level-2 Process P4.3 (Signature Buffer D4.1).

    When ProtoNet/OpenMax routes a signature as UNKNOWN, this class
    determines whether the unknown is:
      - STABLE:    same signature appearing repeatedly → route to user labeling
      - TRANSIENT: one-off noise spike             → log silently (P4.4)

    This prevents alert fatigue from transient electrical noise.
    """

    def __init__(self, window=STABILITY_WINDOW, threshold=STABILITY_THRESH,
                 min_count=MIN_STABLE_COUNT):
        self.window     = window
        self.threshold  = threshold
        self.min_count  = min_count
        self._buffer    = deque(maxlen=window)   # Signature Buffer D4.1
        self._temp_log  = []                     # Temporary Anomaly Log P4.4

    def push(self, embedding: np.ndarray, timestamp: float = None):
        """
        embedding: (EMBED_DIM,) numpy array of the unknown signature.
        Returns: ('stable', cluster_mean) | ('transient', None)
        """
        self._buffer.append((embedding.copy(), timestamp))

        if len(self._buffer) < self.min_count:
            self._temp_log.append({'embedding': embedding, 'ts': timestamp,
                                   'reason': 'buffer_too_small'})
            return 'transient', None

        # Count how many buffered signatures are within threshold of this one
        embeddings = np.stack([e for e, _ in self._buffer])
        dists = np.sum((embeddings - embedding) ** 2, axis=1)
        close_count = int(np.sum(dists <= self.threshold))

        if close_count >= self.min_count:
            cluster_mean = embeddings[dists <= self.threshold].mean(axis=0)
            return 'stable', cluster_mean
        else:
            self._temp_log.append({'embedding': embedding, 'ts': timestamp,
                                   'reason': 'isolated_transient'})
            return 'transient', None

    def recent_log(self, n=10):
        return self._temp_log[-n:]

    def reset(self):
        self._buffer.clear()
```

**4b. Confidence Gate + pipeline wiring (pseudocode for run_pipeline.py):**

In `scripts/run_pipeline.py`, after Step 7 (Temperature Scaling produces `conf`):

```python
# ── Step 8: Confidence Gate ─────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.90   # from report FR3 / SRS NF-Accuracy

if is_unknown:
    # Route to DeltaStabilityAnalyzer
    stability, cluster_mean = delta_analyzer.push(embedding, timestamp=now())
    if stability == 'stable':
        # Route P4.5 — emit label-request event to dashboard via MQTT/WebSocket
        await mqtt.publish('home/unknown_device/label_request', {
            'embedding': cluster_mean.tolist(),
            'power_trace': segment.tolist(),
            'distance_to_nearest_prototype': min_dist,
        })
    else:
        # Log silently P4.4 — no dashboard alert
        db.log_transient_anomaly(embedding, timestamp)

elif conf >= CONFIDENCE_THRESHOLD:
    # Known device with sufficient confidence → allow RL
    await rl_agent.step(device_class, conf, twin_state)

else:
    # Known class but confidence too low → block RL, log
    db.log_low_confidence_event(device_class, conf)
    # Optionally trigger soft warning on dashboard
```

**4c. RL agent confidence gate (agent.py modification):**

Add a guard at the start of `QLearningAgent.step()`:
```python
def step(self, device_class, confidence, twin_state, min_confidence=0.90):
    if confidence < min_confidence:
        # Do NOT act — return no-op action
        return 'no_op', 0.0
    # ... existing step logic continues
```

---

### GAP 5 — Episodic Meta-Learning Training Loop
**File:** `scripts/train_models.py`  — REPLACE with episodic version

```python
# scripts/train_models.py  — FULL REWRITE

"""
Episodic N-way K-shot meta-training for ProtoNet on UK-DALE / synthetic data.

Episode structure (matches report Section 1.5 and FR2):
  N = 5   (number of classes per episode)
  K = 5   (support samples per class, "5-shot")
  Q = 10  (query samples per class per episode)
  Total episodes: 10,000 for full training, 1,000 for quick validation

After training:
  1. Fit Weibull tails via OpenMaxWeibull.fit()
  2. Calibrate temperature T via TemperatureScaler.calibrate()
  3. Save all weights to backend/models/weights/
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
from src.models.protonet   import ProtoNet, OpenMaxWeibull, PrototypeRegistry
from src.models.calibration import TemperatureScaler
from data.synd              import SyntheticUKDALE   # see GAP 6 for this class

# ── Hyper-parameters ────────────────────────────────────────────────────────
N_WAY        = 5
K_SHOT       = 5
Q_QUERY      = 10
N_EPISODES   = 10_000
LR           = 1e-3
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
WEIGHTS_DIR  = 'backend/models/weights'
SEQ_LEN      = 128
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ── Episode Sampler ─────────────────────────────────────────────────────────
def sample_episode(dataset, n_way, k_shot, q_query):
    """
    dataset: dict {class_name: np.ndarray (M, SEQ_LEN)} with M ≥ k+q samples
    Returns: support (n_way, k_shot, SEQ_LEN), query (n_way*q_query, SEQ_LEN),
             query_labels (n_way*q_query,)
    """
    classes = random.sample(list(dataset.keys()), n_way)
    support_list, query_list, label_list = [], [], []

    for idx, cls in enumerate(classes):
        samples = dataset[cls]
        chosen  = np.random.choice(len(samples), k_shot + q_query, replace=False)
        support_list.append(samples[chosen[:k_shot]])
        query_list.append(samples[chosen[k_shot:]])
        label_list.extend([idx] * q_query)

    support = np.stack(support_list)          # (N, K, SEQ_LEN)
    query   = np.concatenate(query_list)      # (N*Q, SEQ_LEN)
    labels  = np.array(label_list)            # (N*Q,)
    return support, query, labels

# ── Training ────────────────────────────────────────────────────────────────
def train():
    print("[Meta-Training] Loading dataset...")
    dataset = SyntheticUKDALE(seq_len=SEQ_LEN).load_all_classes()

    model   = ProtoNet(seq_len=SEQ_LEN).to(DEVICE)
    optim   = torch.optim.Adam(model.parameters(), lr=LR)
    nll     = nn.NLLLoss()

    print(f"[Meta-Training] Starting {N_EPISODES} episodes "
          f"({N_WAY}-way {K_SHOT}-shot)...")

    for ep in range(1, N_EPISODES + 1):
        support_np, query_np, labels_np = sample_episode(
            dataset, N_WAY, K_SHOT, Q_QUERY)

        support = torch.tensor(support_np, dtype=torch.float32).to(DEVICE)
        query   = torch.tensor(query_np,   dtype=torch.float32).to(DEVICE)
        labels  = torch.tensor(labels_np,  dtype=torch.long).to(DEVICE)

        log_probs, _ = model(support, query)
        loss         = nll(log_probs, labels)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if ep % 500 == 0:
            acc = (log_probs.argmax(dim=1) == labels).float().mean().item()
            print(f"  Episode {ep:5d} | loss {loss.item():.4f} | acc {acc:.3f}")

    # ── Save encoder ──
    torch.save(model.state_dict(), f'{WEIGHTS_DIR}/protonet.pt')
    print("[Meta-Training] ProtoNet saved.")

    # ── Build Prototype Registry on ALL classes ──
    print("[Meta-Training] Building Prototype Registry...")
    registry = PrototypeRegistry(model, device=DEVICE)
    for cls_name, segments in dataset.items():
        support_segs = segments[:K_SHOT]
        registry.add_class(cls_name, support_segs)
    registry.save(f'{WEIGHTS_DIR}/prototype_registry.pt')
    print("[Meta-Training] Prototype Registry saved.")

    # ── Fit Weibull tails (OpenMax) ──
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

    # ── Temperature Scaling calibration ──
    print("[Meta-Training] Calibrating temperature scaling...")
    scaler  = TemperatureScaler()
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for ep in range(200):    # 200 calibration episodes
            sup_np, q_np, lbl_np = sample_episode(dataset, N_WAY, K_SHOT, Q_QUERY)
            sup = torch.tensor(sup_np, dtype=torch.float32).to(DEVICE)
            q   = torch.tensor(q_np,   dtype=torch.float32).to(DEVICE)
            log_p, dists = model(sup, q)
            # logits = -dists (before softmax)
            logits_ep = -dists.cpu().numpy()
            all_logits.append(logits_ep)
            all_labels.append(lbl_np)

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    scaler.calibrate(all_logits, all_labels)
    scaler.save(f'{WEIGHTS_DIR}/temperature_scaler.pt')
    print(f"[Meta-Training] Temperature T = {scaler.temperature.item():.4f} saved.")

    print("\n[Meta-Training] ALL DONE.")
    print(f"  Weights saved to: {WEIGHTS_DIR}/")

if __name__ == '__main__':
    train()
```

---

### GAP 6 — 10 Appliance Classes (data + simulator)
**Files:** `data/synd.py` (IMPLEMENT), `backend/scripts/simulate_esp32.py` (EXTEND),
           `config/config.yaml` (EXTEND)

**6a. SyntheticUKDALE data generator (`data/synd.py`):**

Replace the stub with a class that generates realistic power signatures for
all 10 appliance classes used in the report. Each class has a characteristic
watt level and transient shape derived from UK-DALE typical values.

```python
# data/synd.py  — IMPLEMENT

import numpy as np

# 10 appliance classes with (steady_W, transient_peak_W, noise_std)
APPLIANCE_PROFILES = {
    'fridge':          (150,  300,  10),
    'hvac':            (1500, 2200, 80),
    'kettle':          (2200, 2400, 30),
    'tv':              (120,  180,  8),
    'washing_machine': (500,  2000, 100),
    'dishwasher':      (1800, 2000, 60),
    'microwave':       (1200, 1300, 20),
    'oven':            (2000, 2200, 50),
    'ev_charger':      (3300, 3500, 40),
    'laptop':          (60,   90,   5),
}

SEQ_LEN   = 128
SAMPLES_PER_CLASS = 500   # ≥ K_SHOT + Q_QUERY + calibration margin

class SyntheticUKDALE:
    """
    Generates realistic 1 Hz transient power signatures for 10 appliance classes.
    Each segment is SEQ_LEN samples centred on a simulated turn-on transient.
    The signature shape matches UK-DALE 1 Hz profiles described in Kelly &
    Knottenbelt (2015).
    """

    def __init__(self, seq_len=SEQ_LEN, n_samples=SAMPLES_PER_CLASS, seed=42):
        self.seq_len   = seq_len
        self.n_samples = n_samples
        np.random.seed(seed)

    def _make_segment(self, steady_w, peak_w, noise_std):
        seg = np.zeros(self.seq_len, dtype=np.float32)
        half = self.seq_len // 2
        # Pre-transient: near-zero with noise
        seg[:half] = np.random.normal(5, noise_std * 0.1, half)
        # Transient: exponential rise to steady state
        t = np.arange(self.seq_len - half)
        decay = peak_w * np.exp(-t / (self.seq_len * 0.15))
        steady = np.full(self.seq_len - half, steady_w)
        seg[half:] = np.maximum(steady, decay) + np.random.normal(0, noise_std,
                                                                    self.seq_len - half)
        return np.clip(seg, 0, 4000)

    def load_all_classes(self):
        """Returns dict {class_name: np.ndarray (n_samples, seq_len)}"""
        dataset = {}
        for name, (steady, peak, noise) in APPLIANCE_PROFILES.items():
            segs = np.stack([self._make_segment(steady, peak, noise)
                             for _ in range(self.n_samples)])
            dataset[name] = segs
        return dataset
```

**6b. Extend ESP32 simulator to 10 devices:**
In `backend/scripts/simulate_esp32.py`, replace the 4-device dict with all 10
appliances from `APPLIANCE_PROFILES` above. Publish each device's power reading
to `home/power/{device_name}` at 1 Hz. Add random ON/OFF toggling to simulate
realistic duty cycles per appliance (e.g. fridge cycles every 15–30 min).

**6c. config.yaml extensions:**
```yaml
appliances:
  - fridge
  - hvac
  - kettle
  - tv
  - washing_machine
  - dishwasher
  - microwave
  - oven
  - ev_charger
  - laptop

system_safety:
  fridge:          { max_w: 300,  tier0: false }
  hvac:            { max_w: 2500, tier0: false }
  kettle:          { max_w: 2500, tier0: false }
  tv:              { max_w: 300,  tier0: false }
  washing_machine: { max_w: 2200, tier0: false }
  dishwasher:      { max_w: 2200, tier0: false }
  microwave:       { max_w: 1500, tier0: false }
  oven:            { max_w: 2500, tier0: false }
  ev_charger:      { max_w: 3800, tier0: false }
  laptop:          { max_w: 150,  tier0: false }
  # medical_device, alarm_system → tier0: true (not sheddable)
```

---

### GAP 7 — Full PMV Calculation (ISO 7730 Category A)
**File:** `src/models/thermodynamics.py`
**Problem:** Currently "simplified." The report requires 6-variable PMV enforced
in [−0.5, +0.5] for Category A.

```python
# src/models/thermodynamics.py  — VERIFY / REPLACE with this implementation

import math

class PMVThermodynamics:
    """
    ISO 7730 Predicted Mean Vote (PMV) calculation.
    Category A: PMV in [-0.5, +0.5].

    Variables:
      ta   – air temperature (°C)
      tr   – mean radiant temperature (°C)
      va   – air velocity (m/s)
      rh   – relative humidity (%)
      clo  – clothing insulation (clo units; typical: 0.5 summer, 1.0 winter)
      met  – metabolic rate (met; seated=1.0, light activity=1.6)
    """

    CATEGORY_A_MIN = -0.5
    CATEGORY_A_MAX =  0.5

    def pmv(self, ta=22.0, tr=22.0, va=0.1, rh=50.0,
            clo=0.5, met=1.2) -> float:
        """
        Returns PMV value. Raises ValueError if inputs are out of plausible range.
        Implementation follows ISO 7730:2005 Annex D.
        """
        # Clothing surface area factor
        if clo < 0.5:
            f_cl = 1.0 + 1.290 * clo
        else:
            f_cl = 1.05 + 0.645 * clo

        # Metabolic heat production (W/m²)
        m = met * 58.15

        # Clothing thermal resistance (m²K/W)
        i_cl = 0.155 * clo

        # Pa: partial water vapour pressure (Pa)
        pa = rh * 10 * math.exp(16.6536 - 4030.183 / (ta + 235.0))

        # Clothing surface temperature (iterative)
        t_cl = ta   # initial guess
        for _ in range(150):
            hc   = max(2.38 * abs(t_cl - ta) ** 0.25, 12.1 * math.sqrt(va))
            t_cl_new = 35.7 - 0.028 * (m - 0) - i_cl * (
                3.96e-8 * f_cl * ((t_cl + 273)**4 - (tr + 273)**4)
                + f_cl * hc * (t_cl - ta))
            if abs(t_cl_new - t_cl) < 0.001:
                t_cl = t_cl_new
                break
            t_cl = t_cl_new

        hc = max(2.38 * abs(t_cl - ta) ** 0.25, 12.1 * math.sqrt(va))

        # Thermal load L (W/m²)
        L = (m - 3.05e-3 * (5733 - 6.99 * m - pa)
               - 0.42 * (m - 58.15)
               - 1.7e-5 * m * (5867 - pa)
               - 0.0014 * m * (34 - ta)
               - 3.96e-8 * f_cl * ((t_cl + 273)**4 - (tr + 273)**4)
               - f_cl * hc * (t_cl - ta))

        pmv_val = (0.303 * math.exp(-0.036 * m) + 0.028) * L
        return round(pmv_val, 4)

    def is_category_a(self, pmv_val: float) -> bool:
        return self.CATEGORY_A_MIN <= pmv_val <= self.CATEGORY_A_MAX

    def pmv_penalty(self, pmv_val: float) -> float:
        """
        Reward function penalty for RL agent.
        0.0 if in Category A, proportional to violation otherwise.
        """
        if self.is_category_a(pmv_val):
            return 0.0
        return abs(pmv_val) - 0.5    # penalty = distance outside bounds

    def comfort_state(self, ta=22.0, tr=22.0, va=0.1, rh=50.0,
                      clo=0.5, met=1.2) -> dict:
        pmv_val = self.pmv(ta, tr, va, rh, clo, met)
        return {
            'pmv': pmv_val,
            'category_a': self.is_category_a(pmv_val),
            'penalty': self.pmv_penalty(pmv_val),
        }
```

---

### GAP 8 — Policy Promotion Gate
**File:** `src/rl/agent.py` — ADD promotion check

The control flow diagram requires Q-learning policies to be validated in the
digital twin sandbox before any relay command is issued live.

```python
# Add to src/rl/agent.py inside QLearningAgent class

class PolicyPromotionGate:
    """
    Tracks validation episodes run in the digital twin.
    A policy is 'promoted' to live relay control only after completing
    MIN_VALIDATION_EPISODES without a PMV violation or unsafe action.
    """
    MIN_VALIDATION_EPISODES = 50
    PMV_PENALTY_LIMIT       = 0.5    # max allowed cumulative PMV penalty

    def __init__(self):
        self._val_episodes   = 0
        self._cumulative_pmv = 0.0
        self._promoted       = False

    def record_twin_episode(self, pmv_penalty: float):
        self._val_episodes   += 1
        self._cumulative_pmv += pmv_penalty

    @property
    def is_promoted(self) -> bool:
        if self._promoted:
            return True
        if (self._val_episodes >= self.MIN_VALIDATION_EPISODES
                and self._cumulative_pmv <= self.PMV_PENALTY_LIMIT):
            self._promoted = True
        return self._promoted

    def reset(self):
        self._val_episodes   = 0
        self._cumulative_pmv = 0.0
        self._promoted       = False

# In the pipeline, before issuing relay action:
# if not promotion_gate.is_promoted:
#     run_digital_twin_episode(agent, twin)
#     # do NOT issue real relay command
# else:
#     issue_relay_command(action)
```

---

### GAP 9 — Label Request Router (API endpoint)
**File:** `src/api/main.py` — ADD endpoint

When the DeltaStabilityAnalyzer returns 'stable', the backend emits a
label-request to the dashboard. The dashboard must send the user's label
back to the backend, which updates the PrototypeRegistry.

```python
# ADD to src/api/main.py

from pydantic import BaseModel

class DeviceLabelRequest(BaseModel):
    class_name: str
    segments:   list[list[float]]   # list of (128,) float arrays

@app.post("/api/label_device")
async def label_device(req: DeviceLabelRequest):
    """
    Called by dashboard when user labels an unknown device.
    Updates prototype registry without retraining encoder.
    """
    segs = np.array(req.segments, dtype=np.float32)  # (K, 128)
    if segs.shape[-1] != 128:
        raise HTTPException(400, "Each segment must be 128 samples")

    registry.add_class(req.class_name, segs)
    registry.save(f'{WEIGHTS_DIR}/prototype_registry.pt')

    return {"status": "ok",
            "class_name": req.class_name,
            "total_classes": len(registry.class_names())}

@app.get("/api/unknown_devices")
async def unknown_devices():
    """Returns list of pending unknown device signatures needing labels."""
    return {"pending": pending_unknowns_store}   # maintained by pipeline
```

In `frontend/src/components/DigitalTwin.jsx`, ensure the "unknown device prompt"
card has a text input for class name and a submit button that POSTs to
`/api/label_device` with the stored embedding as `segments`.

---

### GAP 10 — Pipeline orchestrator: wire all components together
**File:** `scripts/run_pipeline.py` — MODIFY to integrate all new components

At the top of the pipeline, initialise all new components:
```python
from src.pipeline.aggregate_nilm import NILMTransientDetector
from src.models.protonet         import ProtoNet, OpenMaxWeibull, PrototypeRegistry
from src.models.calibration      import TemperatureScaler
from src.pipeline.delta_stability import DeltaStabilityAnalyzer
from src.models.thermodynamics    import PMVThermodynamics
from src.rl.agent                 import QLearningAgent, PolicyPromotionGate

# Load trained weights
model    = ProtoNet(); model.load_state_dict(torch.load('backend/models/weights/protonet.pt'))
registry = PrototypeRegistry(model); registry.load('backend/models/weights/prototype_registry.pt')
openmax  = OpenMaxWeibull(num_classes=10); openmax.load('backend/models/weights/openmax_weibull.pkl')
scaler   = TemperatureScaler(); scaler.load('backend/models/weights/temperature_scaler.pt')

detector       = NILMTransientDetector()
delta_analyzer = DeltaStabilityAnalyzer()
pmv_model      = PMVThermodynamics()
rl_agent       = QLearningAgent(...)
promo_gate     = PolicyPromotionGate()
```

Then in the async main loop (per 1 Hz power sample per device):
```
1.  Safety check (existing safety.py — unchanged)
2.  Watchdog z-score (existing watchdog.py — unchanged)
3.  detector.push(power_w)  → if transient: segment
4.  model.embed(segment)    → embedding
5.  model.attention already applied inside embed()
6.  registry.classify(segment) → class, dist, all_dists
7.  openmax.predict(dists_arr, softmax_probs) → probs, is_unknown
8.  scaler.calibrated_confidence(logits)  → calibrated_probs, conf
9.  Confidence Gate:
      if is_unknown:
          delta_analyzer.push(embedding) → ('stable'|'transient')
      elif conf >= 0.90:
          if promo_gate.is_promoted:
              rl_agent.step(class, conf, twin_state)  → action → relay
          else:
              run_twin_episode; promo_gate.record_twin_episode(pmv_penalty)
      else:
          log low-confidence event; no action
10. Analytics, DB, broadcast (existing — unchanged)
```

---

## 3. Testing additions

Add the following test cases to `tests/test_pipeline.py`:

```python
# Tests to add

def test_sg_transient_detection():
    d = NILMTransientDetector()
    # Push steady signal then a 100W spike
    for _ in range(20): d.push(100.0)
    is_t, seg = d.push(250.0)   # 150W delta > 50W threshold
    assert is_t
    assert seg.shape == (128,)

def test_openmax_unknown():
    omw = OpenMaxWeibull(num_classes=3)
    for i in range(3):
        omw.fit(i, np.random.exponential(10, 100))
    far_dists   = np.array([500.0, 500.0, 500.0])   # far from all
    near_probs  = np.array([0.33,  0.33,  0.34])
    probs, is_unknown = omw.predict(far_dists, near_probs)
    assert is_unknown

def test_temperature_scaling_reduces_confidence():
    scaler = TemperatureScaler()
    # Artificially high T → lower confidence
    scaler.temperature.data = torch.tensor([3.0])
    logits = np.array([[10.0, 0.1, 0.1]])
    _, conf = scaler.calibrated_confidence(logits)
    assert conf < 0.99   # should be reduced from near-1 raw softmax

def test_delta_stability_stable():
    da = DeltaStabilityAnalyzer(min_count=3)
    emb = np.ones(128) * 5.0
    for _ in range(5):
        result, _ = da.push(emb + np.random.normal(0, 0.5, 128))
    assert result == 'stable'

def test_delta_stability_transient():
    da = DeltaStabilityAnalyzer(min_count=3)
    for i in range(5):
        result, _ = da.push(np.random.normal(i * 100, 1, 128))
    assert result == 'transient'

def test_pmv_category_a():
    thermo = PMVThermodynamics()
    pmv = thermo.pmv(ta=22, tr=22, va=0.1, rh=50, clo=0.5, met=1.2)
    assert -0.5 <= pmv <= 0.5

def test_pmv_hot_violation():
    thermo = PMVThermodynamics()
    pmv = thermo.pmv(ta=35, tr=35, va=0.0, rh=80, clo=1.0, met=2.0)
    assert pmv > 0.5

def test_confidence_gate_blocks_rl(mocker):
    agent = QLearningAgent(...)
    action, reward = agent.step('kettle', confidence=0.75, twin_state={})
    assert action == 'no_op'

def test_policy_not_promoted_initially():
    gate = PolicyPromotionGate()
    assert not gate.is_promoted

def test_policy_promotes_after_validation():
    gate = PolicyPromotionGate()
    for _ in range(50):
        gate.record_twin_episode(pmv_penalty=0.0)
    assert gate.is_promoted
```

---

## 4. Dependency additions (requirements.txt)

Add these to `requirements.txt` if not already present:
```
scipy>=1.11.0          # savgol_filter, exponweib Weibull fit
nilmtk>=0.4.0          # optional but specified in SRS Section 3.5
torch>=2.0.0
numpy>=1.24.0
```

---

## 5. Implementation order (do in this sequence)

```
1.  GAP 6   — data/synd.py (10 classes) + simulator + config
2.  GAP 1   — aggregate_nilm.py (SG + transient)
3.  GAP 2   — protonet.py (full rewrite: attention + openmax + registry)
4.  GAP 3   — calibration.py (temperature scaling)
5.  GAP 5a  — train_models.py (episodic training)
              → run: python scripts/train_models.py
              → this produces all weights in backend/models/weights/
6.  GAP 4   — delta_stability.py + pipeline wiring + rl confidence gate
7.  GAP 7   — thermodynamics.py (full PMV)
8.  GAP 8   — policy_promotion_gate in agent.py
9.  GAP 9   — api/main.py label endpoint + DigitalTwin.jsx UI
10. GAP 10  — run_pipeline.py: wire all 10 components together
11. Tests   — add new test cases and verify make test passes
```

---

## 6. Acceptance criteria for Phase 1 completion

All of the following must pass before Phase 1 is considered complete:

| # | Criterion | Check |
|---|-----------|-------|
| 1 | `make test` passes all 35+ tests (27 existing + 9 new) | `pytest tests/` |
| 2 | ProtoNet 5-shot accuracy ≥ 85% on 10 synthetic classes | `train_models.py` output |
| 3 | Temperature scalar T stored in weights; calibration ECE < 0.05 | logged during training |
| 4 | OpenMax correctly rejects appliance embeddings with distance > 3σ from prototypes | `test_openmax_unknown` |
| 5 | Confidence gate blocks RL when conf < 0.90 | `test_confidence_gate_blocks_rl` |
| 6 | DeltaStability returns 'stable' for repeated unknown; 'transient' for random noise | `test_delta_stability_*` |
| 7 | PMV stays in [−0.5, +0.5] ≥ 95% of simulation steps | Q-learning evaluation loop |
| 8 | Dashboard shows label-request card for stable unknown devices | manual UI check |
| 9 | Label submitted via dashboard adds new prototype to registry without restarting pipeline | POST /api/label_device |
| 10| Policy not applied to relay until ≥ 50 twin validation episodes | `test_policy_promotes_after_validation` |
| 11| All 10 appliance classes visible in device fleet panel | dashboard check |
| 12| `make run` starts cleanly with no import errors | full smoke test |

---

## 7. What is intentionally deferred to Phase 2

Do NOT implement these in Phase 1. Make sure code is structured to accept them:
- Real ESP32 C++ firmware (firmware/esp32_node/src stays as skeleton)
- Real UK-DALE HDF5 loading via NILMTK (uk_dale.py stays as a well-documented stub)
- Physical CT clamp + relay (safety.py simulates this in software)
- Live MQTT stream from real hardware (simulator covers this)
- DQN upgrade for > 15 appliances (tabular Q-table is fine for 10)

---

*End of Phase-1 implementation prompt.*
