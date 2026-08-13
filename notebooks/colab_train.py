# ============================================================
# EMS ProtoNet — Google Colab Training Script (v3 PRODUCTION)
# Runtime → Change runtime type → T4 GPU (or better)
# ============================================================

# ── CELL 0: HDF5 blosc/lz4 codec support (Colab-specific) ────────────
import os, sys, subprocess
# UK-DALE HDF5 uses blosc compression (from NILMTK/PyTables).
# hdf5plugin bundles the actual decompressor binaries (blosc, lz4, zstd)
# and registers them with h5py at import time — no directory hacks needed.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'hdf5plugin', 'tables'], check=False)

# ── CELL 1: Install / imports ─────────────────────────────────
import os, sys, json, math, random, zipfile, pickle, shutil, warnings
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from scipy.signal import savgol_filter
from scipy.stats import exponweib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU : {torch.cuda.get_device_name(0)}')

for d in ['data/raw', 'weights', 'results']:
    os.makedirs(d, exist_ok=True)

# ── CELL 2: UK-DALE download (aria2c — 16 parallel connections) ──
from tqdm.auto import tqdm

UKDALE_URL     = 'https://zenodo.org/records/13917372/files/ukdale.h5?download=1'
H5_PATH        = 'data/raw/ukdale.h5'
UKDALE_MIN_SIZE = 3_000_000_000  # ~3.3 GB — reject truncated files

def download_aria2c(url, dest, desc='ukdale.h5'):
    """Download using aria2c with 16 parallel connections (10-50x faster)."""
    # Check if file exists AND is large enough (not truncated)
    if os.path.exists(dest):
        actual = os.path.getsize(dest)
        if actual >= UKDALE_MIN_SIZE:
            print(f'Already downloaded: {dest} ({actual/1e9:.2f} GB)')
            return True
        else:
            print(f'⚠️  Truncated file detected: {actual/1e6:.0f} MB (expected ≥{UKDALE_MIN_SIZE/1e9:.1f} GB) — re-downloading')
            os.remove(dest)
    dest_dir = os.path.dirname(dest) or '.'
    dest_name = os.path.basename(dest)
    # Install aria2 if not present (Colab/Ubuntu)
    if subprocess.run(['which', 'aria2c'], capture_output=True).returncode != 0:
        print('Installing aria2c...')
        subprocess.run(['apt-get', 'install', '-y', '-qq', 'aria2'], check=False)
    cmd = [
        'aria2c',
        '-x', '16',          # 16 connections per server
        '-s', '16',           # split file into 16 segments
        '-k', '1M',           # minimum split size 1MB
        '--file-allocation=none',  # don't pre-allocate (faster start)
        '--console-log-level=warn',
        '-d', dest_dir,
        '-o', dest_name,
        url,
    ]
    print(f'⬇️  Downloading {desc} via aria2c (16 parallel connections)...')
    result = subprocess.run(cmd)
    if result.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) >= UKDALE_MIN_SIZE:
        size_mb = os.path.getsize(dest) / 1e6
        print(f'✅ Downloaded {desc} ({size_mb:.0f} MB)')
        return True
    print(f'⚠️  aria2c failed (code {result.returncode}), falling back to requests...')
    return False

def download_requests_fallback(url, dest, desc=''):
    """Fallback: single-stream download via requests."""
    import requests
    if os.path.exists(dest): os.remove(dest)  # remove any partial
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(dest, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=desc) as bar:
            for chunk in r.iter_content(65536):
                f.write(chunk); bar.update(len(chunk))
        return True
    except Exception as e:
        print(f'Download failed: {e}'); return False

if not download_aria2c(UKDALE_URL, H5_PATH):
    download_requests_fallback(UKDALE_URL, H5_PATH, 'ukdale.h5')

# ── CELL 3: UK-DALE Parser ──────────────────────────────
import hdf5plugin  # registers blosc/lz4/zstd filters BEFORE h5py uses them
import h5py
SEQ_LEN          = 128
TRANSIENT_THRESH = 20.0  # Lowered to catch more events
SG_WIN, SG_POLY  = 7, 2

UKDALE_METER_LABELS = {
    'building1': {2:'fridge', 5:'washing_machine', 6:'dishwasher', 8:'microwave', 10:'kettle', 11:'laptop', 19:'oven'},
    'building2': {2:'washing_machine', 3:'dishwasher', 4:'tv', 5:'microwave', 6:'kettle', 8:'hvac'},
    'building3': {2:'hvac', 3:'washing_machine', 4:'kettle', 5:'microwave', 6:'laptop', 7:'tv'},
    'building5': {2:'washing_machine', 3:'dishwasher', 4:'fridge', 5:'kettle', 6:'tv', 7:'microwave'},
}

def detect_transients(signal, thresh=TRANSIENT_THRESH):
    signal = np.nan_to_num(np.asarray(signal, dtype=np.float32))
    if len(signal) < SEQ_LEN: return []
    above = signal > thresh
    transitions = np.diff(above.astype(int))
    on_events = np.where(transitions == 1)[0]
    half = SEQ_LEN // 2
    segs = []
    for idx in on_events:
        start = max(0, idx - half)
        end = start + SEQ_LEN
        if end > len(signal):
            start = len(signal) - SEQ_LEN
            end = len(signal)
        if start < 0: continue
        seg = signal[start:end].copy()
        if len(seg) == SEQ_LEN and np.max(seg) > thresh:
            segs.append(seg)
    return segs

def load_ukdale(path=H5_PATH):
    dataset = defaultdict(list)
    if not os.path.exists(path): return dataset
    try:
        with h5py.File(path, 'r') as f:
            for bld, bld_map in UKDALE_METER_LABELS.items():
                if bld not in f or 'elec' not in f[bld]: continue
                elec = f[bld]['elec']
                for meter_key in elec.keys():
                    try: meter_num = int(meter_key.replace('meter', ''))
                    except ValueError: continue
                    label = bld_map.get(meter_num)
                    if not label or 'table' not in elec[meter_key]: continue
                    tbl = elec[meter_key]['table']
                    if not hasattr(tbl, 'shape') or len(tbl.shape) == 0: continue
                    raw = tbl[:]
                    # PyTables stores compound dtype: ('index','values_block_0')
                    if raw.dtype.names and 'values_block_0' in raw.dtype.names:
                        power = raw['values_block_0'].flatten().astype(np.float32)
                    else:
                        power = np.array(raw, dtype=np.float32).flatten()
                    segs = detect_transients(power)
                    if segs: dataset[label].extend(segs)
    except Exception as e: print(f"Error reading UK-DALE: {e}")
    return dataset

print('\n=== Loading UK-DALE ===')
ukdale_data = load_ukdale()
print(f'UK-DALE totals: { {k:len(v) for k,v in ukdale_data.items()} }')

# ── CELL 4: Improved Synthetic Generator (v2 from synd.py) ────────
APPLIANCE_PROFILES = {
    'fridge':          (150,  300,  10,  False, False),
    'hvac':            (1500, 2200, 80,  True,  False),
    'kettle':          (2200, 2400, 30,  False, False),
    'tv':              (120,  180,  8,   False, False),
    'washing_machine': (500,  2000, 100, True,  True),
    'dishwasher':      (1800, 2000, 60,  True,  True),
    'microwave':       (1200, 1300, 20,  False, False),
    'oven':            (2000, 2200, 50,  False, False),
    'ev_charger':      (3300, 3500, 40,  False, False),
    'laptop':          (60,   90,   5,   False, False),
}

class SyntheticUKDALE:
    def __init__(self, seq_len=SEQ_LEN, seed=42):
        self.seq_len = seq_len
        self._rng = np.random.default_rng(seed)

    def _add_harmonics(self, seg):
        t = np.arange(len(seg), dtype=np.float32)
        for harm in [3, 5, 7]:
            phase = self._rng.uniform(0, 2 * np.pi)
            freq = harm * (50.0 / len(seg))
            amp = 0.05 * np.mean(np.abs(seg)) / harm
            seg += amp * np.sin(2 * np.pi * freq * t + phase)
        return seg

    def _add_brownout(self, seg):
        if self._rng.random() < 0.15:
            sag_start = self._rng.integers(0, max(1, len(seg) - 10))
            sag_len = self._rng.integers(5, min(20, len(seg) - sag_start))
            seg[sag_start:sag_start + sag_len] *= self._rng.uniform(0.7, 0.9)
        return seg

    def _add_crosstalk(self, seg):
        bg_freq = self._rng.uniform(0.01, 0.1)
        bg_phase = self._rng.uniform(0, 2 * np.pi)
        bg_amp = self._rng.uniform(3.0, 15.0)
        t = np.arange(len(seg), dtype=np.float32)
        seg += bg_amp * np.sin(2 * np.pi * bg_freq * t + bg_phase)
        if self._rng.random() < 0.15:
            seg[self._rng.integers(0, len(seg)):] += self._rng.uniform(-10, 20)
        return seg

    def _add_sensor_noise(self, seg):
        seg = np.round(seg / 3.0) * 3.0  # ADC quant
        t = np.arange(len(seg), dtype=np.float32) / len(seg)
        seg += self._rng.uniform(-2.0, 2.0) * t # drift
        return seg

    def _make_multi_state(self, steady_w, peak_w, noise_std):
        seg = np.zeros(self.seq_len, dtype=np.float32)
        onset = int(self.seq_len * self._rng.uniform(0.2, 0.5))
        seg[:onset] = self._rng.normal(self._rng.uniform(2.0, 8.0), noise_std * 0.1, onset)
        
        remaining = self.seq_len - onset
        n_states = self._rng.integers(2, 5)
        state_lengths = np.round(self._rng.dirichlet(np.ones(n_states)) * remaining).astype(int)
        state_lengths[-1] = remaining - state_lengths[:-1].sum()

        pos = onset
        for i, slen in enumerate(state_lengths):
            if slen <= 0: continue
            actual_len = min(int(slen), self.seq_len - pos)
            if actual_len <= 0: break
            
            pwr = self._rng.uniform(peak_w * 0.8, peak_w * 1.1) if i == 0 else self._rng.uniform(steady_w * 0.3, steady_w * 1.2)
            seg[pos:pos + actual_len] = self._rng.normal(pwr, noise_std, actual_len)
            pos += actual_len
        return seg

    def make_segment(self, cls):
        steady, peak, noise, has_harm, is_multi = APPLIANCE_PROFILES[cls]
        if is_multi:
            seg = self._make_multi_state(steady, peak, noise)
        else:
            seg = np.zeros(self.seq_len, dtype=np.float32)
            onset = int(self.seq_len * self._rng.uniform(0.2, 0.5))
            seg[:onset] = self._rng.normal(self._rng.uniform(2.0, 15.0), noise * 0.2, onset)
            
            post_len = self.seq_len - onset
            t = np.arange(post_len, dtype=np.float32)
            decay = peak * self._rng.uniform(0.9, 1.1) * np.exp(-t / (self.seq_len * self._rng.uniform(0.08, 0.25)))
            steady_arr = np.full(post_len, steady * self._rng.uniform(0.85, 1.15))
            seg[onset:] = np.maximum(steady_arr, decay) + self._rng.normal(0, noise, post_len)

        if has_harm: seg = self._add_harmonics(seg)
        seg = self._add_brownout(seg)
        seg = self._add_crosstalk(seg)
        seg = self._add_sensor_noise(seg)
        return np.clip(seg, 0, 4000).astype(np.float32)

ALL_CLASSES = list(APPLIANCE_PROFILES.keys())
TARGET_SAMPLES = 600

dataset = {}
synth_gen = SyntheticUKDALE()
for cls in ALL_CLASSES:
    real = list(ukdale_data.get(cls, []))
    n_real = len(real)
    n_synth = max(0, TARGET_SAMPLES - n_real)
    synth = [synth_gen.make_segment(cls) for _ in range(n_synth)]
    
    segs = real + synth
    random.shuffle(segs)
    # Ensure min samples, trim to max
    dataset[cls] = np.array(segs[:TARGET_SAMPLES], dtype=np.float32)
    print(f'{cls:20s}: {n_real:4d} real + {n_synth:4d} synth = {len(dataset[cls])} total')

# Normalize globally by 4000W to match local behavior (or max per class)
for cls in dataset:
    arr = dataset[cls]
    dataset[cls] = arr / np.maximum(arr.max(axis=1, keepdims=True), 1.0)


# ── CELL 5: EXACT Local ProtoNet Architecture ────────────────────────
EMBED_DIM = 128

class CNN1DEncoder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=EMBED_DIM):
        super().__init__()
        layers = []
        ch_in = in_channels
        for ch_out in [32, 64, 128, 128]:
            layers += [
                nn.Conv1d(ch_in, ch_out, 5, padding=2),
                nn.BatchNorm1d(ch_out),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.1),
            ]
            ch_in = ch_out
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.project = nn.Linear(128, embed_dim)
        self.project_bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        h = self.cnn(x)
        h = self.pool(h).squeeze(-1)
        h = self.project(h)
        return self.project_bn(h)

class PreCNNTemporalAttention(nn.Module):
    def __init__(self, seq_len=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(seq_len, seq_len // 4),
            nn.Tanh(),
            nn.Linear(seq_len // 4, seq_len),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.attn(x)

class ProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = PreCNNTemporalAttention(SEQ_LEN)
        self.encoder = CNN1DEncoder(embed_dim=EMBED_DIM)

    def embed(self, x):
        return self.encoder(self.attention(x))

    def forward(self, support, query):
        N, K, L = support.shape
        sup_emb = self.embed(support.view(N * K, L)).view(N, K, -1)
        prototypes = sup_emb.mean(dim=1)
        q_emb = self.embed(query)
        dists = torch.cdist(q_emb.unsqueeze(0), prototypes.unsqueeze(0)).pow(2).squeeze(0)
        return F.log_softmax(-dists, dim=1), dists

model = ProtoNet().to(DEVICE)
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')


# ── CELL 6: Episodic training ─────────────────────────────────
N_WAY, K_SHOT, Q_QUERY = 5, 5, 10
N_EPISODES  = 12000
LR          = 1e-3

def sample_episode(ds, n_way, k_shot, q_query):
    classes = random.sample(list(ds.keys()), n_way)
    sup, qry, lbl = [], [], []
    for i, cls in enumerate(classes):
        arr  = ds[cls]
        need = k_shot + q_query
        idx  = np.random.choice(len(arr), need, replace=len(arr) < need)
        sup.append(arr[idx[:k_shot]]); qry.append(arr[idx[k_shot:]]); lbl.extend([i] * q_query)
    return np.stack(sup), np.concatenate(qry), np.array(lbl)

optim = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, N_EPISODES)
nll = nn.NLLLoss()

losses, accs = [], []
model.train()

pbar = tqdm(range(1, N_EPISODES + 1), desc='Training')
for ep in pbar:
    sup_np, q_np, lbl_np = sample_episode(dataset, N_WAY, K_SHOT, Q_QUERY)
    sup = torch.tensor(sup_np, dtype=torch.float32).to(DEVICE)
    qry = torch.tensor(q_np,   dtype=torch.float32).to(DEVICE)
    lbl = torch.tensor(lbl_np, dtype=torch.long).to(DEVICE)

    optim.zero_grad()
    log_p, _ = model(sup, qry)
    loss = nll(log_p, lbl)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optim.step()
    sched.step()

    acc = (log_p.argmax(1) == lbl).float().mean().item()
    losses.append(loss.item()); accs.append(acc)

    if ep % 500 == 0:
        pbar.set_postfix(loss=f'{np.mean(losses[-500:]):.4f}', acc=f'{np.mean(accs[-500:]):.3f}')

# ── CELL 7: Prototype Registry & Weibull (OpenMax) ─────────────────────
model.eval()
registry = {}
weibull = {}
with torch.no_grad():
    for idx, cls in enumerate(ALL_CLASSES):
        # 10 samples for prototype
        arr = dataset[cls]
        x_proto = torch.tensor(arr[:10], dtype=torch.float32).to(DEVICE)
        registry[cls] = (model.embed(x_proto).mean(0).cpu(), 10)
        
        # 50 samples for weibull fitting
        x_weib = torch.tensor(arr[:50], dtype=torch.float32).to(DEVICE)
        emb = model.embed(x_weib).cpu()
        dists = ((emb - registry[cls][0]) ** 2).sum(1).numpy()
        tail = np.sort(dists)[-20:]
        if len(tail) < 2: tail = np.append(tail, tail[-1] + 1e-5)
        weibull[idx] = exponweib.fit(tail, floc=0)

# ── CELL 8: Temperature Scaling ─────────────────────────────
class TScaler(nn.Module):
    def __init__(self): super().__init__(); self.temperature = nn.Parameter(torch.ones(1))

ts = TScaler()
all_logits, all_labels = [], []
with torch.no_grad():
    for _ in range(200):
        sup_np, q_np, lbl_np = sample_episode(dataset, N_WAY, K_SHOT, Q_QUERY)
        sup = torch.tensor(sup_np, dtype=torch.float32).to(DEVICE)
        qry = torch.tensor(q_np,   dtype=torch.float32).to(DEVICE)
        _, dists = model(sup, qry)
        all_logits.append((-dists).cpu()); all_labels.append(torch.tensor(lbl_np))

lg, lb = torch.cat(all_logits), torch.cat(all_labels)
opt_ts = torch.optim.LBFGS([ts.temperature], lr=0.01, max_iter=500)
def _eval():
    opt_ts.zero_grad()
    l = nn.CrossEntropyLoss()(lg / ts.temperature.clamp(min=0.05), lb)
    l.backward(); return l
opt_ts.step(_eval)
print(f'Temperature T = {ts.temperature.item():.4f}')

# ── CELL 9: Evaluation on held-out test set ─────────────────
y_true, y_pred = [], []
proto_stack = torch.stack([registry[c][0] for c in ALL_CLASSES]).to(DEVICE)

with torch.no_grad():
    for cls_idx, cls in enumerate(ALL_CLASSES):
        arr = dataset[cls]
        test_arr = arr[int(len(arr) * 0.8):]  # last 20%
        if len(test_arr) == 0: continue
        x = torch.tensor(test_arr, dtype=torch.float32).to(DEVICE)
        emb = model.embed(x)
        dists = torch.cdist(emb.unsqueeze(0), proto_stack.unsqueeze(0)).pow(2).squeeze(0)
        preds = dists.argmin(1).cpu().numpy()
        y_true.extend([cls_idx] * len(test_arr))
        y_pred.extend(preds.tolist())

print('\n' + classification_report(y_true, y_pred, target_names=ALL_CLASSES, digits=3))

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay(cm, display_labels=ALL_CLASSES).plot(ax=ax, cmap='Blues', colorbar=False)
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150)

report_dict = {
    'overall_accuracy': round(accuracy_score(y_true, y_pred), 4),
    'macro_f1':         round(f1_score(y_true, y_pred, average='macro'), 4),
    'per_class_f1': dict(zip(ALL_CLASSES, [round(x, 4) for x in f1_score(y_true, y_pred, average=None).tolist()])),
    'temperature_T': round(ts.temperature.item(), 4),
}
with open('results/training_report.json', 'w') as f: json.dump(report_dict, f, indent=2)

# ── CELL 10: Save weights & package ──────────────────────────
torch.save(model.state_dict(), 'weights/protonet.pt')
torch.save(registry, 'weights/prototype_registry.pt')
with open('weights/openmax_weibull.pkl', 'wb') as f: pickle.dump({'weibull': weibull}, f)
torch.save(ts.state_dict(), 'weights/temperature_scaler.pt')

with zipfile.ZipFile('ems_weights.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir('weights'): zf.write(f'weights/{f}', f)
    for f in os.listdir('results'): zf.write(f'results/{f}', f)

# Try Google Drive first (only works in Colab notebook, not script mode)
saved = False
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    shutil.copy('ems_weights.zip', '/content/drive/MyDrive/ems_weights.zip')
    print('✅ Saved to Google Drive: /content/drive/MyDrive/ems_weights.zip')
    saved = True
except Exception:
    print('ℹ️  Google Drive not available (normal when running as a script)')

# Try Hugging Face Hub upload
if not saved:
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj='ems_weights.zip',
            path_in_repo='ems_weights.zip',
            repo_id=os.environ.get('HF_REPO_ID', ''),
            repo_type='model',
        )
        print('✅ Uploaded to Hugging Face Hub')
        saved = True
    except Exception:
        pass

# Local fallback — guard against SameFileError when CWD is /content/
src = os.path.abspath('ems_weights.zip')
dst = os.path.abspath('/content/ems_weights.zip')
if src != dst:
    shutil.copy(src, dst)
    print(f'✅ Saved locally: {dst}')
else:
    print(f'✅ Weights already at: {src}')
print('   Download ems_weights.zip from the Colab sidebar or your CWD.')
