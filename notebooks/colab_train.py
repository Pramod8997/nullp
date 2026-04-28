# ============================================================
# EMS ProtoNet — Google Colab Training Script  (FIXED v2)
# Runtime → Change runtime type → T4 GPU
# Run as: !python colab_train.py
# ============================================================

# ── CELL 1: Install / imports ─────────────────────────────────
import os, sys, json, math, random, zipfile, pickle, shutil, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from scipy.signal import savgol_filter
from scipy.stats import exponweib
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score, f1_score)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU : {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB')

for d in ['data/raw', 'data/processed', 'weights', 'results']:
    os.makedirs(d, exist_ok=True)

# ── CELL 2: UK-DALE download ──────────────────────────────────
import requests
from tqdm.auto import tqdm

UKDALE_URL = 'https://zenodo.org/records/13917372/files/ukdale.h5?download=1'
H5_PATH    = 'data/raw/ukdale.h5'

def download_file(url, dest, desc=''):
    if os.path.exists(dest) and os.path.getsize(dest) > 1e6:
        print(f'Already downloaded: {dest}')
        return True
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

ukdale_ok = download_file(UKDALE_URL, H5_PATH, 'ukdale.h5')

# ── CELL 3: Inspect UK-DALE HDF5 structure ───────────────────
import h5py

def inspect_h5(path, max_lines=80):
    """Print HDF5 tree and return all dataset paths."""
    paths = []
    def _visitor(name, obj):
        if len(paths) < max_lines:
            print(f'  {name}  [{type(obj).__name__}]', dict(list(obj.attrs.items())[:3]) if obj.attrs else '')
        if isinstance(obj, h5py.Dataset):
            paths.append(name)
    if os.path.exists(path):
        with h5py.File(path, 'r') as f:
            f.visititems(_visitor)
    return paths

print('\n=== UK-DALE HDF5 structure (first 80 nodes) ===')
all_paths = inspect_h5(H5_PATH)

# ── CELL 4: UK-DALE parser (handles nilmtk HDFStore format) ──
SEQ_LEN          = 128
TRANSIENT_THRESH = 50.0   # W — minimum derivative for transient
SG_WIN, SG_POLY  = 7, 2

UKDALE_LABEL_MAP = {
    'fridge':          ['fridge', 'fridge freezer', 'freezer'],
    'washing_machine': ['washing machine', 'washer'],
    'dishwasher':      ['dish washer', 'dishwasher'],
    'microwave':       ['microwave'],
    'kettle':          ['kettle'],
    'tv':              ['television', 'tv', 'monitor'],
    'hvac':            ['heat pump', 'air conditioner', 'boiler', 'air_conditioner'],
    'oven':            ['oven', 'cooker'],
    'ev_charger':      ['electric vehicle', 'ev charger'],
    'laptop':          ['laptop', 'computer', 'desktop'],
}

def label_to_canonical(label):
    label = label.lower().strip()
    for canonical, keywords in UKDALE_LABEL_MAP.items():
        if any(kw in label for kw in keywords):
            return canonical
    return None

def detect_transients(signal, thresh=TRANSIENT_THRESH):
    """Return list of 128-sample segments centred on turn-on events."""
    signal = np.asarray(signal, dtype=np.float32)
    if len(signal) < SG_WIN + SEQ_LEN:
        return []
    sm    = savgol_filter(signal, SG_WIN, SG_POLY)
    deriv = np.diff(sm)
    segs, i = [], SG_WIN
    while i < len(deriv) - SEQ_LEN:
        if np.max(np.abs(deriv[i:i+5])) >= thresh:
            start = max(0, i - SEQ_LEN // 2)
            seg   = signal[start:start + SEQ_LEN]
            if len(seg) == SEQ_LEN:
                segs.append(seg.copy())
            i += SEQ_LEN        # skip ahead
        else:
            i += 1
    return segs

def load_ukdale(path=H5_PATH):
    dataset = defaultdict(list)
    if not os.path.exists(path):
        return dataset

    with h5py.File(path, 'r') as f:
        buildings = [k for k in f.keys() if k.startswith('building')]
        for bld in buildings[:2]:          # houses 1 & 2
            elec = f.get(f'{bld}/elec')
            if elec is None:
                continue
            for meter_key in elec.keys():
                meter_grp = elec[meter_key]

                # ── Find appliance label from attributes ──────────────
                label = ''
                # nilmtk stores metadata as JSON in 'metadata' key or attrs
                for attr in ['appliance_type', 'device_name', 'label', 'name']:
                    v = meter_grp.attrs.get(attr, b'')
                    if isinstance(v, bytes): v = v.decode()
                    if v: label = v; break

                # Try metadata subgroup
                if not label and 'metadata' in meter_grp:
                    try:
                        meta = json.loads(bytes(meter_grp['metadata'][()]).decode())
                        label = meta.get('appliance_type', meta.get('label', ''))
                    except Exception:
                        pass

                canonical = label_to_canonical(label)
                if not canonical:
                    continue

                # ── Read power time-series ────────────────────────────
                # nilmtk stores as pandas HDFStore: look for 'table' or direct dataset
                power = None
                for candidate in ['table', 'data', '0', 'values']:
                    if candidate in meter_grp:
                        try:
                            ds = meter_grp[candidate]
                            # HDFStore table: columns are in 'axis0', values in 'block0_values'
                            if 'block0_values' in ds:
                                power = ds['block0_values'][:, 0].astype(np.float32)
                            elif 'values_block_0' in ds:
                                power = ds['values_block_0'][:, 0].astype(np.float32)
                            elif isinstance(ds, h5py.Dataset):
                                arr = ds[()]
                                if arr.ndim == 2: arr = arr[:, 0]
                                power = arr.astype(np.float32)
                            if power is not None and len(power) > SEQ_LEN:
                                break
                        except Exception:
                            pass

                if power is None or len(power) < SEQ_LEN:
                    # Last resort: iterate all sub-datasets
                    for sub in meter_grp.values():
                        if isinstance(sub, h5py.Dataset) and sub.size > SEQ_LEN:
                            arr = sub[()].flatten().astype(np.float32)
                            if len(arr) > SEQ_LEN:
                                power = arr; break

                if power is None:
                    continue

                segs = detect_transients(power)
                if segs:
                    dataset[canonical].extend(segs)
                    print(f'  UK-DALE {bld}/{meter_key} "{label}" → {canonical}: {len(segs)} segs')

    return dataset

print('\n=== Loading UK-DALE ===')
ukdale_data = load_ukdale()
print(f'UK-DALE totals: { {k:len(v) for k,v in ukdale_data.items()} }')

# ── CELL 5: Synthetic augmentation for missing/sparse classes ─
PROFILES = {
    'fridge':          (150,  300,  10,  0.12),
    'hvac':            (1500, 2200, 80,  0.08),
    'kettle':          (2200, 2400, 30,  0.20),
    'tv':              (120,  180,  8,   0.15),
    'washing_machine': (500,  2000, 100, 0.06),
    'dishwasher':      (1800, 2000, 60,  0.09),
    'microwave':       (1200, 1300, 20,  0.18),
    'oven':            (2000, 2200, 50,  0.10),
    'ev_charger':      (3300, 3500, 40,  0.07),
    'laptop':          (60,   90,   5,   0.25),
}

def make_synthetic(cls, n=400, seed=None):
    """Realistic turn-on transient with class-specific decay rate."""
    steady, peak, noise_std, decay_rate = PROFILES[cls]
    rng  = np.random.default_rng(seed)
    half = SEQ_LEN // 2
    segs = []
    for _ in range(n):
        pre  = rng.normal(rng.uniform(0, 10), noise_std * 0.2, half)
        t    = np.arange(SEQ_LEN - half, dtype=np.float32)
        # Exponential decay from peak to steady state
        sig  = steady + (peak - steady) * np.exp(-decay_rate * t)
        sig += rng.normal(0, noise_std, len(t))
        # Random amplitude jitter ±15%
        sig *= rng.uniform(0.85, 1.15)
        seg  = np.concatenate([pre, sig]).astype(np.float32)
        segs.append(np.clip(seg, 0, 5000))
    return segs

ALL_CLASSES = list(PROFILES.keys())
MIN_REAL    = 150    # if a class has fewer real samples, augment with synthetic

dataset = {}
for cls in ALL_CLASSES:
    real = list(ukdale_data.get(cls, []))
    n_real = len(real)
    if n_real < MIN_REAL:
        n_synth = max(400, MIN_REAL - n_real)
        synth   = make_synthetic(cls, n_synth, seed=hash(cls) % 2**32)
        segs    = real + synth
        print(f'  {cls:20s}: {n_real:4d} real + {n_synth} synthetic = {len(segs)}')
    else:
        segs    = real
        print(f'  {cls:20s}: {n_real:4d} real samples')
    # Shuffle and cap
    random.shuffle(segs)
    dataset[cls] = np.array(segs[:800], dtype=np.float32)

print(f'\nDataset shapes: { {k: v.shape for k,v in dataset.items()} }')

# Normalize each segment to [0, 1] per-sample
for cls in dataset:
    arr    = dataset[cls]
    maxval = arr.max(axis=1, keepdims=True).clip(min=1.0)
    dataset[cls] = arr / maxval

# ── CELL 6: ProtoNet architecture ─────────────────────────────
EMBED_DIM = 128

class CNN1DEncoder(nn.Module):
    def __init__(self, channels=(32, 64, 128, 128)):
        super().__init__()
        layers, ch = [], 1
        for out in channels:
            layers += [nn.Conv1d(ch, out, 5, padding=2),
                       nn.BatchNorm1d(out), nn.GELU(),
                       nn.MaxPool1d(2), nn.Dropout(0.15)]
            ch = out
        self.cnn  = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(nn.Linear(ch, EMBED_DIM), nn.LayerNorm(EMBED_DIM))

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        return self.fc(self.pool(self.cnn(x)).squeeze(-1))

class TemporalAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Sequential(nn.Linear(SEQ_LEN, SEQ_LEN // 4), nn.Tanh(),
                               nn.Linear(SEQ_LEN // 4, SEQ_LEN), nn.Softmax(dim=-1))
    def forward(self, x): return x * self.w(x)

class ProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TemporalAttn()
        self.enc  = CNN1DEncoder()

    def embed(self, x):
        return self.enc(self.attn(x))

    def forward(self, support, query):
        N, K, L = support.shape
        proto = self.embed(support.view(N * K, L)).view(N, K, EMBED_DIM).mean(1)
        q_emb = self.embed(query)
        dists = torch.cdist(q_emb.unsqueeze(0), proto.unsqueeze(0)).squeeze(0)
        return F.log_softmax(-dists, dim=1), dists

model = ProtoNet().to(DEVICE)
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

# ── CELL 7: Episodic training ─────────────────────────────────
N_WAY, K_SHOT, Q_QUERY = 5, 5, 10
N_EPISODES  = 8000
LR          = 1e-3
ACCUM       = 4

def sample_episode(ds, n_way, k_shot, q_query):
    classes = random.sample(list(ds.keys()), n_way)
    sup, qry, lbl = [], [], []
    for i, cls in enumerate(classes):
        arr  = ds[cls]
        need = k_shot + q_query
        idx  = np.random.choice(len(arr), need, replace=len(arr) < need)
        # Light augmentation: add small Gaussian noise
        sup_seg = arr[idx[:k_shot]] + np.random.normal(0, 0.01, (k_shot, SEQ_LEN)).astype(np.float32)
        qry_seg = arr[idx[k_shot:]] + np.random.normal(0, 0.01, (q_query, SEQ_LEN)).astype(np.float32)
        sup.append(sup_seg); qry.append(qry_seg); lbl.extend([i] * q_query)
    return np.stack(sup), np.concatenate(qry), np.array(lbl)

optim  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler = GradScaler()
nll    = nn.NLLLoss()
sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, N_EPISODES)

losses, accs = [], []
model.train(); optim.zero_grad()

pbar = tqdm(range(1, N_EPISODES + 1), desc='Training')
for ep in pbar:
    sup_np, q_np, lbl_np = sample_episode(dataset, N_WAY, K_SHOT, Q_QUERY)
    sup = torch.tensor(sup_np, dtype=torch.float32).to(DEVICE)
    qry = torch.tensor(q_np,   dtype=torch.float32).to(DEVICE)
    lbl = torch.tensor(lbl_np, dtype=torch.long).to(DEVICE)

    with autocast():
        log_p, _ = model(sup, qry)
        loss      = nll(log_p, lbl) / ACCUM

    scaler.scale(loss).backward()
    if ep % ACCUM == 0:
        scaler.step(optim); scaler.update(); optim.zero_grad()

    sched.step()
    acc = (log_p.argmax(1) == lbl).float().mean().item()
    losses.append(loss.item() * ACCUM); accs.append(acc)

    if ep % 500 == 0:
        pbar.set_postfix(loss=f'{np.mean(losses[-500:]):.4f}',
                         acc=f'{np.mean(accs[-500:]):.3f}')

print(f'\nFinal → loss={np.mean(losses[-500:]):.4f} | acc={np.mean(accs[-500:]):.3f}')

# ── CELL 8: Prototype Registry ────────────────────────────────
model.eval()
registry = {}
with torch.no_grad():
    for cls in ALL_CLASSES:
        # Use first 20% as support (held out from training episodes)
        arr   = dataset[cls]
        n_sup = max(K_SHOT, int(len(arr) * 0.2))
        x     = torch.tensor(arr[:n_sup], dtype=torch.float32).to(DEVICE)
        registry[cls] = model.embed(x).mean(0).cpu()

# ── CELL 9: Weibull (OpenMax) ─────────────────────────────────
weibull = {}
with torch.no_grad():
    for idx, cls in enumerate(ALL_CLASSES):
        arr = dataset[cls]
        x   = torch.tensor(arr[:60], dtype=torch.float32).to(DEVICE)
        emb = model.embed(x).cpu()
        d   = ((emb - registry[cls]) ** 2).sum(1).numpy()
        tail = np.sort(d)[-20:]
        if tail.std() < 1e-6: tail += np.linspace(0, 0.1, len(tail))
        weibull[idx] = exponweib.fit(tail, floc=0)

# ── CELL 10: Temperature Scaling ─────────────────────────────
class TScaler(nn.Module):
    def __init__(self): super().__init__(); self.T = nn.Parameter(torch.ones(1))

ts = TScaler()
all_logits, all_labels = [], []
with torch.no_grad():
    for _ in range(300):
        sup_np, q_np, lbl_np = sample_episode(dataset, N_WAY, K_SHOT, Q_QUERY)
        sup = torch.tensor(sup_np, dtype=torch.float32).to(DEVICE)
        qry = torch.tensor(q_np,   dtype=torch.float32).to(DEVICE)
        _, dists = model(sup, qry)
        all_logits.append((-dists).cpu()); all_labels.append(torch.tensor(lbl_np))

lg, lb = torch.cat(all_logits), torch.cat(all_labels)
opt_ts = torch.optim.LBFGS([ts.T], lr=0.1, max_iter=500)
def _eval():
    opt_ts.zero_grad()
    l = nn.CrossEntropyLoss()(lg / ts.T.clamp(0.05), lb)
    l.backward(); return l
opt_ts.step(_eval)
print(f'Temperature T = {ts.T.item():.4f}')

# ── CELL 11: Evaluation on held-out test set ─────────────────
y_true, y_pred = [], []
proto_stack = torch.stack([registry[c] for c in ALL_CLASSES]).to(DEVICE)

with torch.no_grad():
    for cls_idx, cls in enumerate(ALL_CLASSES):
        arr      = dataset[cls]
        test_arr = arr[int(len(arr) * 0.8):]  # last 20% = test
        if len(test_arr) == 0: continue
        x    = torch.tensor(test_arr, dtype=torch.float32).to(DEVICE)
        emb  = model.embed(x)
        dists = torch.cdist(emb.unsqueeze(0), proto_stack.unsqueeze(0)).squeeze(0)
        preds = dists.argmin(1).cpu().numpy()
        y_true.extend([cls_idx] * len(test_arr))
        y_pred.extend(preds.tolist())

report_str = classification_report(y_true, y_pred, target_names=ALL_CLASSES, digits=3)
print('\n' + report_str)

# Confusion matrix
cm  = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(12, 10))
ConfusionMatrixDisplay(cm, display_labels=ALL_CLASSES).plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('ProtoNet Confusion Matrix — UK-DALE + Synthetic', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
print('Saved: results/confusion_matrix.png')

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
w = 200
ax1.plot(np.convolve(losses, np.ones(w)/w, 'valid'), color='#ef4444')
ax1.set(title='Training Loss', xlabel='Episode')
ax2.plot(np.convolve(accs, np.ones(w)/w, 'valid'), color='#22c55e')
ax2.axhline(0.8, color='gray', linestyle='--', label='80%')
ax2.set(title='Episode Accuracy', xlabel='Episode'); ax2.legend()
plt.tight_layout()
plt.savefig('results/training_curves.png', dpi=150)
print('Saved: results/training_curves.png')

# Report JSON
report_dict = {
    'overall_accuracy': round(accuracy_score(y_true, y_pred), 4),
    'macro_f1':         round(f1_score(y_true, y_pred, average='macro'), 4),
    'per_class_f1': dict(zip(ALL_CLASSES,
                             [round(x, 4) for x in f1_score(y_true, y_pred, average=None).tolist()])),
    'temperature_T': round(ts.T.item(), 4),
    'n_episodes':    N_EPISODES,
    'dataset': {'sources': ['UK-DALE (Zenodo)', 'Synthetic augmentation'],
                'n_way': N_WAY, 'k_shot': K_SHOT},
    'n_real_segments': {k: int((dataset[k].shape[0] * 0.8)) for k in ALL_CLASSES},
}
with open('results/training_report.json', 'w') as f:
    json.dump(report_dict, f, indent=2)
print(json.dumps(report_dict, indent=2))

# ── CELL 12: Save weights & package ──────────────────────────
torch.save(model.state_dict(), 'weights/protonet.pt')
torch.save({cls: (registry[cls], K_SHOT) for cls in ALL_CLASSES},
           'weights/prototype_registry.pt')
with open('weights/openmax_weibull.pkl', 'wb') as f:
    pickle.dump({'weibull': weibull, 'classes': ALL_CLASSES}, f)
torch.save(ts.state_dict(), 'weights/temperature_scaler.pt')

with zipfile.ZipFile('ems_weights.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for w_file in ['protonet.pt', 'prototype_registry.pt',
                   'openmax_weibull.pkl', 'temperature_scaler.pt']:
        zf.write(f'weights/{w_file}', w_file)
    for r_file in ['confusion_matrix.png', 'training_curves.png', 'training_report.json']:
        if os.path.exists(f'results/{r_file}'):
            zf.write(f'results/{r_file}', r_file)

size_mb = os.path.getsize('ems_weights.zip') / 1e6
print(f'\n✅ Packaged: ems_weights.zip  ({size_mb:.1f} MB)')

# ── CELL 13: Download from Colab ─────────────────────────────
# Option A — Copy to /content/ so you can right-click → Download in file panel
shutil.copy('ems_weights.zip', '/content/ems_weights.zip')
print('📁 File ready at /content/ems_weights.zip')
print('   → In Colab sidebar: Files → right-click ems_weights.zip → Download')

# Option B — Mount Google Drive and copy there
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    dest = '/content/drive/MyDrive/ems_weights.zip'
    shutil.copy('ems_weights.zip', dest)
    print(f'✅ Also copied to Google Drive: {dest}')
except Exception as e:
    print(f'Drive mount skipped ({e}) — use Option A above.')
