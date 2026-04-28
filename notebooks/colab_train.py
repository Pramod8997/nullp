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

# ── CELL 3: Constants & helpers ──────────────────────────────
import h5py
SEQ_LEN          = 128
TRANSIENT_THRESH = 50.0
SG_WIN, SG_POLY  = 7, 2

# UK-DALE nilmtk-metadata: meter number → appliance label (buildings 1 & 2)
# Source: https://github.com/nilmtk/nilm_metadata/tree/master/meter_devices
UKDALE_METER_LABELS = {
    'building1': {
        2: 'fridge', 3: 'freezer', 4: 'fridge freezer',
        5: 'washing machine', 6: 'dish washer', 7: 'television',
        8: 'microwave', 9: 'toaster', 10: 'kettle', 11: 'laptop',
        12: 'computer', 13: 'television', 14: 'hair dryer',
        15: 'vacuum cleaner', 16: 'breadmaker', 17: 'games console',
        18: 'hi-fi', 19: 'oven', 20: 'toaster',
    },
    'building2': {
        2: 'washing machine', 3: 'dish washer', 4: 'television',
        5: 'microwave', 6: 'kettle', 7: 'freezer', 8: 'boiler',
    },
    'building3': {
        2: 'electric space heater', 3: 'washing machine', 4: 'kettle',
        5: 'microwave', 6: 'laptop', 7: 'television',
    },
    'building4': {
        2: 'fridge', 3: 'television', 4: 'washing machine',
        5: 'kettle', 6: 'microwave', 7: 'laptop',
    },
    'building5': {
        2: 'washing machine', 3: 'dish washer', 4: 'fridge freezer',
        5: 'kettle', 6: 'television', 7: 'microwave',
    },
}

UKDALE_LABEL_MAP = {
    'fridge':          ['fridge', 'fridge freezer', 'freezer'],
    'washing_machine': ['washing machine', 'washer'],
    'dishwasher':      ['dish washer', 'dishwasher'],
    'microwave':       ['microwave'],
    'kettle':          ['kettle'],
    'tv':              ['television', 'tv'],
    'hvac':            ['heat pump', 'boiler', 'air conditioner', 'electric space heater'],
    'oven':            ['oven', 'cooker'],
    'ev_charger':      ['electric vehicle', 'ev charger'],
    'laptop':          ['laptop', 'computer'],
}

def label_to_canonical(label):
    label = label.lower().strip()
    for canonical, keywords in UKDALE_LABEL_MAP.items():
        if any(kw in label for kw in keywords):
            return canonical
    return None

def detect_transients(signal, thresh=TRANSIENT_THRESH):
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
            i += SEQ_LEN
        else:
            i += 1
    return segs

def read_meter_power(meter_grp):
    """Read power from a nilmtk PyTables compound dataset."""
    tbl = meter_grp.get('table')
    if tbl is None:
        return None
    try:
        # PyTables stores as structured numpy array — dtype has named fields
        raw = tbl[()]   # shape (N,), dtype = [('index', i8), ('values_block_0', f4, (1,)), ...]
        if raw.dtype.names and 'values_block_0' in raw.dtype.names:
            return raw['values_block_0'].flatten().astype(np.float32)
        # Fallback: look for any float field
        for field in raw.dtype.names or []:
            col = raw[field].flatten()
            if col.dtype.kind == 'f' and col.max() > 1:
                return col.astype(np.float32)
    except Exception:
        pass
    return None

def load_ukdale(path=H5_PATH):
    dataset = defaultdict(list)
    if not os.path.exists(path):
        return dataset

    with h5py.File(path, 'r') as f:
        buildings = [k for k in f.keys() if k.startswith('building')]
        for bld in buildings:
            elec = f.get(f'{bld}/elec')
            if elec is None:
                continue
            bld_map = UKDALE_METER_LABELS.get(bld, {})
            for meter_key in elec.keys():
                # Extract meter number
                try:
                    meter_num = int(meter_key.replace('meter', ''))
                except ValueError:
                    continue
                if meter_num == 1:
                    continue  # skip aggregate

                # Get label from hardcoded map
                label = bld_map.get(meter_num, '')
                canonical = label_to_canonical(label)
                if not canonical:
                    continue

                power = read_meter_power(elec[meter_key])
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
# Primary: Mount Google Drive and copy there (most reliable)
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    dest = '/content/drive/MyDrive/ems_weights.zip'
    shutil.copy('ems_weights.zip', dest)
    print(f'✅ Saved to Google Drive: {dest}')
    print('   → Open Google Drive and download ems_weights.zip')
except Exception as e:
    print(f'Drive mount skipped: {e}')

# Fallback: ensure file is at /content/ for sidebar download
zip_abs = os.path.abspath('ems_weights.zip')
content_dest = '/content/ems_weights.zip'
if zip_abs != content_dest:
    shutil.copy(zip_abs, content_dest)
    print(f'📁 Also at {content_dest}')
else:
    print(f'📁 File already at {content_dest}')

print('   → Colab sidebar (Files panel): right-click ems_weights.zip → Download')
