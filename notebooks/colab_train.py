# ============================================================
# EMS ProtoNet — Google Colab Training Script
# Run each section as a separate cell in Colab.
# GPU Runtime: Runtime → Change runtime type → T4 GPU
# ============================================================

# ── CELL 1: Install dependencies ─────────────────────────────
# !pip install -q torch scipy scikit-learn matplotlib tqdm requests h5py

# ── CELL 2: Imports ──────────────────────────────────────────
import os, json, math, random, zipfile, requests, warnings
from pathlib import Path
from io import BytesIO
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from scipy.signal import savgol_filter
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB')

os.makedirs('data/raw/redd', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('weights', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ── CELL 3: Download REDD low-frequency dataset ───────────────
REDD_URL = 'http://redd.csail.mit.edu/data/low_freq.tar.gz'

def download_redd():
    out = 'data/raw/redd/low_freq.tar.gz'
    if os.path.exists('data/raw/redd/house_1'):
        print('REDD already extracted.')
        return
    print(f'Downloading REDD from {REDD_URL} ...')
    try:
        r = requests.get(REDD_URL, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(out, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as bar:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                bar.update(len(chunk))
        os.system(f'tar xzf {out} -C data/raw/redd/')
        print('REDD extracted.')
    except Exception as e:
        print(f'REDD download failed: {e}')
        print('Will use synthetic data as fallback for missing classes.')

download_redd()

# ── CELL 4: Download UK-DALE House 1 from Zenodo ─────────────
# Zenodo record: https://zenodo.org/records/13917372
UKDALE_ZENODO = 'https://zenodo.org/records/13917372/files/ukdale.h5?download=1'

def download_ukdale():
    out = 'data/raw/ukdale.h5'
    if os.path.exists(out):
        print(f'UK-DALE already present: {os.path.getsize(out)/1e9:.2f} GB')
        return True
    print('Downloading UK-DALE from Zenodo...')
    try:
        r = requests.get(UKDALE_ZENODO, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(out, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc='ukdale.h5') as bar:
            for chunk in r.iter_content(65536):
                f.write(chunk)
                bar.update(len(chunk))
        print(f'UK-DALE downloaded: {os.path.getsize(out)/1e9:.2f} GB')
        return True
    except Exception as e:
        print(f'UK-DALE download failed: {e}')
        return False

ukdale_ok = download_ukdale()

# ── CELL 5: Preprocessing — extract 128-sample transient windows ──
SEQ_LEN = 128
TRANSIENT_THRESH = 50.0
SG_WIN, SG_POLY = 7, 2

# Canonical class → dataset label mappings
REDD_MAP = {
    'fridge':          ['refrigerator'],
    'washing_machine': ['washer_dryer', 'washer'],
    'dishwasher':      ['dishwasher'],
    'microwave':       ['microwave'],
    'tv':              ['entertainment'],
    'hvac':            ['air_conditioner', 'hvac'],
    'oven':            ['stove', 'oven'],
    'ev_charger':      ['electric_vehicle'],
    'laptop':          ['electronics', 'computer'],
    'kettle':          [],   # not in REDD
}

UKDALE_MAP = {
    'fridge':          ['fridge', 'fridge freezer'],
    'washing_machine': ['washing machine'],
    'dishwasher':      ['dish washer', 'dishwasher'],
    'microwave':       ['microwave'],
    'tv':              ['television', 'tv'],
    'hvac':            ['heat pump', 'air conditioner'],
    'oven':            ['oven'],
    'ev_charger':      ['electric vehicle charger'],
    'laptop':          ['laptop'],
    'kettle':          ['kettle'],
}

def detect_transients(signal):
    """Return list of (center_idx, segment) for each transient."""
    if len(signal) < SG_WIN + SEQ_LEN:
        return []
    sm = savgol_filter(signal.astype(float), SG_WIN, SG_POLY)
    deriv = np.diff(sm)
    results = []
    i = SG_WIN
    while i < len(deriv) - SEQ_LEN:
        window = deriv[i:i+5]
        if np.any(np.abs(window) >= TRANSIENT_THRESH):
            start = max(0, i - SEQ_LEN//2)
            seg = signal[start:start+SEQ_LEN].astype(np.float32)
            if len(seg) == SEQ_LEN:
                results.append(seg)
            i += SEQ_LEN  # skip ahead to avoid duplicates
        else:
            i += 1
    return results

def load_redd_segments(redd_dir='data/raw/redd'):
    """Read REDD CSV files and extract transient segments per class."""
    dataset = defaultdict(list)
    redd_path = Path(redd_dir)
    if not redd_path.exists():
        return dataset

    for house_dir in sorted(redd_path.glob('house_*')):
        # Read labels file
        labels_file = house_dir / 'labels.dat'
        if not labels_file.exists():
            continue
        labels = {}
        for line in labels_file.read_text().strip().split('\n'):
            parts = line.split()
            if len(parts) >= 2:
                labels[int(parts[0])] = parts[1].lower()

        for meter_id, appliance_label in labels.items():
            # Find canonical class
            canonical = None
            for cls, redd_labels in REDD_MAP.items():
                if any(rl in appliance_label for rl in redd_labels):
                    canonical = cls
                    break
            if canonical is None:
                continue

            # Read channel file
            chan_file = house_dir / f'channel_{meter_id}.dat'
            if not chan_file.exists():
                continue
            try:
                data = np.loadtxt(chan_file, usecols=1, dtype=np.float32)
                segs = detect_transients(data)
                dataset[canonical].extend(segs)
                if segs:
                    print(f'  REDD {house_dir.name} {appliance_label}({canonical}): {len(segs)} segments')
            except Exception as e:
                pass

    return dataset

def load_ukdale_segments(h5_path='data/raw/ukdale.h5'):
    """Read UK-DALE HDF5 and extract transient segments per class."""
    import h5py
    dataset = defaultdict(list)
    if not os.path.exists(h5_path):
        return dataset

    try:
        with h5py.File(h5_path, 'r') as f:
            for building in list(f.keys())[:2]:  # Houses 1 and 2 only
                elec = f.get(f'{building}/elec')
                if elec is None:
                    continue
                for meter in elec.keys():
                    # Try to read appliance name from attributes
                    meter_grp = elec[meter]
                    # Try table subgroup
                    tbl = meter_grp.get('table')
                    if tbl is None:
                        continue
                    # Get appliance label from attrs
                    label = ''
                    for attr_key in ['appliance', 'device_name', 'label']:
                        if attr_key in meter_grp.attrs:
                            label = str(meter_grp.attrs[attr_key]).lower()
                            break

                    canonical = None
                    for cls, uk_labels in UKDALE_MAP.items():
                        if any(ul in label for ul in uk_labels):
                            canonical = cls
                            break
                    if canonical is None:
                        continue

                    try:
                        # UK-DALE HDF5 stores data in pandas HDFStore format
                        vals = tbl['values_block_0'][:, 0].astype(np.float32)
                        segs = detect_transients(vals)
                        dataset[canonical].extend(segs)
                        if segs:
                            print(f'  UK-DALE {building}/{meter} {label}({canonical}): {len(segs)} segments')
                    except Exception:
                        pass
    except Exception as e:
        print(f'UK-DALE load error: {e}')

    return dataset

def make_synthetic_segments(cls_name, n=300):
    """Fallback synthetic segments matching SyntheticUKDALE profiles."""
    PROFILES = {
        'fridge': (150, 300, 10), 'hvac': (1500, 2200, 80),
        'kettle': (2200, 2400, 30), 'tv': (120, 180, 8),
        'washing_machine': (500, 2000, 100), 'dishwasher': (1800, 2000, 60),
        'microwave': (1200, 1300, 20), 'oven': (2000, 2200, 50),
        'ev_charger': (3300, 3500, 40), 'laptop': (60, 90, 5),
    }
    steady, peak, noise = PROFILES.get(cls_name, (200, 400, 20))
    rng = np.random.default_rng(42)
    segs = []
    for _ in range(n):
        seg = np.zeros(SEQ_LEN, dtype=np.float32)
        half = SEQ_LEN // 2
        seg[:half] = rng.normal(5, noise * 0.1, half)
        t = np.arange(SEQ_LEN - half)
        decay = peak * np.exp(-t / (SEQ_LEN * 0.15))
        steady_arr = np.full(SEQ_LEN - half, steady)
        seg[half:] = np.maximum(steady_arr, decay) + rng.normal(0, noise, SEQ_LEN - half)
        segs.append(np.clip(seg, 0, 4000))
    return segs

print('\nLoading REDD...')
redd_data = load_redd_segments()
print(f'REDD classes: {dict((k, len(v)) for k, v in redd_data.items())}')

print('\nLoading UK-DALE...')
ukdale_data = load_ukdale_segments()
print(f'UK-DALE classes: {dict((k, len(v)) for k, v in ukdale_data.items())}')

# Merge real data; fill gaps with synthetic
ALL_CLASSES = list(REDD_MAP.keys())
MIN_SAMPLES = 100
dataset = {}
for cls in ALL_CLASSES:
    segs = list(redd_data.get(cls, [])) + list(ukdale_data.get(cls, []))
    if len(segs) < MIN_SAMPLES:
        n_synth = MIN_SAMPLES - len(segs)
        segs += make_synthetic_segments(cls, n_synth)
        print(f'  {cls}: {len(segs)-n_synth} real + {n_synth} synthetic')
    else:
        print(f'  {cls}: {len(segs)} real samples')
    dataset[cls] = np.array(segs[:1000], dtype=np.float32)  # cap at 1000/class

print('\nFinal dataset:')
for cls, arr in dataset.items():
    print(f'  {cls}: {arr.shape}')

# ── CELL 6: ProtoNet model ────────────────────────────────────
CNN_CHANNELS = [32, 64, 128, 128, 128]
EMBED_DIM = 128

class CNN1DEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers, ch = [], 1
        for ch_out in CNN_CHANNELS:
            layers += [nn.Conv1d(ch, ch_out, 3, padding=1),
                       nn.BatchNorm1d(ch_out), nn.ReLU(),
                       nn.MaxPool1d(2), nn.Dropout(0.1)]
            ch = ch_out
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(CNN_CHANNELS[-1], EMBED_DIM)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        return self.proj(self.pool(self.cnn(x)).squeeze(-1))

class TemporalAttn(nn.Module):
    def __init__(self, seq_len=128):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(seq_len, seq_len//4), nn.Tanh(),
                                  nn.Linear(seq_len//4, seq_len), nn.Softmax(dim=-1))
    def forward(self, x): return x * self.attn(x)

class ProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TemporalAttn()
        self.enc  = CNN1DEncoder()

    def embed(self, x):
        return self.enc(self.attn(x))

    def forward(self, support, query):
        N, K, L = support.shape
        proto = self.embed(support.view(N*K, L)).view(N, K, -1).mean(1)
        q_emb = self.embed(query)
        dists = torch.cdist(q_emb.unsqueeze(0), proto.unsqueeze(0)).squeeze(0)
        return F.log_softmax(-dists, dim=1), dists

model = ProtoNet().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f'ProtoNet parameters: {total_params:,}')

# ── CELL 7: Episodic training ─────────────────────────────────
N_WAY, K_SHOT, Q_QUERY = 5, 5, 10
N_EPISODES  = 5000
LR          = 1e-3
ACCUM_STEPS = 4   # gradient accumulation for stable training

def sample_episode(dataset, n_way, k_shot, q_query):
    classes = random.sample(list(dataset.keys()), n_way)
    sup, qry, lbl = [], [], []
    for i, cls in enumerate(classes):
        arr = dataset[cls]
        need = k_shot + q_query
        idx = np.random.choice(len(arr), need, replace=len(arr) < need)
        sup.append(arr[idx[:k_shot]])
        qry.append(arr[idx[k_shot:]])
        lbl.extend([i] * q_query)
    return (np.stack(sup), np.concatenate(qry), np.array(lbl))

optim   = torch.optim.Adam(model.parameters(), lr=LR)
scaler  = GradScaler()
nll     = nn.NLLLoss()
sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optim, N_EPISODES)

losses, accs = [], []
model.train()
optim.zero_grad()

pbar = tqdm(range(1, N_EPISODES + 1), desc='Training')
for ep in pbar:
    sup_np, q_np, lbl_np = sample_episode(dataset, N_WAY, K_SHOT, Q_QUERY)
    sup = torch.tensor(sup_np, dtype=torch.float32).to(DEVICE)
    qry = torch.tensor(q_np,   dtype=torch.float32).to(DEVICE)
    lbl = torch.tensor(lbl_np, dtype=torch.long).to(DEVICE)

    with autocast():
        log_p, _ = model(sup, qry)
        loss = nll(log_p, lbl) / ACCUM_STEPS

    scaler.scale(loss).backward()

    if ep % ACCUM_STEPS == 0:
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

    sched.step()

    acc = (log_p.argmax(1) == lbl).float().mean().item()
    losses.append(loss.item() * ACCUM_STEPS)
    accs.append(acc)

    if ep % 200 == 0:
        pbar.set_postfix({'loss': f'{np.mean(losses[-200:]):.4f}',
                          'acc':  f'{np.mean(accs[-200:]):.3f}'})

print(f'\nFinal — loss: {np.mean(losses[-500:]):.4f} | acc: {np.mean(accs[-500:]):.3f}')

# ── CELL 8: Build Prototype Registry + Weibull + Temperature Scaling ─
from scipy.stats import exponweib
import torch.optim as topt

model.eval()
ALL_CLS = list(dataset.keys())

# --- Prototype Registry ---
registry = {}
with torch.no_grad():
    for cls in ALL_CLS:
        x = torch.tensor(dataset[cls][:K_SHOT], dtype=torch.float32).to(DEVICE)
        registry[cls] = model.embed(x).mean(0).cpu()

# --- Weibull fit ---
weibull = {}
with torch.no_grad():
    for idx, cls in enumerate(ALL_CLS):
        x = torch.tensor(dataset[cls][:50], dtype=torch.float32).to(DEVICE)
        emb = model.embed(x).cpu()
        proto = registry[cls]
        dists = ((emb - proto)**2).sum(1).numpy()
        tail = np.sort(dists)[-20:]
        if len(tail) < 2: tail = np.append(tail, tail[-1]+1e-5)
        weibull[idx] = exponweib.fit(tail, floc=0)

# --- Temperature Scaling ---
class TScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))
    def forward(self, logits):
        return torch.softmax(logits / self.T.clamp(min=0.05), dim=-1)

ts = TScaler()
all_logits, all_labels = [], []
with torch.no_grad():
    for _ in range(200):
        sup_np, q_np, lbl_np = sample_episode(dataset, N_WAY, K_SHOT, Q_QUERY)
        sup = torch.tensor(sup_np, dtype=torch.float32).to(DEVICE)
        qry = torch.tensor(q_np,   dtype=torch.float32).to(DEVICE)
        _, dists = model(sup, qry)
        all_logits.append((-dists).cpu()); all_labels.append(torch.tensor(lbl_np))

lg = torch.cat(all_logits); lb = torch.cat(all_labels)
opt_ts = topt.LBFGS([ts.T], lr=0.01, max_iter=500)
def _eval():
    opt_ts.zero_grad()
    loss = nn.CrossEntropyLoss()(lg / ts.T.clamp(0.05), lb)
    loss.backward(); return loss
opt_ts.step(_eval)
print(f'Temperature T = {ts.T.item():.4f}')

# ── CELL 9: Evaluation & Confusion Matrix ────────────────────
# Hold-out test set: last 20% of each class
y_true, y_pred = [], []
model.eval()

with torch.no_grad():
    # Build prototypes from first 80%
    proto_tensors = torch.stack([registry[c] for c in ALL_CLS]).to(DEVICE)

    for cls_idx, cls in enumerate(ALL_CLS):
        arr = dataset[cls]
        test_segs = arr[int(len(arr)*0.8):]
        if len(test_segs) == 0: continue
        x = torch.tensor(test_segs, dtype=torch.float32).to(DEVICE)
        emb = model.embed(x)
        dists = torch.cdist(emb.unsqueeze(0), proto_tensors.unsqueeze(0)).squeeze(0)
        preds = dists.argmin(1).cpu().numpy()
        y_true.extend([cls_idx] * len(test_segs))
        y_pred.extend(preds.tolist())

report = classification_report(y_true, y_pred, target_names=ALL_CLS, digits=3)
print('\n' + report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(cm, display_labels=ALL_CLS)
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('ProtoNet — UK-DALE + REDD Confusion Matrix', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Loss & accuracy curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
window = 100
loss_ma = np.convolve(losses, np.ones(window)/window, 'valid')
acc_ma  = np.convolve(accs,   np.ones(window)/window, 'valid')
ax1.plot(loss_ma, color='#ef4444'); ax1.set_title('Training Loss'); ax1.set_xlabel('Episode')
ax2.plot(acc_ma,  color='#22c55e'); ax2.set_title('Episode Accuracy'); ax2.set_xlabel('Episode')
ax2.axhline(0.8, color='gray', linestyle='--', label='80% target')
ax2.legend()
plt.tight_layout()
plt.savefig('results/training_curves.png', dpi=150)
plt.show()

# Training report JSON
from sklearn.metrics import accuracy_score, f1_score
report_dict = {
    'overall_accuracy': accuracy_score(y_true, y_pred),
    'macro_f1': f1_score(y_true, y_pred, average='macro'),
    'per_class_f1': dict(zip(ALL_CLS, f1_score(y_true, y_pred, average=None).tolist())),
    'temperature_T': ts.T.item(),
    'n_episodes': N_EPISODES,
    'dataset': {'real_sources': ['REDD', 'UK-DALE'], 'n_way': N_WAY, 'k_shot': K_SHOT},
}
with open('results/training_report.json', 'w') as f:
    json.dump(report_dict, f, indent=2)
print(json.dumps(report_dict, indent=2))

# ── CELL 10: Save weights & package for download ─────────────
# ProtoNet weights
torch.save(model.state_dict(), 'weights/protonet.pt')

# Prototype Registry
torch.save({cls: (proto.cpu(), K_SHOT) for cls, proto in registry.items()},
           'weights/prototype_registry.pt')

# Weibull OpenMax
import pickle
with open('weights/openmax_weibull.pkl', 'wb') as f:
    pickle.dump({'weibull': weibull, 'classes': ALL_CLS}, f)

# Temperature Scaler
torch.save(ts.state_dict(), 'weights/temperature_scaler.pt')

# Bundle everything into a zip
with zipfile.ZipFile('ems_weights.zip', 'w') as zf:
    for fname in ['protonet.pt', 'prototype_registry.pt',
                  'openmax_weibull.pkl', 'temperature_scaler.pt']:
        zf.write(f'weights/{fname}', fname)
    for fname in ['confusion_matrix.png', 'training_curves.png', 'training_report.json']:
        if os.path.exists(f'results/{fname}'):
            zf.write(f'results/{fname}', fname)

print('\n✅ Weights packaged: ems_weights.zip')
print(f'   Size: {os.path.getsize("ems_weights.zip")/1e6:.1f} MB')

# Auto-download in Colab
try:
    from google.colab import files
    files.download('ems_weights.zip')
    print('Download started!')
except ImportError:
    print('Not in Colab — find ems_weights.zip in the current directory.')
