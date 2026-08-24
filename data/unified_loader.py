"""
Unified NILM Dataset Loader — combines UK-DALE, REDD, and Synthetic data.

Provides a single interface for the training script to load appliance
power signatures from multiple sources:
  1. UK-DALE (real data from Zenodo processed tensors or HDF5)
  2. REDD   (real data from Zenodo processed tensors or HDF5)
  3. SYND   (improved synthetic data from data/synd.py)

All sources are normalized to the same format:
  {class_name: np.ndarray of shape (N, seq_len)}

Usage:
    from data.unified_loader import UnifiedNILMDataset
    dataset = UnifiedNILMDataset(seq_len=128, sources=['ukdale', 'redd', 'synd'])
    data = dataset.load_all_classes()
"""
import os
import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Path constants
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_DATA_DIR = os.path.join(DATA_DIR, 'real')
REAL_CACHE_DIR = os.path.join(REAL_DATA_DIR, 'cache')

# npz key prefix for the per-window "<dataset>/<building>" provenance tags
_GRP = '__groups__'

# Canonical appliance class mapping — maps dataset-specific names to unified names
# This handles naming inconsistencies across UK-DALE, REDD, and our ESP32 naming
CANONICAL_MAP = {
    # UK-DALE / REDD names → canonical
    'fridge freezer': 'fridge',
    'fridge': 'fridge',
    'refrigerator': 'fridge',
    'dish washer': 'dishwasher',
    'dishwasher': 'dishwasher',
    'washing machine': 'washing_machine',
    'washer dryer': 'washing_machine',
    'washer': 'washing_machine',
    'kettle': 'kettle',
    'microwave': 'microwave',
    'boiler': 'hvac',
    'hvac': 'hvac',
    'air conditioner': 'hvac',
    'furnace': 'hvac',
    'electric furnace': 'hvac',
    'electric space heater': 'hvac',
    'television': 'tv',
    'tv': 'tv',
    'computer': 'laptop',
    'laptop': 'laptop',
    'laptop computer': 'laptop',
    'desktop computer': 'laptop',
    'oven': 'oven',
    'electric oven': 'oven',
    'stove': 'oven',
    'electric stove': 'oven',
    'freezer': 'fridge',
    'fridge freezer': 'fridge',
    'ev charger': 'ev_charger',
    'ev_charger': 'ev_charger',
    'lighting': 'laptop',  # Similar power range, group together
    'light': 'laptop',
    # ESP32 naming
    'esp32_fridge': 'fridge',
    'esp32_hvac': 'hvac',
    'esp32_kettle': 'kettle',
    'esp32_tv': 'tv',
    'esp32_washer': 'washing_machine',
    'esp32_dishwasher': 'dishwasher',
    'esp32_microwave': 'microwave',
    'esp32_oven': 'oven',
    'esp32_lighting': 'laptop',
    'esp32_dryer': 'washing_machine',
}

# Target classes for the ProtoNet (must match train_models.py and config)
TARGET_CLASSES = [
    'fridge', 'hvac', 'kettle', 'tv', 'washing_machine',
    'dishwasher', 'microwave', 'oven', 'ev_charger', 'laptop'
]

# ── Demo class set ───────────────────────────────────────────────────────────
# For a bench demo the plugged-in loads are consumer electronics (laptop,
# projector, monitor, desktop, phone charger) rather than kitchen appliances.
# All of these except the charger have real UK-DALE coverage, so the demo model
# is trained on measured data too — see DEMO_EXTRA_MAP for the meter sources.
#
# Spanning 3 W to 3.5 kW in one 10-way model is what makes the low-power classes
# weakest (laptop F1 = 0.16 on unseen houses). A dedicated low-power class set
# discriminates far better because the whole set lives in one decade of power.
DEMO_CLASSES = [
    'laptop', 'desktop_computer', 'monitor', 'projector',
    'tv', 'router', 'phone_charger',
]

# Extra canonical names needed by DEMO_CLASSES. Merged into CANONICAL_MAP below
# only when a demo class set is requested, so the default 10-class behaviour and
# the existing test baseline are untouched.
DEMO_EXTRA_MAP = {
    # UK-DALE b1/m51, b5/m3, b5/m14
    'desktop computer': 'desktop_computer',
    'server computer': 'desktop_computer',
    'htpc': 'desktop_computer',
    # UK-DALE b1/m14, b2/m3, b5/m6, b5/m10
    'computer monitor': 'monitor',
    # UK-DALE b3/m5  (single instance — no unseen-house validation possible)
    'projector': 'projector',
    # UK-DALE b1/m18, b2/m18, b1/m21
    'broadband router': 'router',
    'modem': 'router',
    'ethernet switch': 'router',
    'network attached storage': 'router',
    # UK-DALE b1/m34, b1/m32, b1/m27
    'mobile phone charger': 'phone_charger',
    'wireless phone charger': 'phone_charger',
    'tablet computer charger': 'phone_charger',
    'charger': 'phone_charger',
    # keep laptop/desktop distinct for the demo (default map folds both to laptop)
    'laptop computer': 'laptop',
    'laptop': 'laptop',
    'computer': 'desktop_computer',
    'television': 'tv',
    'tv': 'tv',
}


class UnifiedNILMDataset:
    """
    Loads and combines appliance power signatures from multiple NILM datasets.
    
    All data is resampled/padded to seq_len and normalized to consistent
    class names for the ProtoNet meta-training pipeline.
    """

    def __init__(self, seq_len: int = 128,
                 sources: List[str] = None,
                 min_samples_per_class: int = 50,
                 max_samples_per_class: int = 500,
                 seed: int = 42,
                 target_classes: Optional[List[str]] = None,
                 extra_canonical_map: Optional[Dict[str, str]] = None,
                 on_threshold_w: Optional[float] = None,
                 cache_tag: str = ''):
        """
        Args:
            seq_len: Window length for all segments.
            sources: List of data sources to use: 'ukdale', 'redd', 'synd'.
                     Default: tries ukdale+redd first, falls back to synd.
            min_samples_per_class: Minimum samples required per class.
            max_samples_per_class: Cap samples per class to prevent imbalance.
            seed: Random seed for reproducibility.
            target_classes: Override the class set (e.g. DEMO_CLASSES).
            extra_canonical_map: Additional dataset-name -> class mappings,
                     layered over CANONICAL_MAP (e.g. DEMO_EXTRA_MAP).
            on_threshold_w: Override the "appliance is on" power floor used when
                     extracting real windows. Must be lowered for low-power demo
                     loads — a phone charger draws 3-10 W, well under the 20 W
                     default, and would otherwise yield zero windows.
            cache_tag: Suffix for the extraction cache, so a demo class set does
                     not collide with the default 10-class cache.
        """
        self.seq_len = seq_len
        self.sources = sources or ['ukdale', 'redd', 'synd']
        self.min_samples = min_samples_per_class
        self.max_samples = max_samples_per_class
        self._rng = np.random.default_rng(seed)
        self._real_groups: Dict[str, np.ndarray] = {}
        self.real_data: Dict[str, np.ndarray] = {}
        self.provenance: Dict[str, str] = {}

        self.target_classes = list(target_classes) if target_classes else list(TARGET_CLASSES)
        self.canonical_map = dict(CANONICAL_MAP)
        if extra_canonical_map:
            self.canonical_map.update(extra_canonical_map)
        self.on_threshold_w = on_threshold_w
        self.cache_tag = cache_tag

    def _load_ukdale_h5(self) -> Dict[str, np.ndarray]:
        """Load UK-DALE from HDF5 file (Zenodo processed format)."""
        import h5py
        path = os.path.join(REAL_DATA_DIR, 'ukdale.h5')
        if not os.path.exists(path):
            logger.warning(f"UK-DALE HDF5 not found at {path}")
            return {}

        data = {}
        try:
            with h5py.File(path, 'r') as f:
                # Try nilmtk format: /building{N}/elec/meter{M}
                for key in f.keys():
                    if key.startswith('building'):
                        building = f[key]
                        if 'elec' in building:
                            elec = building['elec']
                            for meter_key in elec.keys():
                                if 'table' in elec[meter_key]:
                                    table = elec[meter_key]['table']
                                    # Get the power data
                                    if hasattr(table, 'shape') and len(table.shape) > 0:
                                        raw = np.array(table[:], dtype=np.float32)
                                        # Extract active power column if structured
                                        if raw.ndim == 2:
                                            power = raw[:, 0]  # First column is usually active power
                                        else:
                                            power = raw.flatten()
                                        
                                        # Segment into windows
                                        label = self._get_label_from_metadata(f, key, meter_key)
                                        if label:
                                            segments = self._segment_power_series(power)
                                            if len(segments) > 0:
                                                canonical = CANONICAL_MAP.get(label.lower(), label.lower())
                                                if canonical in TARGET_CLASSES:
                                                    if canonical not in data:
                                                        data[canonical] = []
                                                    data[canonical].extend(segments)
                    # Also try flat format: /appliances/{name}/windows
                    elif key == 'appliances':
                        for app_name in f['appliances'].keys():
                            grp = f['appliances'][app_name]
                            if 'windows' in grp:
                                windows = np.array(grp['windows'][:], dtype=np.float32)
                                canonical = CANONICAL_MAP.get(app_name.lower(), app_name.lower())
                                if canonical in TARGET_CLASSES:
                                    segs = self._resize_windows(windows)
                                    if canonical not in data:
                                        data[canonical] = []
                                    data[canonical].extend(list(segs))

            logger.info(f"UK-DALE HDF5: loaded {sum(len(v) for v in data.values())} segments "
                       f"across {len(data)} classes")
        except Exception as e:
            logger.error(f"Failed to load UK-DALE HDF5: {e}")

        return {k: np.array(v, dtype=np.float32) for k, v in data.items() if len(v) > 0}

    def _load_redd_h5(self) -> Dict[str, np.ndarray]:
        """Load REDD from HDF5 file (Zenodo processed format)."""
        import h5py
        path = os.path.join(REAL_DATA_DIR, 'redd.h5')
        if not os.path.exists(path):
            logger.warning(f"REDD HDF5 not found at {path}")
            return {}

        data = {}
        try:
            with h5py.File(path, 'r') as f:
                for key in f.keys():
                    if key.startswith('building'):
                        building = f[key]
                        if 'elec' in building:
                            elec = building['elec']
                            for meter_key in elec.keys():
                                if 'table' in elec[meter_key]:
                                    table = elec[meter_key]['table']
                                    if hasattr(table, 'shape') and len(table.shape) > 0:
                                        raw = np.array(table[:], dtype=np.float32)
                                        if raw.ndim == 2:
                                            power = raw[:, 0]
                                        else:
                                            power = raw.flatten()
                                        
                                        label = self._get_label_from_metadata(f, key, meter_key)
                                        if label:
                                            segments = self._segment_power_series(power)
                                            if len(segments) > 0:
                                                canonical = CANONICAL_MAP.get(label.lower(), label.lower())
                                                if canonical in TARGET_CLASSES:
                                                    if canonical not in data:
                                                        data[canonical] = []
                                                    data[canonical].extend(segments)
                    elif key == 'appliances':
                        for app_name in f['appliances'].keys():
                            grp = f['appliances'][app_name]
                            if 'windows' in grp:
                                windows = np.array(grp['windows'][:], dtype=np.float32)
                                canonical = CANONICAL_MAP.get(app_name.lower(), app_name.lower())
                                if canonical in TARGET_CLASSES:
                                    segs = self._resize_windows(windows)
                                    if canonical not in data:
                                        data[canonical] = []
                                    data[canonical].extend(list(segs))

            logger.info(f"REDD HDF5: loaded {sum(len(v) for v in data.values())} segments "
                       f"across {len(data)} classes")
        except Exception as e:
            logger.error(f"Failed to load REDD HDF5: {e}")

        return {k: np.array(v, dtype=np.float32) for k, v in data.items() if len(v) > 0}

    def _load_nilmtk(self, source: str) -> Dict[str, np.ndarray]:
        """
        Load REAL labelled windows from a nilmtk HDF5 dataset (ukdale / redd).

        Uses data/nilmtk_reader.py, which handles the two things that
        previously made real data unreadable: Blosc decompression
        (hdf5plugin) and pickle-encoded appliance metadata.

        A compressed cache under data/real/cache/ is used when present, since
        a full extraction pass reads several GB of HDF5.
        """
        cache_path = os.path.join(REAL_CACHE_DIR,
                                  f'{source}_windows{self.cache_tag}.npz')
        if os.path.exists(cache_path):
            try:
                with np.load(cache_path, allow_pickle=True) as z:
                    data = {k: z[k].astype(np.float32) for k in z.files
                            if k in self.target_classes}
                    grp = {k[len(_GRP):]: z[k] for k in z.files
                           if k.startswith(_GRP) and k[len(_GRP):] in data}
                if data:
                    self._pending_groups = {
                        c: np.asarray(g, dtype=object) for c, g in grp.items()}
                    logger.info(
                        f"{source}: {sum(len(v) for v in data.values())} REAL windows "
                        f"from cache across {len(data)} classes"
                    )
                    return data
            except Exception as e:
                logger.warning(f"{source} cache unreadable ({e}) — re-extracting")

        from data.nilmtk_reader import NILMTKReader
        path = os.path.join(REAL_DATA_DIR, f'{source}.h5')
        reader = NILMTKReader(path, self.canonical_map, self.target_classes,
                              seq_len=self.seq_len,
                              on_threshold_w=self.on_threshold_w)
        data = reader.load()
        groups = getattr(reader, 'window_groups', {}) or {}
        self._pending_groups = groups

        if data:
            try:
                os.makedirs(REAL_CACHE_DIR, exist_ok=True)
                payload = dict(data)
                for cls, g in groups.items():
                    payload[_GRP + cls] = np.asarray(g, dtype=object)
                np.savez_compressed(cache_path, **payload)
                logger.info(f"Cached {source} real windows -> {cache_path}")
            except Exception as e:
                logger.warning(f"Could not write {source} cache: {e}")
        return data

    def _load_labelled_npy(self) -> Dict[str, np.ndarray]:
        """
        Load per-appliance pre-extracted window files, e.g.
        data/real/ukdale_tv.npy -> class 'tv'.

        These are already (N, 128) real windows. Ignored by the previous
        loader, which only looked for '{source}_tensor.npy'.
        """
        data: Dict[str, List[np.ndarray]] = {}
        if not os.path.isdir(REAL_DATA_DIR):
            return {}

        for fname in sorted(os.listdir(REAL_DATA_DIR)):
            if not fname.endswith('.npy') or fname.endswith('_tensor.npy'):
                continue
            stem = fname[:-4]
            # strip a leading dataset prefix: ukdale_tv -> tv
            for prefix in ('ukdale_', 'redd_', 'synd_'):
                if stem.startswith(prefix):
                    stem = stem[len(prefix):]
                    break
            canonical = self.canonical_map.get(stem.replace('_', ' ').lower(),
                                               self.canonical_map.get(stem.lower(), stem.lower()))
            if canonical not in self.target_classes:
                continue
            try:
                arr = np.load(os.path.join(REAL_DATA_DIR, fname), allow_pickle=False)
            except Exception as e:
                logger.debug(f"Skipping {fname}: {e}")
                continue
            if arr.ndim != 2 or arr.shape[0] == 0:
                continue
            segs = self._resize_windows(np.asarray(arr, dtype=np.float32))
            data.setdefault(canonical, []).append(segs)
            logger.info(f"Labelled npy {fname} -> {canonical}: {segs.shape[0]} windows")

        return {k: np.concatenate(v, axis=0).astype(np.float32)
                for k, v in data.items() if v}

    def _load_tensor_npy(self, source: str) -> Dict[str, np.ndarray]:
        """Load pre-processed tensor .npy files from Zenodo."""
        filename = f"{source}_tensor.npy"
        path = os.path.join(REAL_DATA_DIR, filename)
        if not os.path.exists(path):
            logger.warning(f"Tensor file not found: {path}")
            return {}

        data = {}
        try:
            tensor = np.load(path, allow_pickle=True)
            # The Zenodo tensors may be structured differently
            # Try common formats:
            if isinstance(tensor, np.ndarray):
                if tensor.dtype == object:
                    # It's a dict-like object
                    item = tensor.item()
                    if isinstance(item, dict):
                        for name, arr in item.items():
                            canonical = CANONICAL_MAP.get(str(name).lower(), str(name).lower())
                            if canonical in TARGET_CLASSES:
                                segs = self._resize_windows(np.array(arr, dtype=np.float32))
                                data[canonical] = segs
                elif tensor.ndim == 3:
                    # Shape: (n_classes, n_samples, seq_len) or (n_samples, n_classes, seq_len)
                    logger.info(f"Tensor shape: {tensor.shape}")
                    # Without class labels, we can't map — log and skip
                    logger.warning(f"Tensor is unlabeled 3D array {tensor.shape} — need metadata")
                elif tensor.ndim == 2:
                    # Shape: (n_samples, seq_len) — single class or flat
                    logger.info(f"Tensor shape: {tensor.shape} — single flat array")

            logger.info(f"{source} tensor: loaded {sum(len(v) for v in data.values())} segments "
                       f"across {len(data)} classes")
        except Exception as e:
            logger.error(f"Failed to load {source} tensor: {e}")

        return data

    def _load_synthetic(self) -> Dict[str, np.ndarray]:
        """Load improved synthetic data from data/synd.py."""
        from data.synd import SyntheticUKDALE
        synth = SyntheticUKDALE(seq_len=self.seq_len, n_samples=self.max_samples)
        data = synth.load_all_classes()
        logger.info(f"Synthetic: loaded {sum(len(v) for v in data.values())} segments "
                   f"across {len(data)} classes")
        return data

    def _load_mock_h5(self) -> Dict[str, np.ndarray]:
        """Load mock UK-DALE HDF5 from backend/data/ as fallback."""
        import h5py
        path = os.path.join(os.path.dirname(DATA_DIR), 'backend', 'data', 'mock_ukdale.h5')
        if not os.path.exists(path):
            return {}

        data = {}
        try:
            with h5py.File(path, 'r') as f:
                if 'appliances' in f:
                    for app_name in f['appliances'].keys():
                        grp = f['appliances'][app_name]
                        if 'windows' in grp:
                            windows = np.array(grp['windows'][:], dtype=np.float32)
                            canonical = CANONICAL_MAP.get(app_name.lower(), app_name.lower())
                            if canonical in TARGET_CLASSES:
                                segs = self._resize_windows(windows)
                                if canonical not in data:
                                    data[canonical] = []
                                data[canonical] = list(segs)

            logger.info(f"Mock HDF5: loaded {sum(len(v) for v in data.values())} segments "
                       f"across {len(data)} classes")
        except Exception as e:
            logger.error(f"Failed to load mock HDF5: {e}")

        return {k: np.array(v, dtype=np.float32) for k, v in data.items() if len(v) > 0}

    def _get_label_from_metadata(self, f, building_key, meter_key) -> Optional[str]:
        """Extract appliance label from nilmtk-format HDF5 metadata."""
        try:
            # nilmtk stores metadata in attributes or separate groups
            meter_grp = f[building_key]['elec'][meter_key]
            # Check for 'device_model' or label attributes
            if 'device_model' in meter_grp.attrs:
                return str(meter_grp.attrs['device_model'])
            if 'type' in meter_grp.attrs:
                return str(meter_grp.attrs['type'])
            # Try reading from the table's metadata
            if 'table' in meter_grp:
                table = meter_grp['table']
                if 'title' in table.attrs:
                    return str(table.attrs['title'])
        except Exception:
            pass
        return None

    def _segment_power_series(self, power: np.ndarray,
                               min_power_w: float = 20.0) -> List[np.ndarray]:
        """
        Segment a continuous power time-series into transient windows.
        
        Detects ON-events (power crossing above min_power_w from below)
        and extracts windows of seq_len centered on each event.
        """
        segments = []
        if len(power) < self.seq_len:
            return segments

        # Remove NaN/Inf
        power = np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)

        # Find ON-events: transitions from below to above threshold
        above = power > min_power_w
        transitions = np.diff(above.astype(int))
        on_events = np.where(transitions == 1)[0]

        half = self.seq_len // 2
        for idx in on_events:
            start = max(0, idx - half)
            end = start + self.seq_len
            if end > len(power):
                start = len(power) - self.seq_len
                end = len(power)
            if start < 0:
                continue

            seg = power[start:end].copy().astype(np.float32)
            if len(seg) == self.seq_len and np.max(seg) > min_power_w:
                segments.append(seg)

        # If no ON-events found, sample random windows with activity
        if len(segments) == 0:
            n_windows = min(50, len(power) // self.seq_len)
            for _ in range(n_windows):
                start = self._rng.integers(0, max(1, len(power) - self.seq_len))
                seg = power[start:start + self.seq_len].copy().astype(np.float32)
                if np.max(seg) > min_power_w:
                    segments.append(seg)

        return segments

    def _resize_windows(self, windows: np.ndarray) -> np.ndarray:
        """Resize windows to target seq_len using interpolation."""
        if windows.ndim == 1:
            windows = windows.reshape(1, -1)
        
        n, current_len = windows.shape
        if current_len == self.seq_len:
            return windows

        # Linear interpolation to target length
        x_old = np.linspace(0, 1, current_len)
        x_new = np.linspace(0, 1, self.seq_len)
        resized = np.array([
            np.interp(x_new, x_old, w) for w in windows
        ], dtype=np.float32)

        return resized

    def _merge_datasets(self, datasets: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Merge multiple dataset dicts, concatenating per-class arrays."""
        merged = {}
        for ds in datasets:
            for cls_name, segments in ds.items():
                if cls_name not in self.target_classes:
                    continue
                if cls_name not in merged:
                    merged[cls_name] = []
                if isinstance(segments, np.ndarray):
                    merged[cls_name].append(segments)
                elif isinstance(segments, list):
                    if len(segments) > 0:
                        merged[cls_name].append(np.array(segments, dtype=np.float32))

        result = {}
        for cls_name, arrays in merged.items():
            if len(arrays) == 0:
                continue
            # Filter out empty arrays
            valid = [a for a in arrays if a.size > 0 and a.ndim >= 1]
            if not valid:
                continue
            # Ensure 2D
            valid_2d = []
            for a in valid:
                if a.ndim == 1:
                    a = a.reshape(1, -1)
                valid_2d.append(a)
            combined = np.concatenate(valid_2d, axis=0)

            # Enforce sample limits
            if len(combined) > self.max_samples:
                idx = self._rng.choice(len(combined), self.max_samples, replace=False)
                combined = combined[idx]

            if len(combined) >= self.min_samples:
                result[cls_name] = combined
            else:
                logger.warning(f"Class '{cls_name}' has only {len(combined)} samples "
                             f"(min: {self.min_samples}) — skipping")

        return result

    def load_all_classes(self) -> Dict[str, np.ndarray]:
        """
        Load data from all configured sources and merge.

        Real nilmtk data (UK-DALE / REDD) takes priority; synthetic data is
        only used to backfill classes that have no real coverage (in practice
        just ev_charger, which neither public dataset contains).

        Returns:
            Dict mapping class names to (N, seq_len) numpy arrays.
        """
        datasets = []
        self.real_data = {}
        self.provenance = {}
        self._real_groups = {}

        def _absorb_real(ds: Dict[str, np.ndarray], groups: Dict[str, np.ndarray]):
            """Concatenate a real source into real_data, keeping tags aligned."""
            for cls, arr in ds.items():
                tags = groups.get(cls)
                if tags is None or len(tags) != len(arr):
                    tags = np.array(['unknown'] * len(arr), dtype=object)
                if cls in self.real_data:
                    self.real_data[cls] = np.concatenate([self.real_data[cls], arr], axis=0)
                    self._real_groups[cls] = np.concatenate([self._real_groups[cls], tags])
                else:
                    self.real_data[cls] = arr
                    self._real_groups[cls] = tags

        # ── 1. Real nilmtk datasets ──
        for source in self.sources:
            if source in ('ukdale', 'redd'):
                self._pending_groups = {}
                ds = self._load_nilmtk(source)
                if ds:
                    datasets.append(ds)
                    _absorb_real(ds, self._pending_groups)

        # ── 2. Real pre-extracted per-appliance .npy windows ──
        labelled = self._load_labelled_npy()
        if labelled:
            datasets.append(labelled)
            _absorb_real(labelled, {})

        real_data_found = bool(self.real_data)
        if real_data_found:
            logger.info(
                f"REAL data: {sum(len(v) for v in self.real_data.values())} windows "
                f"across {len(self.real_data)} classes {sorted(self.real_data.keys())}"
            )
        else:
            logger.warning(
                "No real data loaded. Install hdf5plugin and place ukdale.h5 / "
                "redd.h5 in data/real/ — otherwise the model trains on synthetic "
                "signatures only and will NOT reflect real-world accuracy."
            )

        # ── 3. Synthetic (explicitly requested, or required as a fallback) ──
        if 'synd' in self.sources or not real_data_found:
            datasets.append(self._load_synthetic())

        merged = self._merge_datasets(datasets)

        # Fill gaps: any target class with no real data at all falls back to synthetic
        if real_data_found:
            missing = [c for c in self.target_classes if c not in merged]
            if missing:
                synth = self._load_synthetic()
                for cls in missing:
                    if cls in synth:
                        logger.info(f"Backfilling class '{cls}' with synthetic data "
                                    f"(no real coverage in UK-DALE/REDD)")
                        merged[cls] = synth[cls][:self.max_samples]
                still_missing = [c for c in missing if c not in merged]
                if still_missing:
                    logger.warning(
                        f"No data at all for {still_missing} — these classes are "
                        f"absent from both the real datasets and the synthetic "
                        f"generator, so the model cannot predict them."
                    )

        for cls in merged:
            self.provenance[cls] = 'real' if cls in self.real_data else 'synthetic'

        # Log summary
        total = sum(len(v) for v in merged.values())
        logger.info(f"Unified dataset: {total} total segments across {len(merged)} classes")
        for cls in sorted(merged.keys()):
            logger.info(f"  {cls}: {len(merged[cls])} segments, "
                       f"shape={merged[cls].shape}, "
                       f"mean={merged[cls].mean():.1f}W, "
                       f"std={merged[cls].std():.1f}W "
                       f"[{self.provenance.get(cls, '?')}]")

        return merged

    def get_house_holdout_split(self, holdout_frac: float = 0.3
                                ) -> tuple:
        """
        Split REAL data by source house, so validation contains only appliances
        from buildings the model never trained on.

        This is the metric that predicts field behaviour. A random split scores
        much higher because the same physical fridge appears on both sides of
        it, so the model can match instance quirks instead of appliance class.

        Returns:
            (train_real, val_real, info) where info records the held-out houses.
        """
        if not self.real_data:
            self.load_all_classes()
        if not self.real_data:
            return {}, {}, {'holdout_houses': [], 'reason': 'no real data'}

        all_houses = sorted({h for g in self._real_groups.values()
                             for h in np.unique(g) if h != 'unknown'})
        if len(all_houses) < 2:
            return dict(self.real_data), {}, {
                'holdout_houses': [], 'reason': 'need >=2 houses'}

        # Choose held-out houses greedily, smallest-contribution first, so the
        # training set keeps the bulk of the data and as many classes as possible.
        house_counts = {
            h: int(sum(int((g == h).sum()) for g in self._real_groups.values()))
            for h in all_houses
        }
        total = sum(house_counts.values())
        target = holdout_frac * total

        holdout, acc = [], 0
        for h in sorted(all_houses, key=lambda x: house_counts[x]):
            if acc >= target:
                break
            holdout.append(h)
            acc += house_counts[h]
        holdout_set = set(holdout)

        train_real, val_real = {}, {}
        for cls, arr in self.real_data.items():
            g = self._real_groups.get(cls)
            if g is None or len(g) != len(arr):
                train_real[cls] = arr
                continue
            mask = np.array([t in holdout_set for t in g], dtype=bool)
            tr, va = arr[~mask], arr[mask]
            if len(tr) >= 2:
                train_real[cls] = tr[:self.max_samples]
            if len(va) >= 2:
                val_real[cls] = va[:self.max_samples]

        info = {
            'holdout_houses': sorted(holdout_set),
            'train_houses': sorted(set(all_houses) - holdout_set),
            'holdout_windows': int(acc),
            'total_real_windows': int(total),
            'val_classes': sorted(val_real.keys()),
            'train_classes': sorted(train_real.keys()),
        }
        logger.info(f"House holdout: val houses={info['holdout_houses']} "
                    f"({acc}/{total} windows) | val classes={info['val_classes']}")
        return train_real, val_real, info

    def get_real_only(self) -> Dict[str, np.ndarray]:
        """
        Real (UK-DALE / REDD) windows only, capped to max_samples per class.

        This is the honest held-out domain for evaluation: a model scored on
        synthetic data it was also trained on tells you nothing about
        real-world accuracy.
        """
        if not hasattr(self, 'real_data'):
            self.load_all_classes()
        out = {}
        for cls, arr in self.real_data.items():
            if len(arr) > self.max_samples:
                idx = self._rng.choice(len(arr), self.max_samples, replace=False)
                arr = arr[idx]
            if len(arr) >= max(2, self.min_samples // 5):
                out[cls] = arr
        return out

    def get_train_val_split(self, val_fraction: float = 0.2
                           ) -> tuple:
        """
        Load data and split into train/val sets per class.

        Returns:
            (train_data, val_data) — each is {class_name: np.ndarray}
        """
        data = self.load_all_classes()
        train_data, val_data = {}, {}

        for cls_name, segments in data.items():
            n = len(segments)
            n_val = max(1, int(n * val_fraction))
            idx = self._rng.permutation(n)
            val_data[cls_name] = segments[idx[:n_val]]
            train_data[cls_name] = segments[idx[n_val:]]

        return train_data, val_data
