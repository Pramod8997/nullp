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
    'electric space heater': 'hvac',
    'television': 'tv',
    'tv': 'tv',
    'computer': 'laptop',
    'laptop': 'laptop',
    'oven': 'oven',
    'electric oven': 'oven',
    'stove': 'oven',
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
                 seed: int = 42):
        """
        Args:
            seq_len: Window length for all segments.
            sources: List of data sources to use: 'ukdale', 'redd', 'synd'.
                     Default: tries ukdale+redd first, falls back to synd.
            min_samples_per_class: Minimum samples required per class.
            max_samples_per_class: Cap samples per class to prevent imbalance.
            seed: Random seed for reproducibility.
        """
        self.seq_len = seq_len
        self.sources = sources or ['ukdale', 'redd', 'synd']
        self.min_samples = min_samples_per_class
        self.max_samples = max_samples_per_class
        self._rng = np.random.default_rng(seed)

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
                if cls_name not in TARGET_CLASSES:
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

        Returns:
            Dict mapping class names to (N, seq_len) numpy arrays.
        """
        datasets = []
        real_data_found = False

        for source in self.sources:
            if source == 'ukdale':
                # Try tensor first, then HDF5
                ds = self._load_tensor_npy('ukdale')
                if not ds:
                    ds = self._load_ukdale_h5()
                if not ds:
                    ds = self._load_mock_h5()
                if ds:
                    datasets.append(ds)
                    real_data_found = True

            elif source == 'redd':
                ds = self._load_tensor_npy('redd')
                if not ds:
                    ds = self._load_redd_h5()
                if ds:
                    datasets.append(ds)
                    real_data_found = True

            elif source == 'synd':
                ds = self._load_synthetic()
                datasets.append(ds)

        if not datasets:
            logger.error("No data sources loaded! Falling back to synthetic only.")
            datasets.append(self._load_synthetic())

        merged = self._merge_datasets(datasets)

        # Fill gaps: if a target class has no real data, fill with synthetic
        if real_data_found:
            synth = self._load_synthetic()
            for cls in TARGET_CLASSES:
                if cls not in merged and cls in synth:
                    logger.info(f"Backfilling class '{cls}' with synthetic data")
                    merged[cls] = synth[cls][:self.max_samples]

        # Log summary
        total = sum(len(v) for v in merged.values())
        logger.info(f"Unified dataset: {total} total segments across {len(merged)} classes")
        for cls in sorted(merged.keys()):
            logger.info(f"  {cls}: {len(merged[cls])} segments, "
                       f"shape={merged[cls].shape}, "
                       f"mean={merged[cls].mean():.1f}W, "
                       f"std={merged[cls].std():.1f}W")

        return merged

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
