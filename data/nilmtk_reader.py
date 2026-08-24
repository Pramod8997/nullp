"""
NILMTK HDF5 Reader — extracts *real* labelled appliance windows from UK-DALE / REDD.

Why this module exists
----------------------
`data/unified_loader.py` previously could not read either real dataset:

  1. The nilmtk HDF5 tables are Blosc-compressed. Without the Blosc filter
     registered, h5py raises "can't open directory (/usr/local/lib/plugin)".
     Fixed by importing `hdf5plugin` before opening the file.
  2. Appliance labels are NOT stored in meter-group attrs (`device_model` /
     `type`). They live in `/building{N}.attrs['metadata']`, encoded as a
     **Python pickle (protocol 1)** — not JSON. Parsed here with
     `pickle.loads(..., encoding='latin1')`.

Temporal contract (matters for real-world accuracy)
---------------------------------------------------
UK-DALE submeters sample at 6 s, REDD at 3-4 s, but the deployed PZEM-004T
publishes at ~1 Hz with a 500-1000 ms register refresh, which makes its output
a *staircase* (see HARDWARE_DEPLOYMENT_GUIDE.md, Hazard 4).

So a window here is defined by **wall-clock duration**, not sample count:
`WINDOW_SECONDS` of real time, resampled onto a 1 Hz grid of `SEQ_LEN` points
using **zero-order hold**. ZOH is deliberate — linear interpolation would
invent smooth ramps the PZEM never produces, while ZOH reproduces the same
quantised staircase the real sensor emits.

Label integrity
---------------
A single physical meter can host several appliances (e.g. UK-DALE building1
meter10 carries kettle + food processor + toasted sandwich maker). Such meters
are **excluded** unless every appliance on them maps to the same canonical
class — otherwise the label is noise that silently poisons training.
"""
import os
import pickle
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import hdf5plugin  # noqa: F401  — registers Blosc/LZF filters with libhdf5
    _HAVE_HDF5PLUGIN = True
except ImportError:  # pragma: no cover - surfaced as a clear runtime warning
    _HAVE_HDF5PLUGIN = False

import h5py

logger = logging.getLogger(__name__)

# ── Temporal contract ────────────────────────────────────────────────────────
SEQ_LEN         = 128     # samples per window (model input width)
TARGET_HZ       = 1.0     # deployment cadence (PZEM-004T effective rate)
WINDOW_SECONDS  = SEQ_LEN / TARGET_HZ   # 128 s of wall-clock per window

# ── Extraction tuning ────────────────────────────────────────────────────────
ON_THRESHOLD_W    = 20.0    # power above which an appliance counts as "on"
MAX_ROWS_PER_METER = 4_000_000
MAX_WINDOWS_PER_METER = 400
MAX_GAP_S         = 1800.0  # reject windows spanning a >30 min logging gap

# Quality gates — reject near-idle blips that would otherwise dominate a class.
# Without these, an HVAC meter idling at 5 W emits thousands of 25 W "events"
# and the class prototype collapses toward zero.
MIN_STEP_W        = 25.0    # required rise from pre-edge baseline to on-state
MIN_ON_MEDIAN_W   = 30.0    # required median power over the on-portion
MIN_ON_FRACTION   = 0.15    # >=15% of window samples must be above threshold


class NILMTKReader:
    """Reads canonical-labelled appliance windows out of a nilmtk HDF5 file."""

    def __init__(self, h5_path: str, canonical_map: Dict[str, str],
                 target_classes: List[str], seq_len: int = SEQ_LEN,
                 seed: int = 42, on_threshold_w: Optional[float] = None):
        self.path = h5_path
        self.canonical_map = canonical_map
        self.target_classes = set(target_classes)
        self.seq_len = seq_len
        self.window_seconds = seq_len / TARGET_HZ
        self._rng = np.random.default_rng(seed)

        # Detection floor. Scale the quality gates with it, so low-power demo
        # loads (phone charger 3-10 W, monitor 20-40 W) are not rejected by
        # thresholds tuned for kitchen appliances.
        self.on_threshold_w = (float(on_threshold_w)
                               if on_threshold_w is not None else ON_THRESHOLD_W)
        scale = self.on_threshold_w / ON_THRESHOLD_W
        self.min_step_w = MIN_STEP_W * scale
        self.min_on_median_w = MIN_ON_MEDIAN_W * scale

        if not _HAVE_HDF5PLUGIN:
            logger.warning(
                "hdf5plugin is not installed — Blosc-compressed nilmtk tables "
                "cannot be decompressed. Run: pip install hdf5plugin"
            )

    # ── Metadata ─────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_building_metadata(f: h5py.File, building: str) -> dict:
        """nilmtk stores building metadata as a protocol-1 pickle, not JSON."""
        raw = f[building].attrs.get('metadata')
        if raw is None:
            return {}
        if hasattr(raw, 'tobytes'):
            raw = raw.tobytes()
        elif isinstance(raw, str):
            raw = raw.encode('latin1')
        try:
            return pickle.loads(raw, encoding='latin1')
        except Exception as e:
            logger.debug(f"{building}: metadata unpickle failed: {e}")
            return {}

    def _meter_labels(self, f: h5py.File, building: str) -> Dict[int, str]:
        """
        Map meter number -> canonical class, keeping only unambiguous meters.

        A meter is dropped when the appliances sharing it resolve to more than
        one canonical class (mixed-load meter => useless label).
        """
        meta = self._parse_building_metadata(f, building)
        per_meter: Dict[int, set] = {}

        for app in meta.get('appliances', []):
            raw_type = str(app.get('type', '')).strip().lower()
            canonical = self.canonical_map.get(raw_type)
            if canonical is None:
                # fall back to the dataset's own original_name
                orig = str(app.get('original_name', '')).strip().lower()
                canonical = self.canonical_map.get(orig)
            for meter_id in app.get('meters', []) or []:
                try:
                    mid = int(meter_id)
                except (TypeError, ValueError):
                    continue
                per_meter.setdefault(mid, set()).add(canonical)

        labels: Dict[int, str] = {}
        for mid, classes in per_meter.items():
            if len(classes) != 1:
                continue                        # mixed-load meter -> ambiguous
            (cls,) = tuple(classes)
            if cls in self.target_classes:
                labels[mid] = cls
        return labels

    # ── Table access ─────────────────────────────────────────────────────────
    @staticmethod
    def _read_meter_series(f: h5py.File, building: str, meter: str
                           ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (timestamps_seconds, active_power_watts) for one meter."""
        try:
            table = f[building]['elec'][meter]['table']
        except (KeyError, TypeError):
            return None

        n = int(table.shape[0])
        if n < 32:
            return None
        n = min(n, MAX_ROWS_PER_METER)

        try:
            rows = table[:n]
        except Exception as e:
            logger.debug(f"{building}/{meter}: table read failed: {e}")
            return None

        try:
            t = rows['index'].astype(np.int64) / 1e9        # ns -> s
            vals = np.asarray(rows['values_block_0'], dtype=np.float32)
        except (KeyError, ValueError, IndexError) as e:
            logger.debug(f"{building}/{meter}: unexpected table dtype: {e}")
            return None

        if vals.ndim == 2:
            vals = vals[:, 0]                # column 0 == active power (W)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        return t, vals.astype(np.float32)

    # ── Windowing ────────────────────────────────────────────────────────────
    def _zoh_resample(self, t: np.ndarray, p: np.ndarray,
                      t_start: float) -> Optional[np.ndarray]:
        """
        Zero-order-hold resample [t_start, t_start+window_seconds) onto a
        1 Hz grid of seq_len points. Mirrors PZEM-004T staircase output.
        """
        grid = t_start + np.arange(self.seq_len, dtype=np.float64) / TARGET_HZ
        # index of the last real sample at or before each grid point
        idx = np.searchsorted(t, grid, side='right') - 1
        if idx[0] < 0:
            return None
        seg = p[idx]
        return seg.astype(np.float32)

    def _extract_windows(self, t: np.ndarray, p: np.ndarray) -> List[np.ndarray]:
        """Extract ON-transient windows, each WINDOW_SECONDS of wall-clock."""
        windows: List[np.ndarray] = []
        if len(t) < 8:
            return windows

        above = p > self.on_threshold_w
        # rising edges: off -> on
        edges = np.where(np.diff(above.astype(np.int8)) == 1)[0] + 1
        if len(edges) == 0:
            return windows

        if len(edges) > MAX_WINDOWS_PER_METER:
            edges = self._rng.choice(edges, MAX_WINDOWS_PER_METER, replace=False)
            edges.sort()

        # place the transient ~25% into the window so the model sees pre+post
        lead_s = 0.25 * self.window_seconds
        t_end_limit = t[-1] - self.window_seconds

        for e in edges:
            t_start = t[e] - lead_s
            if t_start < t[0] or t_start > t_end_limit:
                continue
            # reject windows straddling a long logging gap
            lo = np.searchsorted(t, t_start, side='left')
            hi = np.searchsorted(t, t_start + self.window_seconds, side='right')
            if hi - lo < 4:
                continue
            if np.max(np.diff(t[lo:hi])) > MAX_GAP_S:
                continue

            seg = self._zoh_resample(t, p, t_start)
            if seg is None:
                continue
            if not np.all(np.isfinite(seg)):
                continue
            if not self._is_useful_window(seg):
                continue
            windows.append(seg)

        return windows

    def _is_useful_window(self, seg: np.ndarray) -> bool:
        """
        Reject near-idle windows. The rising edge sits at ~25% into the window,
        so compare the pre-edge baseline against the on-portion that follows.
        """
        edge = int(0.25 * self.seq_len)
        pre, post = seg[:edge], seg[edge:]
        if pre.size == 0 or post.size == 0:
            return False

        on_mask = post > self.on_threshold_w
        if on_mask.mean() < MIN_ON_FRACTION:
            return False
        if float(np.median(post[on_mask])) < self.min_on_median_w:
            return False
        if float(np.median(post[on_mask]) - np.median(pre)) < self.min_step_w:
            return False
        return True

    # ── Public API ───────────────────────────────────────────────────────────
    def load(self) -> Dict[str, np.ndarray]:
        """
        Returns {canonical_class: (N, seq_len) float32} of real appliance
        windows resampled to the 1 Hz deployment grid.

        Also populates ``self.window_groups``: {class: (N,) array of
        "<dataset>/<building>" tags}, aligned row-for-row with the returned
        arrays. Those tags let the trainer hold out an *entire house*, which is
        the only split that measures generalisation to a home the model has
        never seen — a random split leaks the same appliance instances into
        both train and validation.
        """
        self.window_groups: Dict[str, np.ndarray] = {}

        if not os.path.exists(self.path):
            logger.warning(f"NILMTK file not found: {self.path}")
            return {}

        dataset_tag = os.path.splitext(os.path.basename(self.path))[0]
        out: Dict[str, List[np.ndarray]] = {}
        groups: Dict[str, List[str]] = {}

        try:
            with h5py.File(self.path, 'r') as f:
                buildings = sorted(k for k in f.keys() if k.startswith('building'))
                for building in buildings:
                    labels = self._meter_labels(f, building)
                    if not labels:
                        continue
                    try:
                        elec_keys = set(f[building]['elec'].keys())
                    except (KeyError, TypeError):
                        continue

                    for mid, cls in sorted(labels.items()):
                        meter = f"meter{mid}"
                        if meter not in elec_keys:
                            continue
                        series = self._read_meter_series(f, building, meter)
                        if series is None:
                            continue
                        wins = self._extract_windows(*series)
                        if wins:
                            out.setdefault(cls, []).extend(wins)
                            groups.setdefault(cls, []).extend(
                                [f"{dataset_tag}/{building}"] * len(wins))
                            logger.info(
                                f"  {os.path.basename(self.path)} {building}/{meter} "
                                f"-> {cls}: {len(wins)} windows"
                            )
        except Exception as e:
            logger.error(f"Failed reading {self.path}: {e}")
            return {}

        result = {
            cls: np.stack(w).astype(np.float32)
            for cls, w in out.items() if len(w) > 0
        }
        self.window_groups = {
            cls: np.array(groups[cls], dtype=object) for cls in result
        }
        total = sum(len(v) for v in result.values())
        logger.info(
            f"{os.path.basename(self.path)}: {total} REAL windows across "
            f"{len(result)} classes -> {sorted(result.keys())}"
        )
        return result


def load_real_windows(canonical_map: Dict[str, str], target_classes: List[str],
                      real_dir: str, datasets: Optional[List[str]] = None,
                      seq_len: int = SEQ_LEN, seed: int = 42
                      ) -> Dict[str, Dict[str, np.ndarray]]:
    """Convenience: load each requested nilmtk dataset separately."""
    datasets = datasets or ['ukdale', 'redd']
    per_source: Dict[str, Dict[str, np.ndarray]] = {}
    for name in datasets:
        path = os.path.join(real_dir, f"{name}.h5")
        reader = NILMTKReader(path, canonical_map, target_classes,
                             seq_len=seq_len, seed=seed)
        per_source[name] = reader.load()
    return per_source
