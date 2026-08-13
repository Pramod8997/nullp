"""
Extract per-appliance windows from unlabeled aggregate meter data.

The Zenodo UK-DALE and REDD tensors contain raw meter-level power readings
without per-appliance labels. This script uses event-based disaggregation
to extract transient events and cluster them into appliance classes based
on power signature characteristics.

This is a heuristic approach — not as accurate as NILMTK disaggregation,
but sufficient for training data augmentation alongside labeled synthetic data.

Usage:
    source .venv/bin/activate
    python scripts/extract_real_data.py
"""
import os
import sys
import logging
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REAL_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'real')
WINDOW = 128


def detect_events(power: np.ndarray, threshold: float = 50.0,
                  min_gap: int = 30) -> list:
    """
    Detect ON-events in a power time-series.
    
    Returns list of (start_idx, peak_power, duration) tuples.
    """
    events = []
    above = power > threshold
    transitions = np.diff(above.astype(int))
    on_idxs = np.where(transitions == 1)[0]
    off_idxs = np.where(transitions == -1)[0]
    
    for on_idx in on_idxs:
        # Find corresponding off event
        offs = off_idxs[off_idxs > on_idx]
        if len(offs) > 0:
            off_idx = offs[0]
            duration = off_idx - on_idx
            if duration >= min_gap:
                peak = np.max(power[on_idx:off_idx])
                events.append((on_idx, peak, duration))
    
    return events


def classify_event(peak_power: float, duration: int, 
                   segment: np.ndarray) -> str:
    """
    Classify an event based on power characteristics.
    
    Uses power magnitude, duration, and shape to assign a likely appliance class.
    This is a heuristic — real NILMTK would be more accurate.
    """
    mean_power = np.mean(segment[segment > 20])  # Mean active power
    variance = np.std(segment)
    
    # High power, short duration → kettle or microwave
    if peak_power > 2000 and duration < 600:
        if variance < 200:
            return 'kettle'  # Steady high power
        else:
            return 'oven'  # Variable high power
    
    # Very high power → ev_charger
    if peak_power > 3000:
        return 'ev_charger'
    
    # High power, long duration → HVAC or oven
    if peak_power > 1500:
        if duration > 1800:  # > 30 min
            return 'hvac'
        else:
            return 'oven'
    
    # Medium-high power with cycling → washing machine or dishwasher
    if 500 < peak_power < 2000 and variance > 300:
        if duration > 3600:  # > 1 hour
            return 'dishwasher'
        else:
            return 'washing_machine'
    
    # Medium power → microwave
    if 800 < peak_power < 1500 and duration < 600:
        return 'microwave'
    
    # Low power, continuous → fridge
    if 100 < peak_power < 300 and duration > 600:
        return 'fridge'
    
    # Very low power → laptop/tv
    if peak_power < 200:
        if mean_power < 100:
            return 'laptop'
        else:
            return 'tv'
    
    # Default
    return 'washing_machine'


def extract_from_tensor(tensor_path: str, source_name: str) -> dict:
    """
    Extract per-appliance windows from a Zenodo tensor file.
    
    Returns:
        {class_name: np.ndarray of shape (N, WINDOW)}
    """
    if not os.path.exists(tensor_path):
        logger.warning(f"File not found: {tensor_path}")
        return {}
    
    try:
        data = np.load(tensor_path, allow_pickle=True)
    except ValueError:
        # Handle shape mismatch — try memmap
        import struct
        with open(tensor_path, 'rb') as f:
            magic = f.read(6)
            version = f.read(2)
            if version[0] == 1:
                header_len = struct.unpack('<H', f.read(2))[0]
            else:
                header_len = struct.unpack('<I', f.read(4))[0]
            header = f.read(header_len).decode('latin1')
            # Parse shape from header
            shape_str = header.split("'shape': ")[1].split('}')[0].strip().rstrip(',')
            shape = eval(shape_str)
            offset = f.tell()
        
        data = np.memmap(tensor_path, dtype=np.float64, mode='r',
                        offset=offset, shape=shape)
    
    n_meters = data.shape[0]
    results = defaultdict(list)
    
    logger.info(f"Processing {source_name}: shape={data.shape}")
    
    for meter_idx in range(n_meters):
        power = data[meter_idx, :, 0].copy()  # Active power (col 0)
        power = np.nan_to_num(power, nan=0.0)
        
        if np.max(power) < 20:
            continue
        
        # Detect events
        events = detect_events(power, threshold=30.0, min_gap=20)
        
        for start_idx, peak, duration in events:
            # Extract window centered on event start
            half = WINDOW // 2
            win_start = max(0, start_idx - half)
            win_end = win_start + WINDOW
            if win_end > len(power):
                win_start = len(power) - WINDOW
                win_end = len(power)
            if win_start < 0:
                continue
            
            segment = power[win_start:win_end].astype(np.float32)
            if len(segment) != WINDOW:
                continue
            
            # Classify
            cls = classify_event(peak, duration, segment)
            results[cls].append(segment)
    
    # Convert to arrays
    output = {}
    for cls, segments in results.items():
        arr = np.array(segments, dtype=np.float32)
        if len(arr) >= 10:  # Minimum viable sample count
            output[cls] = arr
            logger.info(f"  {source_name}/{cls}: {len(arr)} windows extracted")
    
    return output


def main():
    os.makedirs(REAL_DIR, exist_ok=True)
    all_data = defaultdict(list)
    
    # Process UK-DALE tensor
    ukdale_path = os.path.join(REAL_DIR, 'ukdale_tensor.npy')
    if os.path.exists(ukdale_path):
        ukdale_data = extract_from_tensor(ukdale_path, 'UK-DALE')
        for cls, arr in ukdale_data.items():
            all_data[cls].append(arr)
    
    # Process REDD tensor
    redd_path = os.path.join(REAL_DIR, 'redd_tensor.npy')
    if os.path.exists(redd_path):
        redd_data = extract_from_tensor(redd_path, 'REDD')
        for cls, arr in redd_data.items():
            all_data[cls].append(arr)
    
    # Save combined per-class files
    output_path = os.path.join(REAL_DIR, 'extracted_appliances.npz')
    save_dict = {}
    for cls, arrays in sorted(all_data.items()):
        combined = np.concatenate(arrays, axis=0)
        save_dict[cls] = combined
        logger.info(f"Combined {cls}: {len(combined)} total windows")
    
    if save_dict:
        np.savez(output_path, **save_dict)
        logger.info(f"Saved to {output_path}")
    else:
        logger.warning("No data extracted!")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTION COMPLETE")
    total = sum(len(v) for v in save_dict.values())
    logger.info(f"Total: {total} windows across {len(save_dict)} classes")
    for cls in sorted(save_dict.keys()):
        logger.info(f"  {cls:20s}: {len(save_dict[cls]):5d} windows")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
