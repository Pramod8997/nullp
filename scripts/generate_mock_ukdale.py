import os
import numpy as np
import h5py
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import yaml

DATA_DIR = "backend/data"

# Read seq_len from config
try:
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        WINDOW_SIZE = config.get("protonet", {}).get("seq_len", 128)
except Exception:
    WINDOW_SIZE = 128

NUM_WINDOWS = 500  # 500 episodes worth of windows per class

CLASSES = {
    "node_fridge": {"rated": 200.0, "tau": 2.0},
    "node_microwave": {"rated": 1200.0, "tau": 0.5},
    "node_kettle": {"rated": 2500.0, "tau": 1.0},
    "node_hvac": {"rated": 2000.0, "tau": 5.0},
    "esp32_tv": {"rated": 150.0, "tau": 1.0},
    "esp32_washer": {"rated": 1800.0, "tau": 3.0},
    "esp32_dryer": {"rated": 2000.0, "tau": 4.0},
    "esp32_dishwasher": {"rated": 1500.0, "tau": 3.0},
    "esp32_oven": {"rated": 3000.0, "tau": 2.0},
    "esp32_lighting": {"rated": 100.0, "tau": 0.1},
}

def generate_window(class_name, config):
    rated = config["rated"]
    tau = config["tau"]
    std = rated * 0.03
    
    window = np.zeros(WINDOW_SIZE)
    
    # Class-specific signatures (simplified representations within a window)
    if class_name == "node_fridge":
        # Compressor kick-in
        t = np.arange(WINDOW_SIZE)
        window = rated * (1 - np.exp(-t / tau))
    elif class_name == "node_microwave":
        # Instant ON, pulse
        window[5:55] = rated
    elif class_name == "node_kettle":
        # Steady
        window[10:] = rated
    elif class_name == "node_hvac":
        # Variable with startup peak
        window[5:] = rated + 500 * np.exp(-np.arange(WINDOW_SIZE - 5) / (tau / 2))
    elif class_name == "esp32_tv":
        # Steady
        window[5:] = rated
    elif class_name == "esp32_washer":
        # Complex steps
        window[5:25] = 200  # fill
        window[25:45] = 800  # wash
        window[45:] = rated  # spin
    elif class_name == "esp32_dryer":
        # Steady with cycles
        window[5:] = rated
        window[20:30] = 0
        window[40:50] = 0
    elif class_name == "esp32_dishwasher":
        # Heating element cycles
        window[10:30] = rated
        window[40:] = rated
    elif class_name == "esp32_oven":
        # Thermostat on/off
        window[5:20] = rated
        window[30:45] = rated
    elif class_name == "esp32_lighting":
        # Step changes
        window[10:30] = rated / 2
        window[30:] = rated
    else:
        # Default transient
        t_start = 10
        t = np.arange(WINDOW_SIZE - t_start)
        window[t_start:] = rated * (1 - np.exp(-t / tau))
        
    # Add noise
    noise = np.random.normal(0, std, WINDOW_SIZE)
    window += noise
    
    # Ensure no negative power
    return np.clip(window, 0, None)

def generate_mock_ukdale():
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, "mock_ukdale.h5")
    
    logger.info(f"Generating synthetic UK-DALE data for {len(CLASSES)} classes...")
    
    class_names = list(CLASSES.keys())
    
    with h5py.File(file_path, 'w') as hf:
        # Metadata
        meta_grp = hf.create_group('metadata')
        meta_grp.create_dataset('class_names', data=np.array(class_names, dtype='S'))
        meta_grp.create_dataset('sample_rate_hz', data=1)
        
        appliances_grp = hf.create_group('appliances')
        
        for class_idx, class_name in enumerate(tqdm(class_names, desc="Classes")):
            grp = appliances_grp.create_group(class_name)
            config = CLASSES[class_name]
            
            windows = []
            for _ in range(NUM_WINDOWS):
                win = generate_window(class_name, config)
                windows.append(win)
                
            windows = np.array(windows, dtype=np.float32)
            labels = np.full(NUM_WINDOWS, class_idx, dtype=np.int32)
            
            grp.create_dataset('windows', data=windows)
            grp.create_dataset('labels', data=labels)

    logger.info(f"Successfully generated {file_path}")

if __name__ == "__main__":
    generate_mock_ukdale()
