import os
import numpy as np
import pandas as pd
import logging
import h5py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = "backend/data"

def generate_mock_ukdale(num_rows=10000):
    """
    Generates a synthetic UK-DALE dataset with aggregate mains and isolated appliances.
    Injects synthetic step-changes (microwaves, kettles) for CNN/ProtoNet feature extraction.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, "mock_ukdale.h5")
    
    logger.info(f"Generating {num_rows} rows of synthetic UK-DALE data...")
    
    # 1Hz timestamp spanning roughly 2.7 hours
    timestamps = pd.date_range(start='2026-01-01', periods=num_rows, freq='S')
    
    # Base noise floor (always on devices like routers)
    base_noise = np.random.normal(loc=50.0, scale=2.0, size=num_rows)
    
    # Appliance 1: Fridge (cyclic compressor)
    # Cycles on for 600 seconds, off for 1200 seconds. Draw: ~150W
    appliance_1 = np.zeros(num_rows)
    for i in range(num_rows):
        if (i % 1800) < 600:
            appliance_1[i] = np.random.normal(150.0, 5.0)
            
    # Appliance 2: Microwave (Short, massive spike)
    # Spikes for 60 seconds every ~3000 seconds. Draw: ~1200W
    appliance_2 = np.zeros(num_rows)
    for i in range(num_rows):
        if (i % 3000) > 2900:
            appliance_2[i] = np.random.normal(1200.0, 15.0)
            
    # Aggregate Mains = Base Noise + Sum of Appliances + Minor mains noise
    aggregate = base_noise + appliance_1 + appliance_2 + np.random.normal(0, 1.0, num_rows)
    
    logger.info("Saving to HDF5 format...")
    
    with h5py.File(file_path, 'w') as hf:
        # Mimicking the hierarchical structure
        building1 = hf.create_group('building1')
        elec = building1.create_group('elec')
        
        # Mains
        elec.create_dataset('mains', data=aggregate)
        
        # Appliances
        elec.create_dataset('meter1', data=appliance_1)
        elec.create_dataset('meter2', data=appliance_2)
        
        # Timestamps as unix epoch
        timestamps_unix = timestamps.astype(np.int64) // 10**9
        elec.create_dataset('timestamps', data=timestamps_unix)

    logger.info(f"Successfully generated {file_path}")

if __name__ == "__main__":
    generate_mock_ukdale()
