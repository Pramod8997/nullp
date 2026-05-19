"""
UK-DALE / REDD HDF5 Replay Script (Phase 2 — WS-4.1)

Reads appliance-level power data from HDF5 files and publishes
samples at 1Hz to MQTT, simulating real hardware sensors for
benchmarking and model validation.

Usage:
    python scripts/nilmtk_replay.py --file backend/data/mock_ukdale.h5
    python scripts/nilmtk_replay.py --file /path/to/ukdale.h5 --speed 10
    python scripts/nilmtk_replay.py --file /path/to/ukdale.h5 --broker 192.168.1.100

The HDF5 file should have the structure:
    /appliances/{class_name}/windows  → (N, seq_len) float32 array
"""
import os
import sys
import asyncio
import argparse
import logging
import signal

import h5py
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import aiomqtt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def replay(hdf5_path: str, broker: str = "localhost", port: int = 1883,
                 speed: float = 1.0, loop_forever: bool = False):
    """
    Replay HDF5 appliance data as MQTT messages.

    Args:
        hdf5_path:    Path to HDF5 file
        broker:       MQTT broker hostname
        port:         MQTT broker port
        speed:        Playback speed multiplier (10 = 10x faster)
        loop_forever: If True, restart from beginning when data exhausted
    """
    if not os.path.exists(hdf5_path):
        logger.error(f"HDF5 file not found: {hdf5_path}")
        return

    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler():
        if not shutdown.is_set():
            logger.info("Replay shutting down...")
            shutdown.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass

    delay = 1.0 / speed

    while not shutdown.is_set():
        try:
            async with aiomqtt.Client(broker, port=port) as client:
                logger.info(f"✅ Replay connected to {broker}:{port}")
                logger.info(f"   File: {hdf5_path}")
                logger.info(f"   Speed: {speed}x (delay: {delay:.3f}s)")

                with h5py.File(hdf5_path, 'r') as f:
                    if 'appliances' not in f:
                        logger.error("HDF5 file has no 'appliances' group")
                        return

                    appliance_names = list(f['appliances'].keys())
                    logger.info(f"   Appliances: {appliance_names}")

                    # Load all windows into memory for interleaved playback
                    all_data = {}
                    max_windows = 0
                    for name in appliance_names:
                        grp = f[f'appliances/{name}']
                        if 'windows' in grp:
                            windows = grp['windows'][:]  # (N, seq_len)
                            all_data[name] = windows
                            max_windows = max(max_windows, len(windows))
                            logger.info(f"   {name}: {windows.shape[0]} windows × {windows.shape[1]} samples")

                    if not all_data:
                        logger.error("No appliance data found in HDF5 file")
                        return

                    total_samples = 0

                    # Interleave: for each window, play all appliances simultaneously
                    for w_idx in range(max_windows):
                        if shutdown.is_set():
                            break

                        # Get the window for each appliance (wrap if one has fewer windows)
                        windows = {}
                        for name, data in all_data.items():
                            windows[name] = data[w_idx % len(data)]

                        seq_len = max(len(w) for w in windows.values())

                        # Play sample by sample
                        for s_idx in range(seq_len):
                            if shutdown.is_set():
                                break

                            for name, window in windows.items():
                                if s_idx < len(window):
                                    power = float(window[s_idx])
                                    topic = f"home/sensor/{name}/power"
                                    await client.publish(topic, f"{power:.2f}")

                            total_samples += 1
                            await asyncio.sleep(delay)

                            if total_samples % 100 == 0:
                                logger.info(
                                    f"📊 Replayed {total_samples} samples "
                                    f"(window {w_idx+1}/{max_windows})"
                                )

                    logger.info(f"✅ Replay complete: {total_samples} total samples")

                if not loop_forever:
                    break

                logger.info("🔄 Looping — restarting replay...")

        except aiomqtt.MqttError as e:
            if shutdown.is_set():
                break
            logger.error(f"MQTT Error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)

    logger.info("Replay stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Replay UK-DALE/REDD HDF5 data as MQTT messages"
    )
    parser.add_argument("--file", type=str, required=True,
                        help="Path to HDF5 file")
    parser.add_argument("--broker", type=str, default="localhost",
                        help="MQTT broker hostname")
    parser.add_argument("--port", type=int, default=1883,
                        help="MQTT broker port")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (e.g., 10 = 10x faster)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop forever (restart when data exhausted)")
    args = parser.parse_args()

    asyncio.run(replay(
        hdf5_path=args.file,
        broker=args.broker,
        port=args.port,
        speed=args.speed,
        loop_forever=args.loop,
    ))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
