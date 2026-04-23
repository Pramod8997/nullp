"""
ESP32 Simulator — Mock 1Hz sensor data generator.
Simulates multiple devices with realistic power profiles and occasional spikes.
"""
import asyncio
import random
import logging
import argparse
import signal

import aiomqtt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Realistic device profiles
DEVICES = [
    {"id": "esp32_fridge",    "base": 150.0,  "var": 8.0,   "spike_prob": 0.01, "cutoff": 3500.0, "cycle": True, "cycle_on": 600, "cycle_off": 1200},
    {"id": "esp32_hvac",      "base": 1200.0, "var": 40.0,  "spike_prob": 0.005, "cutoff": 3500.0, "cycle": False},
    {"id": "esp32_kettle",    "base": 0.0,    "var": 0.0,   "spike_prob": 0.0,   "cutoff": 3500.0, "cycle": False, "burst": True, "burst_power": 2200.0, "burst_prob": 0.02, "burst_duration": 30},
    {"id": "esp32_tv",        "base": 95.0,   "var": 3.0,   "spike_prob": 0.0,   "cutoff": 3500.0, "cycle": False},
]


async def simulate_device(client: aiomqtt.Client, cfg: dict):
    """Simulate a single device publishing at 1Hz."""
    device_id = cfg["id"]
    topic = f"home/sensor/{device_id}/power"
    logger.info(f"▶ Simulating {device_id} on {topic}")

    tick = 0
    burst_remaining = 0

    while True:
        try:
            await asyncio.sleep(1.0)
            tick += 1

            # Cyclic devices (fridge compressor on/off)
            if cfg.get("cycle"):
                cycle_len = cfg["cycle_on"] + cfg["cycle_off"]
                if (tick % cycle_len) < cfg["cycle_on"]:
                    power = max(0.0, random.gauss(cfg["base"], cfg["var"]))
                else:
                    # Phantom load when off
                    power = random.uniform(1.0, 4.0)
            # Burst devices (kettle)
            elif cfg.get("burst"):
                if burst_remaining > 0:
                    power = random.gauss(cfg["burst_power"], 15.0)
                    burst_remaining -= 1
                elif random.random() < cfg.get("burst_prob", 0.02):
                    burst_remaining = cfg.get("burst_duration", 30)
                    power = random.gauss(cfg["burst_power"], 15.0)
                    logger.info(f"☕ {device_id} burst started ({burst_remaining}s)")
                else:
                    power = random.uniform(0.5, 2.0)  # Standby phantom
            else:
                # Normal device with some variance
                power = max(0.0, random.gauss(cfg["base"], cfg["var"]))

            # Random safety-test spike injection (rare)
            if cfg["spike_prob"] > 0 and random.random() < cfg["spike_prob"]:
                power = cfg["cutoff"] + random.uniform(100, 500)
                logger.warning(f"💉 Spike injected on {device_id}: {power:.1f}W")

            payload = f"{power:.2f}"
            await client.publish(topic, payload=payload)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in {device_id} loop: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Mock ESP32 1Hz Data Generator")
    parser.add_argument("--broker", type=str, default="localhost", help="MQTT Broker address")
    parser.add_argument("--port", type=int, default=1883, help="MQTT Broker port")
    args = parser.parse_args()

    shutdown_event = asyncio.Event()

    def handle_signal():
        logger.info("Simulator shutting down...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_signal)
        except NotImplementedError:
            pass

    while not shutdown_event.is_set():
        try:
            async with aiomqtt.Client(args.broker, port=args.port) as client:
                logger.info(f"✅ Simulator connected to {args.broker}:{args.port}")
                logger.info(f"   Devices: {[d['id'] for d in DEVICES]}")

                tasks = [asyncio.create_task(simulate_device(client, cfg)) for cfg in DEVICES]

                # Wait until shutdown or connection error
                done, pending = await asyncio.wait(
                    [asyncio.create_task(shutdown_event.wait()), *tasks],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                break

        except aiomqtt.MqttError as e:
            logger.error(f"MQTT Error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)

    logger.info("Simulator stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
