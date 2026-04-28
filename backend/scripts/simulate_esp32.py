"""
ESP32 Simulator — Mock 1Hz sensor data generator.
Simulates 10 devices with realistic power profiles, duty cycles, and noise.
Publishes at 1 Hz to MQTT topic: home/sensor/{device_id}/power
"""
import asyncio
import random
import logging
import argparse
import signal
import math

import aiomqtt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# All 10 device profiles matching config.yaml
DEVICES = [
    {
        "id": "esp32_fridge", "rated": 200.0, "var": 6.0,
        "spike_prob": 0.005, "cutoff": 3500.0,
        "cycle": True, "cycle_on": 600, "cycle_off": 900,
        "phantom": 3.0,
    },
    {
        "id": "esp32_microwave", "rated": 1200.0, "var": 15.0,
        "spike_prob": 0.0, "cutoff": 3500.0,
        "cycle": False, "burst": True,
        "burst_power": 1200.0, "burst_prob": 0.015, "burst_duration": 45,
        "phantom": 2.0,
    },
    {
        "id": "esp32_kettle", "rated": 2500.0, "var": 20.0,
        "spike_prob": 0.0, "cutoff": 3500.0,
        "cycle": False, "burst": True,
        "burst_power": 2500.0, "burst_prob": 0.01, "burst_duration": 60,
        "phantom": 1.0,
    },
    {
        "id": "esp32_hvac", "rated": 2000.0, "var": 50.0,
        "spike_prob": 0.003, "cutoff": 3500.0,
        "cycle": True, "cycle_on": 1800, "cycle_off": 600,
        "phantom": 10.0,
        "startup_peak": 500.0, "startup_duration": 10,
    },
    {
        "id": "esp32_tv", "rated": 150.0, "var": 3.0,
        "spike_prob": 0.0, "cutoff": 3500.0,
        "cycle": False, "on_prob": 0.002, "off_prob": 0.001,
        "phantom": 5.0,
    },
    {
        "id": "esp32_washer", "rated": 1800.0, "var": 30.0,
        "spike_prob": 0.0, "cutoff": 3500.0,
        "cycle": False, "burst": True,
        "burst_power": 1800.0, "burst_prob": 0.008, "burst_duration": 180,
        "phantom": 2.0,
        "multi_phase": True,
        "phases": [
            {"power": 200.0, "duration": 60},   # fill
            {"power": 800.0, "duration": 60},   # wash
            {"power": 1800.0, "duration": 60},  # spin
        ],
    },
    {
        "id": "esp32_dryer", "rated": 2000.0, "var": 40.0,
        "spike_prob": 0.0, "cutoff": 3500.0,
        "cycle": True, "cycle_on": 2400, "cycle_off": 1200,
        "phantom": 3.0,
        "drum_pause": True, "drum_off_period": 120, "drum_off_duration": 30,
    },
    {
        "id": "esp32_dishwasher", "rated": 1500.0, "var": 25.0,
        "spike_prob": 0.0, "cutoff": 3500.0,
        "cycle": False, "burst": True,
        "burst_power": 1500.0, "burst_prob": 0.006, "burst_duration": 240,
        "phantom": 2.0,
        "heating_cycle": True, "heat_period": 120,
    },
    {
        "id": "esp32_oven", "rated": 3000.0, "var": 50.0,
        "spike_prob": 0.002, "cutoff": 3500.0,
        "cycle": False, "burst": True,
        "burst_power": 3000.0, "burst_prob": 0.005, "burst_duration": 120,
        "phantom": 1.0,
        "thermostat": True, "thermostat_period": 60,
    },
    {
        "id": "esp32_lighting", "rated": 100.0, "var": 2.0,
        "spike_prob": 0.0, "cutoff": 3500.0,
        "cycle": False, "on_prob": 0.003, "off_prob": 0.001,
        "phantom": 1.5,
        "dimming": True,
    },
]


async def simulate_device(client: aiomqtt.Client, cfg: dict):
    """Simulate a single device publishing at 1Hz."""
    device_id = cfg["id"]
    topic = f"home/sensor/{device_id}/power"
    logger.info(f"▶ Simulating {device_id} on {topic}")

    tick = 0
    burst_remaining = 0
    is_on = cfg.get("cycle", False)  # Cyclic devices start ON
    phase_idx = 0
    phase_remaining = 0

    while True:
        try:
            await asyncio.sleep(1.0)
            tick += 1

            # ── Cyclic devices (fridge compressor, HVAC, dryer) ──
            if cfg.get("cycle"):
                cycle_len = cfg["cycle_on"] + cfg["cycle_off"]
                cycle_pos = tick % cycle_len
                if cycle_pos < cfg["cycle_on"]:
                    is_on = True
                    base = cfg["rated"]

                    # HVAC startup peak
                    if cfg.get("startup_peak") and cycle_pos < cfg.get("startup_duration", 10):
                        base += cfg["startup_peak"]

                    # Dryer drum pauses
                    if cfg.get("drum_pause"):
                        drum_cycle = tick % cfg.get("drum_off_period", 120)
                        if drum_cycle < cfg.get("drum_off_duration", 30):
                            base = 0.0

                    power = max(0.0, random.gauss(base, cfg["var"]))
                else:
                    is_on = False
                    power = random.uniform(0.5, cfg.get("phantom", 3.0))

            # ── Multi-phase devices (washer) ──
            elif cfg.get("multi_phase") and burst_remaining > 0:
                phases = cfg["phases"]
                if phase_remaining <= 0 and phase_idx < len(phases) - 1:
                    phase_idx += 1
                    phase_remaining = phases[phase_idx]["duration"]
                current_phase = phases[min(phase_idx, len(phases) - 1)]
                power = max(0.0, random.gauss(current_phase["power"], cfg["var"]))
                burst_remaining -= 1
                phase_remaining -= 1
                is_on = True

            # ── Burst devices (microwave, kettle, etc.) ──
            elif cfg.get("burst"):
                if burst_remaining > 0:
                    base = cfg["burst_power"]

                    # Oven thermostat cycling
                    if cfg.get("thermostat"):
                        period = cfg.get("thermostat_period", 60)
                        if (tick % period) > period * 0.6:
                            base = 0.0  # Element off during coast

                    # Dishwasher heating element cycling
                    if cfg.get("heating_cycle"):
                        period = cfg.get("heat_period", 120)
                        if (tick % period) > period * 0.5:
                            base *= 0.3  # Reduced during rinse

                    power = max(0.0, random.gauss(base, cfg["var"]))
                    burst_remaining -= 1
                    is_on = True
                elif random.random() < cfg.get("burst_prob", 0.02):
                    total_burst = cfg.get("burst_duration", 30)
                    burst_remaining = total_burst
                    phase_idx = 0
                    if cfg.get("multi_phase"):
                        phase_remaining = cfg["phases"][0]["duration"]
                    power = max(0.0, random.gauss(cfg["burst_power"], cfg["var"]))
                    is_on = True
                    logger.info(f"☕ {device_id} burst started ({burst_remaining}s)")
                else:
                    power = random.uniform(0.5, cfg.get("phantom", 2.0))
                    is_on = False

            # ── On/Off toggle devices (TV, lighting) ──
            elif cfg.get("on_prob"):
                if not is_on and random.random() < cfg["on_prob"]:
                    is_on = True
                elif is_on and random.random() < cfg.get("off_prob", 0.001):
                    is_on = False

                if is_on:
                    base = cfg["rated"]
                    # Lighting dimming
                    if cfg.get("dimming"):
                        dim_factor = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(tick / 300.0))
                        base *= dim_factor
                    power = max(0.0, random.gauss(base, cfg["var"]))
                else:
                    power = random.uniform(0.5, cfg.get("phantom", 2.0))

            else:
                power = max(0.0, random.gauss(cfg.get("rated", 100), cfg["var"]))

            # ── Rare safety-test spike injection ──
            if cfg["spike_prob"] > 0 and random.random() < cfg["spike_prob"]:
                power = cfg["cutoff"] + random.uniform(100, 500)
                logger.warning(f"💉 Spike injected on {device_id}: {power:.1f}W")

            # Add Gaussian noise to simulate sensor noise
            power += random.gauss(0, max(1.0, cfg["rated"] * 0.01))
            power = max(0.0, power)

            payload = f"{power:.2f}"
            await client.publish(topic, payload=payload)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in {device_id} loop: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Mock ESP32 1Hz Data Generator (10 devices)")
    parser.add_argument("--broker", type=str, default="localhost", help="MQTT Broker address")
    parser.add_argument("--port", type=int, default=1883, help="MQTT Broker port")
    args = parser.parse_args()

    shutdown_event = asyncio.Event()
    _shutdown_logged = False

    def handle_signal():
        nonlocal _shutdown_logged
        if not _shutdown_logged:
            logger.info("Simulator shutting down...")
            _shutdown_logged = True
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
                logger.info(f"   Devices ({len(DEVICES)}): {[d['id'] for d in DEVICES]}")

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
