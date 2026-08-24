# ESP32 Firmware — Phase 2 Implementation Plan

## Current Status (Phase 1)
The ESP32 firmware at `main.cpp` is a reference implementation for the physical hardware layer. In Phase 1, all device simulation is handled by `backend/scripts/simulate_esp32.py`, which publishes synthetic 1Hz power data to MQTT.

## Phase 2: Hardware Deployment

### Architecture
- **Tier-0 Safety Relay**: Runs independently on ESP32 Core 0 (`SafetySamplingTask`, 100 ms). NO ML inference on device — the ESP32 publishes power readings via MQTT and trips the relay locally.
- **Relay Trigger**: At **125% of `RATED_WATTS`** (`CRITICAL_PCT = 1.25`, matching `critical_pct` in config). The relay disconnects on-device, independent of the MQTT broker or the ML pipeline.
- **Metering**: **PZEM-004T v3.0, 10 A direct-connect variant**, read over UART Modbus RTU. Load current passes through the module's screw terminals; the ESP32 reads true-RMS V / I / P / PF / kWh registers digitally.

> ⚠️ **There is no analog CT path.** An earlier revision of this file specified an
> SCT-013-030 clamp into GPIO 34 (ADC) with a 33 Ω burden resistor, the relay on
> GPIO 5, and a USB supply. That was wrong on three counts: the sensor is a digital
> UART module rather than an analog clamp, GPIO 5 is an ESP32 strapping pin, and a
> USB supply on a mains-connected node is a shock hazard. Authoritative spec:
> [`claude_debug/HARDWARE_FINAL_SPEC.md`](../../../claude_debug/HARDWARE_FINAL_SPEC.md).

### Communication
- MQTT publish: `home/sensor/{device_id}/power` at 1Hz (matching UK-DALE sample rate)
- MQTT subscribe: `home/plug/{device_id}/command` for RL agent control (ON/OFF)
- WiFi reconnection with exponential backoff

### Replacing the Simulator
1. Flash `main.cpp` to the ESP32 DevKit V1 — **WROOM-32D, not WROVER** (WROVER uses GPIO 16/17 for PSRAM, which are the PZEM UART pins)
2. Wire the PZEM UART: GPIO 16 (RX2) ← PZEM TX, GPIO 17 (TX2) → PZEM RX. **Measure PZEM TX idle voltage first** — 5 V push-pull variants exceed the ESP32's 3.6 V absolute maximum (issue B-4)
3. Wire the relay: GPIO 18 → 1 kΩ → BSS138 gate, **100 kΩ gate→GND pull-down**, drain → relay IN, source → GND. Net polarity at the GPIO is **active-HIGH**, so `RELAY_ACTIVE_LOW = false`
4. Set `DEVICE_ID`, `RATED_WATTS`, WiFi credentials and MQTT broker IP in firmware
5. Remove `simulate_esp32.py` from the `Makefile` run target
6. Run the pipeline with `--config config/config.demo.yaml` (600 W ceiling). The rest of `run_pipeline.py` works unchanged — it subscribes to the `home/sensor/+/power` wildcard

### Hardware BOM
Single source of truth: [`claude_debug/HARDWARE_FINAL_SPEC.md`](../../../claude_debug/HARDWARE_FINAL_SPEC.md) §2. Do not order parts from this file.
