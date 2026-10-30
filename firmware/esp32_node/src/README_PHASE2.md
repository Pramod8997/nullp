# ESP32 Firmware — Phase 2 Implementation Plan

## Current Status (Phase 1)
The ESP32 firmware at `main.cpp` is a reference implementation for the physical hardware layer. In Phase 1, all device simulation is handled by `backend/scripts/simulate_esp32.py`, which publishes synthetic 1Hz power data to MQTT.

## Phase 2: Hardware Deployment

> **Current bench build:** ESP32 DevKit V1 (30-pin). Boot strap pin GPIO 12 is
> strictly avoided; GPIO 16/17 are reserved exclusively for PZEM UART2.
> Use `docs/ems_hardware/ACTUAL_100A_CT_WIRING.md` for the current low-voltage
> pinout and `config/config.hardware.yaml` for the 250 W hardware profile.

### Architecture
- **Tier-0 Safety Relay**: Runs independently on ESP32 Core 0 (`SafetySamplingTask`, 100 ms). NO ML inference on device — the ESP32 publishes power readings via MQTT and trips the relay locally.
- **Relay Trigger**: At **125% of `RATED_WATTS`** (`CRITICAL_PCT = 1.25`, matching `critical_pct` in config). The relay disconnects on-device, independent of the MQTT broker or the ML pipeline.
- **Metering**: The available **PZEM-004T-100A(V4.0)-type external-CT variant** is read over the v3/v4-compatible UART Modbus register protocol. Its external CT surrounds one live conductor; its `L/N` terminals are voltage sense. The ESP32 reads digital V / I / P / PF registers; it does not calculate ADC RMS.

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
1. Flash `main.cpp` to the **ESP32 DevKit V1 (30-pin)** — **WROOM-32, not WROVER** (WROVER uses GPIO 16/17 for PSRAM, which are the PZEM UART pins)
2. Wire the PZEM UART: GPIO 16 (RX2) ← PZEM TX, GPIO 17 (TX2) → PZEM RX (3.3V logic supply to PZEM 3V3 pin, 9600 baud, 8N1)
3. Wire the 2-channel active-LOW relay module:
   - IN1 (Relay 1, Laptop Charger) ← GPIO 26
   - IN2 (Relay 2, Phone Charger) ← GPIO 27
   - Opto VCC ← GPIO 25 (driven HIGH in setup() as 3.3V source for opto LEDs, few mA)
   - Coil JD-VCC ← External 5V / VIN (jumper removed between VCC and JD-VCC)
   - GND ← Common DC GND
   - Polarity is **ACTIVE-LOW** (`RELAY_ON = LOW`, `RELAY_OFF = HIGH`). Startup order guarantees relays stay OFF through boot and reset.
4. Set `DEVICE_ID`, `RATED_WATTS`, WiFi credentials and MQTT broker IP in firmware
5. Remove `simulate_esp32.py` from the `Makefile` run target
6. Run the pipeline with `--config config/config.hardware.yaml` (250 W ceiling). Do not use `config.demo.yaml` for the physical rig; the 600 W ceiling would make the hardware cutoff unreachable.

### Hardware BOM
Use the current procurement report and verify every module against the wiring
gates before ordering. The older 30-pin BOM text in
[`claude_debug/HARDWARE_FINAL_SPEC.md`](../../../claude_debug/HARDWARE_FINAL_SPEC.md)
is not a compatible mechanical specification for the current 38-pin board.
