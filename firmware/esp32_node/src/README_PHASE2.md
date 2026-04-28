# ESP32 Firmware — Phase 2 Implementation Plan

## Current Status (Phase 1)
The ESP32 firmware at `main.cpp` is a reference implementation for the physical hardware layer. In Phase 1, all device simulation is handled by `backend/scripts/simulate_esp32.py`, which publishes synthetic 1Hz power data to MQTT.

## Phase 2: Hardware Deployment

### Architecture
- **Tier-0 Safety Relay**: Runs independently on ESP32 via hardwired comparator circuit. NO ML inference on device — the ESP32 only publishes power readings via MQTT.
- **Relay Trigger**: At **125% of rated wattage** (matching `config.yaml` `critical_pct: 1.25`), the relay physically disconnects the load. This happens at the hardware level, independent of the MQTT broker or ML pipeline.
- **CT Clamp Sensors**: Each ESP32 node reads current via a CT clamp (SCT-013-030) and computes power using `P = V * I * PF` where V=230V and PF is estimated.

### Communication
- MQTT publish: `home/sensor/{device_id}/power` at 1Hz (matching UK-DALE sample rate)
- MQTT subscribe: `home/plug/{device_id}/command` for RL agent control (ON/OFF)
- WiFi reconnection with exponential backoff

### Replacing the Simulator
1. Flash `main.cpp` to each ESP32 DevKit
2. Connect CT clamp to GPIO 34 (ADC1_CH6)
3. Connect relay module to GPIO 5
4. Configure WiFi and MQTT broker IP in firmware
5. Remove `simulate_esp32.py` from `Makefile` run target
6. The rest of the pipeline (`run_pipeline.py`) works unchanged

### Hardware BOM (per node)
- ESP32 DevKit V1
- SCT-013-030 CT clamp (30A)
- Burden resistor (33Ω)
- Relay module (5V, 10A)
- 3.5mm jack for CT clamp
- 5V USB power supply
