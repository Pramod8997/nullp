# Hardware Pinout & Wiring Specification

> **Rig:** Single-node, **single-socket** prototype — laptop + phone charger, ≤250 W, India 230 V 50 Hz.
> **Current board:** 38-pin ESP32-WROOM-32 DevKit. The 38-pin low-voltage guide is [`docs/archive/claude_debug/WIRING_STEP_BY_STEP.md`](./docs/archive/docs/archive/claude_debug/WIRING_STEP_BY_STEP.md); older 30-pin references are not mechanically compatible.
> **Authoritative current build:** [`docs/ems_hardware/ACTUAL_100A_CT_WIRING.md`](./docs/ems_hardware/ACTUAL_100A_CT_WIRING.md). The older 10 A shunt specification is historical and must not be mixed with the available CT meter.
>
> **Build class:** zero soldering, integrated modules only, ≤₹3,000. Production
> pins are already assigned; the repository also provides a separate compiled
> `bench` firmware for mains-disconnected relay continuity testing.

* **PZEM-004T v4.0 (100A) External-CT Digital Meter:**
  * **TTL Header (Low Voltage)**:
    * 5V -> External 5V Rail (ESP32 VIN)
    * GND -> Common DC GND
    * TX -> 1kΩ / 2kΩ voltage divider (or 5V-to-3.3V level shifter) -> ESP32 GPIO 16 (UART2 RX)
    * RX -> ESP32 GPIO 17 (UART2 TX)
  * **High-Voltage Screw Terminals (PZEM-004T v4.0)**:
    * Terminal 1 (L): 230V AC Live (via 1A fast-blow inline fuse)
    * Terminal 2 (N): 230V AC Neutral
    * Terminal 3 (CT+): 100A Split-Core CT Red wire
    * Terminal 4 (CT-): 100A Split-Core CT Black wire
    * *Safety Note*: Screw terminals 3 & 4 tightly BEFORE clamping the CT around the single insulated Live (L) wire. Note that the PZEM-004T AC side must be powered by 230V AC for the onboard metering ASIC to respond to UART polls.
  * MCU must be **ESP32-WROOM-32**, **38-pin DevKit** (or 30-pin DevKit V1). **Not WROVER** — WROVER uses GPIO 16/17 for PSRAM.
* **4-Channel Active-LOW Opto-Isolated Relay Module**:
  * VCC -> 5V Rail (>= 2A external supply recommended; or JD-VCC = 5V and VCC = 3.3V with jumper removed for full optical isolation)
  * GND -> Common DC GND
  * IN1 -> GPIO 26
  * IN2 -> GPIO 27
  * IN3 -> GPIO 14
  * IN4 -> GPIO 32 (CRITICAL FIX: Moved off strapping pin GPIO 12 / MTDI, which would pull flash voltage to 1.8V at boot and brick the ESP32)
  * **Logic Polarity**: **ACTIVE-LOW** (RELAY_ON = LOW, RELAY_OFF = HIGH). Pins are pre-initialized to HIGH (RELAY_OFF) in firmware setup before pin direction configuration to prevent boot-up relay chatter. Fail-safe autonomous trip instantly trips all channels HIGH.
  * RC snubber (100 Ω 2 W flameproof + 0.1 µF 275 VAC Class-X2 across COM–NO): **optional** for this load set — no inductive load remains in scope. Reinstate for any motor/fan/pump/transformer load.

* **Wiring interface:**
   * ESP32 plugs into a **38-pin screw-terminal expansion shield** — match pin count *and* board width. Pins are only broken out; no remapping. A 30-pin shield must not be forced onto this board.
* **Power Supply (5 V Rail):**
   * **BIS-marked 5 V 2 A USB charger** ──> USB-A-female-to-screw-terminal breakout ──> shield VIN / GND. The adapter/cable alone is not this breakout.
  * Charger lives **outside** the enclosure on its own wall socket. This deletes the HLK-5M05, the MOV, the 0.5 A PSU fuse, the 1000 µF bulk cap, the 100 nF cap and the entire second mains tap.
  * ESP32 VIN (5 V), Relay VCC (+5 V), PZEM VCC (+5 V) from adjacent terminals
  * ⚠️ **Laptop USB and mains are mutually exclusive.** Flash on laptop USB with mains disconnected; run on the 5 V charger with the laptop cable removed. The 2-pin charger has no earth reference and is the intended run-time supply.
 * **AC Mains & Protective Earth (PE) — electrician-only orientation:**
  * For the actual CT meter: **fuse ──> PZEM L voltage sense / relay ──> CT around live ──> socket.** Do not route load current through PZEM `L/N` as a shunt path.
  * Mains Live (L) ──> 5 A ceramic fuse ──┬─> PZEM `L` sense
    * └─> Relay COM ──[ Relay NO ]─(CT around live conductor)─> Socket Live
  * Mains Neutral (N) ──> PZEM `N` sense and Socket Neutral
  * Protective Earth (PE) ──> PE barrier block (enclosure-bonded) ──> Socket PE (**unswitched**)
  * Load socket: **one** IS 1293 6 A 3-pin panel mount, with an **earthed 3-pin multi-plug adapter** in it
  * Upstream: **30 mA RCBO** (portable adapter, or a verified house-DB RCCB)
 * **The two loads share one socket, simultaneously** — laptop brick + phone charger through the same external CT. That overlap is what NILM disaggregates; one device at a time is single-appliance classification and never exercises `OverlapAwareNILMDetector`.

**Coordination ladder** — must stay monotonic, see spec §D8:
`185 W nominal (0.80 A) → WARNING 275 W → relay trips 312 W (1.36 A) → 5 A fuse → verified relay/socket rating → 1.0 mm² mains wiring`

**Run the pipeline with the hardware config, not the demo config:**
```bash
python scripts/run_pipeline.py --config config/config.hardware.yaml   # 250 W ceiling, real rig
```
`config/config.demo.yaml` keeps a 600 W ceiling for its ~1130 W *simulated* fleet. Using it on the real rig puts CRITICAL at 750 W, which a 250 W prototype can never reach — the cutoff would never fire.
