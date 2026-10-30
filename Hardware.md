# Hardware Pinout & Wiring Specification

> **Rig:** Single-node prototype — laptop charger (Relay 1) + phone charger (Relay 2), ≤250 W, India 230 V 50 Hz.
> **Current board:** ESP32 DevKit V1 (30-pin). Boot strap pin GPIO 12 is strictly avoided; GPIO 16/17 are reserved exclusively for PZEM UART2.
> **Authoritative current build:** [`docs/ems_hardware/ACTUAL_100A_CT_WIRING.md`](./docs/ems_hardware/ACTUAL_100A_CT_WIRING.md).
>
> **Build class:** zero soldering, integrated modules only, ≤₹3,000. Production
> pins are already assigned; the repository also provides a separate compiled
> `bench` firmware for mains-disconnected relay continuity testing.

* **PZEM-004T v4.0 (100A) External-CT Digital Meter:**
  * **TTL Header (Low Voltage)**:
    * 3V3 / VCC -> ESP32 3V3 pin (3.3 V logic)
    * GND -> Common DC GND
    * TX -> ESP32 GPIO 16 (UART2 RX, wired to PZEM TX)
    * RX -> ESP32 GPIO 17 (UART2 TX, wired to PZEM RX)
    * Default address: `0xF8`, 9600 baud, 8N1, Modbus-RTU
  * **High-Voltage Screw Terminals (PZEM-004T v4.0)**:
    * Terminal 1 (L): 230V AC Live (via 1A fast-blow inline fuse)
    * Terminal 2 (N): 230V AC Neutral
    * Terminal 3 (CT+): 100A Split-Core CT Red wire
    * Terminal 4 (CT-): 100A Split-Core CT Black wire
    * *Safety Note*: Screw terminals 3 & 4 tightly BEFORE clamping the CT around the single insulated Live (L) wire. Note that the PZEM-004T AC side must be powered by 230V AC for the onboard metering ASIC to respond to UART polls.
  * MCU must be **ESP32 DevKit V1 (30-pin)**. **Not WROVER** — WROVER uses GPIO 16/17 for PSRAM.
* **2-Channel Active-LOW Opto-Isolated Relay Module**:
  * **VCC (Opto-isolator LED source)** -> ESP32 GPIO 25 (driven HIGH in setup() as 3.3V source for opto LEDs; feeds only a few mA).
  * **JD-VCC (Relay Coil Supply)** -> Dedicated 5V External Supply / VIN (jumper removed between VCC and JD-VCC).
  * **GND** -> Common DC GND.
  * **IN1** -> GPIO 26 (Relay 1: Laptop Charger)
  * **IN2** -> GPIO 27 (Relay 2: Phone Charger)
  * **Boot strap pin caution**: GPIO 12 (MTDI) is strictly avoided. GPIO 16/17 are not used for anything else.
  * **Logic Polarity**: **ACTIVE-LOW** (`RELAY_ON = LOW`, `RELAY_OFF = HIGH`). Relays are held OFF through boot and reset:
    1. IN1 (GPIO26) and IN2 (GPIO27) are set to OUTPUT and written HIGH (RELAY_OFF).
    2. THEN RELAY_VCC_PIN (GPIO25) is set to OUTPUT and written HIGH.
    3. Delay ~100 ms, then initialize PZEM on Serial2 (GPIO16/17).
  * Safe-state function `allRelaysOff()` writes HIGH to GPIO26/27 on any fatal error, watchdog trip, comms loss, or restart.
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
