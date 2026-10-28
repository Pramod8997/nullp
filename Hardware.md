# Hardware Pinout & Wiring Specification

> **Rig:** Single-node, **single-socket** prototype — laptop + phone charger, ≤250 W, India 230 V 50 Hz.
> **Current board:** 38-pin ESP32-WROOM-32 DevKit. The 38-pin low-voltage guide is [`claude_debug/WIRING_STEP_BY_STEP.md`](./claude_debug/WIRING_STEP_BY_STEP.md); older 30-pin references are not mechanically compatible.
> **Authoritative current build:** [`docs/ems_hardware/ACTUAL_100A_CT_WIRING.md`](./docs/ems_hardware/ACTUAL_100A_CT_WIRING.md). The older 10 A shunt specification is historical and must not be mixed with the available CT meter.
>
> **Build class:** zero soldering, integrated modules only, ≤₹3,000. Production
> pins are already assigned; the repository also provides a separate compiled
> `bench` firmware for mains-disconnected relay continuity testing.

* **PZEM-004T-100A(V4.0)-type external-CT meter:**
  * ESP32 GPIO 16 (RX2)  <──  PZEM TTL TX; **measure TX idle voltage before connecting**. A measured 5 V output requires the documented 1 kΩ/2 kΩ divider or a verified level converter.
  * ESP32 GPIO 17 (TX2)  ──>  PZEM TTL RX
  * PZEM TTL header: `GND, TX, RX, 5V`; AC/CT header: `CT, CT, N, L` per the exact manual.
  * The external CT surrounds one live conductor only; load current does not pass through a PZEM shunt.
  * There is no ESP32 ADC, ZMPT101B, ACS712, SCT-013 or external burden-resistor path.
  * MCU must be **ESP32-WROOM-32**, **38-pin DevKit**. **Not WROVER** — WROVER uses GPIO 16/17 for PSRAM.
* **Relay actuation — direct drive, NO generic level shifter** (verified 5 V opto-isolated board; the available board may have four channels):
  * Use channel 1 only; channels 2–4 remain unused.
  * Module jumper set to **H (high-trigger)**
  * ESP32 GPIO 18 ──────────> Relay Module **IN**   ← direct, no MOSFET, no series resistor
  * Relay **IN** ──[ 100kΩ ]──> GND  ← **mandatory pull-down**: GPIO 18 is Hi-Z during reset/boot, and without this the relay state is undefined. Resistor legs go straight into the screw terminals — no soldering
  * Relay VCC ──> +5 V · Relay GND ──> shared common ground with the ESP32
  * **Net polarity at GPIO 18 is ACTIVE-HIGH** (`RELAY_ACTIVE_LOW = false` in `main.cpp`). With a high-trigger input there are **zero inversions** in the chain: HIGH = closed, LOW = open, Hi-Z = open (fail-safe). GPIO 18 sources ~2.1 mA into the opto LED.
   * ⚠️ **Why not a low-trigger module:** at IN = 3.3 V a 5 V-referenced opto still passes ~0.5 mA — not a guaranteed OFF; the relay may fail to release or chatter. If you can only get a low-trigger board, set module VCC = 3.3 V with JD-VCC on 5 V, flip the `RELAY_ACTIVE_LOW` constant in `main.cpp` to `true`, and move the 100 kΩ to a pull-**up**. Re-verify at Stage 2.
    * **Module identity gate:** the actual board's H/L behavior is not
     approved by its contact-current label alone. Before wiring, verify the
     actual board's H/L jumper, VCC/JD-VCC arrangement, 5 V input current and
     COM/NO contact labeling against its datasheet.
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
