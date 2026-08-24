# Hardware Pinout & Wiring Specification

> **Rig:** Single-node, **single-socket** prototype — laptop + phone charger, ≤250 W, India 230 V 50 Hz.
> **Authoritative spec (BOM, rationale, bring-up):** [`claude_debug/HARDWARE_FINAL_SPEC.md`](./claude_debug/HARDWARE_FINAL_SPEC.md)
> This file is the pinout quick reference only. Where it disagrees with the spec, the spec wins.
>
> **Build class:** zero soldering, integrated modules only, ≤₹3,000. **No firmware change is required** — `main.cpp:59-69` is already correct for this wiring.

* **PZEM-004T v3.0 Metering UART** — **10 A direct-connect (shunt) variant, not the 100 A CT variant:**
  * ESP32 GPIO 16 (RX2)  <──  PZEM-004T TX  ⚠️ **measure TX idle voltage before connecting** — 5 V push-pull variants exceed the ESP32's 3.6 V absolute maximum. ≈3.3 V or floating → direct (10 kΩ pull-up to 3.3 V if floating); ≈5 V → 1 kΩ/2 kΩ divider
  * ESP32 GPIO 17 (TX2)  ──>  PZEM-004T RX
  * Load current passes **through** the module's screw terminals. There is no CT clamp and no ADC input.
  * MCU must be **ESP32-WROOM-32D**, **30-pin**. **Not WROVER** — WROVER uses GPIO 16/17 for PSRAM.
* **Relay actuation — direct drive, NO level shifter** (SRD-05VDC-SL-C, opto-isolated, 5 V coil, **2-channel with H/L jumper**):
  * Module jumper set to **H (high-trigger)**
  * ESP32 GPIO 18 ──────────> Relay Module **IN**   ← direct, no MOSFET, no series resistor
  * Relay **IN** ──[ 100kΩ ]──> GND  ← **mandatory pull-down**: GPIO 18 is Hi-Z during reset/boot, and without this the relay state is undefined. Resistor legs go straight into the screw terminals — no soldering
  * Relay VCC ──> +5 V · Relay GND ──> shared common ground with the ESP32
  * **Net polarity at GPIO 18 is ACTIVE-HIGH** (`RELAY_ACTIVE_LOW = false`, `main.cpp:67`). With a high-trigger input there are **zero inversions** in the chain: HIGH = closed, LOW = open, Hi-Z = open (fail-safe). GPIO 18 sources ~2.1 mA into the opto LED.
  * ⚠️ **Why not a low-trigger module:** at IN = 3.3 V a 5 V-referenced opto still passes ~0.5 mA — not a guaranteed OFF; the relay may fail to release or chatter. If you can only get a low-trigger board, set module VCC = 3.3 V with JD-VCC on 5 V, flip `main.cpp:67` to `true`, and move the 100 kΩ to a pull-**up**. Re-verify at Stage 2.
  * RC snubber (100 Ω 2 W flameproof + 0.1 µF 275 VAC Class-X2 across COM–NO): **optional** for this load set — no inductive load remains in scope. Reinstate for any motor/fan/pump/transformer load.
* **Wiring interface:**
  * ESP32 plugs into a **30-pin screw-terminal expansion shield** — match pin count *and* board width. Pins are only broken out; no remapping.
* **Power Supply (5 V Rail):**
  * **BIS-marked 5 V 2 A USB charger** ──> USB-A-female-to-screw-terminal breakout ──> shield VIN / GND
  * Charger lives **outside** the enclosure on its own wall socket. This deletes the HLK-5M05, the MOV, the 0.5 A PSU fuse, the 1000 µF bulk cap, the 100 nF cap and the entire second mains tap.
  * ESP32 VIN (5 V), Relay VCC (+5 V), PZEM VCC (+5 V) from adjacent terminals
  * ⚠️ **Laptop USB and mains are mutually exclusive.** Flash on laptop USB with mains disconnected; run on the 5 V charger with the laptop cable removed. The 2-pin charger has no earth reference and is the intended run-time supply.
* **AC Mains & Protective Earth (PE):**
  * Order is fixed: **fuse ──> PZEM ──> relay ──> socket.** The PZEM must sit upstream of the relay so a trip reads 230 V / 0 A / 0 W (load proven dead) instead of 0 V / 0 A (indistinguishable from a dead sensor).
  * Mains Live (L) ──> 5 A ceramic fuse ──> PZEM in-L ──> PZEM out-L ──> Relay COM ──[ Relay NO ]──> Socket Live
  * Mains Neutral (N) ──> PZEM N & Socket Neutral
  * Protective Earth (PE) ──> PE barrier block (enclosure-bonded) ──> Socket PE (**unswitched**)
  * Load socket: **one** IS 1293 6 A 3-pin panel mount, with an **earthed 3-pin multi-plug adapter** in it
  * Upstream: **30 mA RCBO** (portable adapter, or a verified house-DB RCCB)
* **The two loads share one socket, simultaneously** — laptop brick + phone charger through the same PZEM shunt. That overlap is what NILM disaggregates; one device at a time is single-appliance classification and never exercises `OverlapAwareNILMDetector`.

**Coordination ladder** — must stay monotonic, see spec §D8:
`185 W nominal (0.80 A) → WARNING 275 W → relay trips 312 W (1.36 A) → 5 A fuse → 10 A relay/PZEM rating → 13 A wire`

**Run the pipeline with the hardware config, not the demo config:**
```bash
python scripts/run_pipeline.py --config config/config.hardware.yaml   # 250 W ceiling, real rig
```
`config/config.demo.yaml` keeps a 600 W ceiling for its ~1130 W *simulated* fleet. Using it on the real rig puts CRITICAL at 750 W, which a 250 W prototype can never reach — the cutoff would never fire.
