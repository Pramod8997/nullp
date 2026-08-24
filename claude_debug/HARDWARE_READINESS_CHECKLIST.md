# Hardware Readiness & Order Checklist

> **Companion to:** [`HARDWARE_DEPLOYMENT_GUIDE.md`](./HARDWARE_DEPLOYMENT_GUIDE.md)
> **Purpose:** Pre-procurement review, order list, and staged bring-up gate before any mains is applied.

> ✅ **All blocking issues raised here are now RESOLVED. Scope is decided.**
> Order and wire from **[`HARDWARE_FINAL_SPEC.md`](./HARDWARE_FINAL_SPEC.md)** — it is
> authoritative and supersedes the order tables below. Build: **single aggregate node,
> ~600 W consumer electronics, India 230 V.**
>
> Two corrections to this document's own recommendations:
> * **§0.5's "2 A or 3 A" load fuse was wrong** — it anti-coordinates with the 600 W /
>   125 % CRITICAL trip recommended in §7. At 750 W the branch draws 3.26 A (4.17 A in a
>   180 V brownout, since SMPS loads are constant-power), so a 2–3 A fuse blows *before*
>   the relay can ever demonstrate a cutoff. **Use 5 A** — spec §D8.
> * **Two new blocking issues were found** that this checklist did not catch:
>   **B-5** (`firmware/.../README_PHASE2.md` documented an entirely different analog-CT
>   sensor chain on GPIO 34/5) and **B-7** (`RELAY_ACTIVE_LOW = true` inverted the relay
>   once the MOSFET level shifter was added — boot energised the load and safety cutoffs
>   *closed* the relay). Both are fixed; see spec §4 and §D5.


---

## 0.5 DEMO LOADS: the 100 A CT is the wrong sensor — 🔴 **B-0, affects your order**

If the demo plugs in a **laptop, phone charger, and projector** instead of a kettle/HVAC, the BOM's sensor cannot resolve them.

PZEM-004T v3.0 ships in two variants, and the difference is decisive at low power:

| | **100 A CT version** (in the BOM) | **10 A direct-connect version** |
| :--- | :--- | :--- |
| Range | 0–100 A (0–23 kW) | 0–10 A (0–2.3 kW) |
| Starting current | 0.02 A ≈ **4.6 W** | 0.01 A ≈ **2.3 W** |
| Current resolution | 0.01 A ≈ **2.3 W/step** | 0.001 A ≈ **0.23 W/step** |
| Connection | CT clamp around Live | Load current passes through the module |

Against the actual demo loads:

| Demo load | Typical power | Current @ 230 V | On the 100 A CT | On the 10 A version |
| :--- | :--- | :--- | :--- | :--- |
| Phone charger | 3–10 W | 0.013–0.043 A | 🔴 **at/below the 4.6 W start threshold — reads 0 W or flickers** | 🟢 ~20–45 steps, usable |
| Laptop charger | 30–90 W | 0.13–0.39 A | 🟡 13–39 steps, coarse and noisy | 🟢 130–390 steps, clean |
| Computer monitor | 20–40 W | 0.09–0.17 A | 🟡 marginal | 🟢 clean |
| Desktop computer | 80–350 W | 0.35–1.5 A | 🟢 workable | 🟢 clean |
| Projector | 200–400 W | 0.9–1.7 A | 🟢 workable | 🟢 clean |

**Order decision:**

| ✅ | Item | Why |
| :-: | :--- | :--- |
| ☐ | **PZEM-004T v3.0 — 10 A direct-connect (shunt) variant** | **Required for the demo.** 10× better low-end resolution; 2.3 kW ceiling is far above the ~400 W demo total |
| ☐ | PZEM-004T v3.0 — 100 A CT variant | Keep **only** if you also want whole-house aggregate monitoring. Not usable for the phone charger |

**Also note for the demo:**

* A **phone charger will not produce a detectable event.** `TRANSIENT_THRESHOLD_W` is 20 W (`src/pipeline/aggregate_nilm.py:27`), and a 3–10 W charger is below it — correctly so, since that step is inside the sensor's own noise. Do not build the demo script around the phone being identified. Use it to show *phantom / standby load* tracking instead, which is what `PhantomTracker` is for.
* **The 3500 W safety ceiling will never trigger** on ~400 W of demo load (`config/config.yaml`, `system_safety.max_aggregate_wattage`). Use the scaled demo profile so the cutoff is demonstrable — see §7.
* Load-branch protection (§0 B-1) should be sized to the demo load, not 16 A: a **2 A or 3 A fuse** is correct for ~400 W and will actually demonstrate protection.

---

## 0. BLOCKING ISSUES — fix the schematic first

### 🔴 B-1: The 0.5 A fuse is in series with the load branch

`HARDWARE_DEPLOYMENT_GUIDE.md` §1 shows:

```
[ 230V AC Mains In (L) ] ──[ 0.5A Fuse ]──┬──[ HLK-5M05 AC-L ]
                                          ├──[ 10D471K MOV ]──[ Mains (N) ]
                                          └──[ Relay COM ] ──> [ Appliance Load ]
```

Every appliance the project targets draws far more than 0.5 A:

| Target load | Current @ 230 V | Result on a 0.5 A fuse |
| :--- | :--- | :--- |
| Kettle (2.2 kW) | 9.6 A | blows instantly |
| Oven (3 kW) | 13.0 A | blows instantly |
| EV charger (3.3 kW) | 14.3 A | blows instantly |

The 0.5 A fuse is correctly sized for the HLK-5M05 (5 W ⇒ ~25 mA) but must **not** feed the relay. Split the branches:

```
[ 230V AC Mains In (L) ] ──┬──[ 0.5A Slow-Blow ]──[ HLK-5M05 AC-L ]
                           ├──[ 10D471K MOV ]────[ Mains (N) ]
                           └──[ 16A MCB / 20A fuse ]──[ Relay COM ]──[ Relay NO ]──> [ Load ]
```

Size the load-branch device to the appliance, never above the relay contact rating or the wire ampacity.

### 🔴 B-2: No protective earth (PE) anywhere in the design

The schematic routes only Live and Neutral. There is no PE conductor, no earth terminal, and no earth continuity to the load. Any Class I appliance (metal-cased kettle, oven, washing machine — most of the target list) requires an uninterrupted earth path, and an earthed enclosure is what makes an insulation fault trip the RCD instead of energising the case.

**Required:** add a PE terminal to the DIN enclosure, bond it to incoming mains earth, and carry earth straight through to the load socket. **Never switch PE through the relay.**

### 🟡 B-3: `Hardware.md` (repo root) contradicts the deployment guide

`Hardware.md` still documents the pre-audit wiring and will reproduce two already-solved hazards if followed:

| `Hardware.md` says | Should be | Hazard if followed |
| :--- | :--- | :--- |
| `HLK-PM01 5V` | **HLK-5M05** (5 V 1 A) | Hazard 3 — brownout bootloop |
| `GPIO 18 ──> Relay IN` (direct) | GPIO 18 → 1 kΩ → 2N7000 → Relay IN | Hazard 1 — relay never turns off / chatters |

Treat `HARDWARE_DEPLOYMENT_GUIDE.md` as authoritative and update `Hardware.md` to match.

### 🟡 B-4: Verify PZEM-004T TX logic level before connecting GPIO 16

ESP32 GPIO absolute maximum is 3.6 V. PZEM-004T v3.0 boards ship in two variants: an opto-isolated open-collector TX (safe — pull up to 3.3 V) and a 5 V push-pull TTL TX (**will exceed the ESP32 limit**).

**Bench check before wiring:** power the PZEM from 5 V with TX unconnected, and measure TX-to-GND idle voltage.
* ≈3.3 V or floating → connect directly (add a 10 kΩ pull-up to 3.3 V if floating).
* ≈5 V → insert a divider (1 kΩ series / 2 kΩ to GND) or a level shifter on the PZEM-TX → GPIO-16 line.

---

## 1. Order Checklist — Core BOM (from the deployment guide)

Quantities include a spare where the part is destroyed by a wiring error or is consumed during the 8 bench tests.

| ✅ | Item | Exact Part | Order Qty | Notes |
| :-: | :--- | :--- | :-: | :--- |
| ☐ | ESP32 DevKit V1 | ESP32-WROOM-32D | **2** | 1 spare; confirm 30-pin variant matches your breadboard/PCB footprint |
| ☐ | AC energy meter | PZEM-004T **v3.0** | **1** | v3.0 only — v1/v2 use a different, incompatible register map. Usually **ships with its own matched CT** |
| ☐ | Split-core CT | 100 A / 50 mA matched | **1** | ⚠️ **Check before ordering** — likely already included with the PZEM v3.0 100 A kit. Do not double-order |
| ☐ | 30 A relay module | SLA-05VDC-SL-C, 5 V coil, opto-isolated | **1** | Confirm the module is opto-isolated and note whether it is active-LOW |
| ☐ | AC-DC PSU | Hi-Link **HLK-5M05** (5 V 1 A) | **1** | **Not** HLK-PM01 — see Hazard 3 |
| ☐ | Bulk cap | 1000 µF 16 V **low-ESR** electrolytic | **2** | Low-ESR is required, not optional |
| ☐ | HF decoupling cap | 100 nF 50 V ceramic X7R | **5** | Cheap in strips |
| ☐ | Level-shift MOSFET | 2N7000 N-channel (TO-92) | **5** | BSS138 (SOT-23) is an acceptable substitute |
| ☐ | Gate resistor | 1 kΩ 0.25 W metal film | **5** | |
| ☐ | Snubber resistor | 100 Ω 2 W flameproof metal oxide | **2** | Flameproof is mandatory |
| ☐ | Snubber cap | 0.1 µF 275 VAC **Class X2** polypropylene | **2** | X2 safety-rated only — never a general-purpose film cap |
| ☐ | PSU-branch fuse | 0.5 A 250 V slow-blow | **5** | For the HLK-5M05 branch **only** (see B-1) |
| ☐ | MOV | 10D471K (470 V clamp) | **2** | Line-to-neutral, upstream of the PSU |
| ☐ | Enclosure | DIN rail ABS, UL94-V0 flame retardant | **1** | Must close fully with mains inside |

## 2. Order Checklist — Missing from the BOM but required to assemble

None of the following appear in the guide's BOM, and assembly cannot complete without them.

| ✅ | Item | Spec | Qty | Why it is required |
| :-: | :--- | :--- | :-: | :--- |
| ☐ | **Load-branch protection** | 16 A MCB or 20 A ceramic fuse + holder | 1 | **Resolves B-1.** Sized to the appliance |
| ☐ | **PE / earth terminal block** | DIN earth block, green/yellow | 1 | **Resolves B-2** |
| ☐ | **Earth wire** | 2.5 mm² green/yellow stranded | 2 m | **Resolves B-2** |
| ☐ | Fuse holder | Panel or DIN, 250 V, for the 0.5 A glass fuse | 1 | The BOM lists a fuse with nothing to hold it |
| ☐ | Mains wire | 2.5 mm² (14 AWG) stranded, 300/500 V — brown / blue | 3 m ea | 2.5 mm² carries 20 A safely; thinner overheats |
| ☐ | DC wire | 0.5 mm² (22 AWG) stranded, multiple colours | 2 m | ESP32 / PZEM / relay low-voltage runs |
| ☐ | Ferrules + crimper | 2.5 mm² and 0.5 mm² bootlace ferrules | 1 kit | Bare stranded wire in screw terminals loosens and arcs |
| ☐ | Mains terminal blocks | 3-way 30 A barrier or DIN, screw type | 2 | L / N / PE distribution |
| ☐ | DIN rail | TS35 steel, cut to enclosure | 1 | The enclosure is DIN-mount; the rail is separate |
| ☐ | Cable glands | M16 or M20 with locknuts | 3 | Strain relief at every enclosure entry |
| ☐ | Heat-shrink | Assorted 2–10 mm | 1 pack | Insulate every mains joint |
| ☐ | High-voltage sleeving | Fibreglass or silicone, 600 V | 1 m | Reinforce mains runs inside the enclosure |
| ☐ | Warning label | "⚠ 230 VAC INSIDE — ISOLATE BEFORE OPENING" | 1 | |

## 3. Order Checklist — Bench safety and instrumentation

Do not energise anything before these are on hand.

| ✅ | Item | Why |
| :-: | :--- | :--- |
| ☐ | **RCD / GFCI portable adapter (30 mA)** | Non-negotiable. First line of protection while a mains circuit is open on the bench |
| ☐ | **Isolation transformer (1:1, ≥500 VA)** | Breaks the earth reference so an accidental touch is not a path to ground |
| ☐ | True-RMS multimeter, CAT III 600 V | Sanity-check PZEM readings; verify B-4 logic level |
| ☐ | Clamp meter (AC, ≥30 A) | Independent current reference for CT calibration |
| ☐ | Oscilloscope, ≥20 MHz (isolated / differential probe for mains) | Tests 1, 3, 4, 7 in `REAL_WORLD_TESTING_PLAN.md` |
| ☐ | Variac (0–260 V) | Test 2 — sag / swell transients |
| ☐ | IR thermometer or thermal camera | Test 5 — 30 A thermal stress |
| ☐ | Insulated screwdrivers (VDE 1000 V) | Working near live terminals |
| ☐ | Test loads: 2 kW kettle (resistive) + vacuum or drill (inductive) | Tests 3, 4 need real inrush and real back-EMF |

---

## 4. Pre-Power-On Inspection Gate

Complete **all** items before mains is connected for the first time.

**Wiring integrity**
* ☐ Continuity: mains L → PZEM L → relay COM, with the load-branch MCB in circuit
* ☐ Continuity: incoming PE → enclosure earth block → load socket earth (**unswitched**)
* ☐ Isolation ≥10 MΩ: L/N to the 5 V DC rail, and L/N to enclosure earth
* ☐ Creepage ≥6.3 mm everywhere between mains and 3.3 V/5 V nets (IEC 62368) — inspect under magnification
* ☐ Every conductor ferruled and every screw terminal torqued; tug-test each one
* ☐ 0.5 A fuse feeds **only** the HLK-5M05 branch (B-1 resolved)
* ☐ CT clamp closed around the **Live conductor only**, arrow pointing toward the load

**Polarity and orientation**
* ☐ 1000 µF electrolytic polarity correct (reversed = venting)
* ☐ 2N7000 pinout verified: GPIO 18 → 1 kΩ → Gate, Drain → relay IN, Source → GND
* ☐ PZEM UART crossed: ESP32 GPIO 16 (RX2) ← PZEM TX, GPIO 17 (TX2) → PZEM RX
* ☐ RC snubber across relay **COM and NO** (not across the coil)

**Firmware and configuration**
* ☐ Firmware flashed and boots clean on USB power alone
* ☐ Relay defaults to **OFF** on cold boot and stays off through a WiFi connect cycle
* ☐ `config/` MQTT broker address, credentials, and device ID set for the real deployment
* ☐ Aggregate ceiling and per-device limits reviewed for the actual circuit rating

---

## 5. Staged Bring-Up Sequence

Never jump stages. Each stage has an abort condition.

| Stage | Connect | Verify | Abort if |
| :-: | :--- | :--- | :--- |
| **1** | USB only, no mains, relay unplugged | ESP32 boots, WiFi joins, MQTT connects, relay pin idles OFF | Bootloop or relay drives active |
| **2** | USB + relay module (no mains on contacts) | `set_relay(True/False)` audibly clicks and follows commands exactly | Relay chatters or inverts — revisit Hazard 1 / 2N7000 |
| **3** | HLK-5M05 on mains via RCD + isolation transformer; **USB removed**; relay contacts still open | 5 V rail holds ≥4.75 V through a WiFi TX burst (Test 1) | Rail dips below 4.4 V — check the 1000 µF low-ESR cap |
| **4** | PZEM sense (L/N + CT), load branch still open | Voltage reads ≈230 V; current ≈0 A; verify B-4 logic level first | No UART response, or TX measures 5 V |
| **5** | Resistive load (kettle) through relay + load MCB | PZEM power within ±2 % of clamp meter; overcurrent trips the relay locally with MQTT unplugged | Cutoff needs the network to work |
| **6** | Inductive load (vacuum/drill) | Snubber damps arcing; cold inrush does **not** trigger a false arc-fault (Test 4) | Contacts arc visibly or weld |
| **7** | 30 A continuous soak, enclosure closed, 1 h | No conductor or terminal exceeds +30 °C rise (Test 5) | Any hotspot, discolouration, or smell |

---

## 6. Verify Before Trusting the Software

The edge safety path is intentionally independent of the ML stack, and that must be confirmed on real hardware, not just in simulation.

* ☐ **Relay cutoff works with the network down.** Pull the WiFi/broker and confirm overcurrent still opens the relay from Core 0 alone.
* ☐ **Relay cutoff works with ML disabled.** Delete `backend/models/weights/protonet.pt`, restart the pipeline, and confirm safety still trips while classification degrades to the heuristic fallback (see `src/pipeline/heuristic_fallback.py`).
* ☐ **CT calibration against a known load.** Run `scripts/calibrate_ct.py` with the clamp meter as reference; PZEM error should sit within ±2 %.
* ☐ **Appliance identification is advisory, not actionable.** Per-appliance attribution comes from a model validated on unseen houses at moderate accuracy (see `training_results/training_report.json`). Do not wire it to billing or to unattended shedding of a critical load.
* ☐ **dP/dt is not an arc-fault detector.** The PZEM register refresh quantises rate-of-change to a ~500 ms window (Hazard 4). It is a thermal-runaway and overcurrent proxy only; a real AFDD is a separate certified device.

---

## 7. Demo Configuration & What to Claim

Run the demo with the scaled profile, not the default config:

```bash
python scripts/run_pipeline.py --config config/config.demo.yaml
```

`config/config.demo.yaml` drops the aggregate ceiling from 3500 W to **600 W**, so a laptop + projector + monitor (~530 W) enters the warning band and adding the desktop trips CRITICAL. With the default 3500 W ceiling the safety path can never fire at bench power levels and will simply look broken.

### Train the demo-profile model

The general 10-class model spans 3 W – 3.5 kW and is weakest in exactly the band your demo occupies. Train the consumer-electronics model instead:

```bash
python scripts/train_demo_models.py          # → backend/models/weights_demo/
```

Classes: `laptop`, `desktop_computer`, `monitor`, `projector`, `tv`, `router`, `phone_charger` — all extracted from real UK-DALE meters.

### Claims to make and claims to avoid

| ✅ Safe to demonstrate | ❌ Do not claim |
| :--- | :--- |
| Live power/voltage/current from a real appliance | "Identifies any appliance automatically" |
| Local relay cutoff with the network unplugged | Per-appliance billing accuracy |
| Threshold + rate-of-change safety events | Arc-fault detection |
| Appliance identification as **advisory**, with confidence shown | Unattended shedding of a critical load |
| Graceful degradation when ML is removed | A specific headline accuracy number without naming the protocol |

**On accuracy:** quote the unseen-house figure from `training_results/training_report.json`, not the episodic number. If asked why it is moderate, the honest answer is that it is measured on houses the model never trained on — a random split would score much higher but would not predict field behaviour.

**`projector` caveat:** UK-DALE contains exactly one projector meter in one house, so that class cannot be validated on an unseen house. Its score is optimistic; say so if asked.

**Phone chargers & Powerbanks (5–120 W):**
- Trickle and standby loads (3–10 W) sit below the 20 W transient threshold (`TRANSIENT_THRESHOLD_W = 20.0 W`) and are tracked continuously as **standby/phantom load** via `PhantomTracker`.
- Fast chargers, USB-PD power supplies, and high-capacity powerbanks (18–120 W) cross the 20 W step-change threshold, triggering active `NILMTransientDetector` event detection and classification via `ProtoNet` and `HeuristicApplianceClassifier`.

