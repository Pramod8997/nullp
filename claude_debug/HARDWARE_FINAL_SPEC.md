# Physical Hardware — FINAL LOCKED SPECIFICATION

> **Status:** ✅ **DECIDED — this document is authoritative.**
> Supersedes the BOM and schematic in [`HARDWARE_DEPLOYMENT_GUIDE.md`](./HARDWARE_DEPLOYMENT_GUIDE.md)
> and the order list in [`HARDWARE_READINESS_CHECKLIST.md`](./HARDWARE_READINESS_CHECKLIST.md)
> wherever they disagree.
>
> **Scope:** Single-node, **single-socket** consumer-electronics prototype. **Laptop + phone charger only, ≤250 W**, India 230 V 50 Hz.
> **Explicitly NOT rated for:** kettle, oven, HVAC, washing machine, EV charger, projector, desktop PC.
> **Build class:** ≤₹3,000 rig budget, **integrated modules only, zero soldering, zero firmware change.**

---

## 0. The five scope decisions everything else follows from

| | Decision | Consequence |
| :-- | :--- | :--- |
| **S1** | **Prototype only, ~250 W envelope** | 10 A sensing, 10 A switching, 5 A fusing, 1.0 mm² wire. The 3.5 kW household profile in `config/config.yaml` and the 600 W fleet in `config/config.demo.yaml` are **software-only** and have no hardware. |
| **S2** | **One aggregate sense point (true NILM)** | 1× ESP32 + 1× PZEM. NILM must disaggregate — which is the entire point of the ML stack. No per-appliance sub-meters. |
| **S3** | **India, 230 V 50 Hz, IS 1293** | 6 A 3-pin load socket, 30 mA RCBO, PE mandatory on every Class I load. |
| **S4** | **Builder has no hardware background → integrated modules, no solder, ≤₹3,000** | Every discrete/SMD part is replaced by a pre-assembled module or a screw-terminal-mounted resistor. No SMD, no PCB-pin modules, no hand-soldered mains. See D3′/D6′ and §2. |
| **S5** | 🆕 **ONE switched socket. Laptop + phone charger, plugged in *simultaneously*** | A **3-pin multi-plug adapter** sits in that socket so both loads share the PZEM shunt. This is load-bearing, not convenience — see D12′. Ceiling drops 600 W → **250 W**, so the cutoff is physically reachable. |

### S5 — why one socket must still carry two loads at once

> 🔴 **One device at a time is not disaggregation.** With a single load energised, the aggregate signal *is* that appliance — that is single-appliance classification. `OverlapAwareNILMDetector` (power subtraction) would never execute on real data, and the central ML claim of the project would be demonstrated only in simulation.
>
> A ₹50 multi-plug adapter restores the real behaviour: laptop and charger draw through **one** PZEM at **one** sense point, and the pipeline must separate them. One port, one meter, two overlapping loads — S2 fully intact.

**Demonstration sequence** (each step is a real NILM event on one meter):
1. Laptop alone → baseline + `laptop` classification.
2. Phone charger plugged in **while the laptop runs** → step edge on top of an existing load → overlap-aware detection → `phone_charger`.
3. Laptop unplugged, charger left → negative edge, charger persists.
4. Charger idle at 3–10 W → below `TRANSIENT_THRESHOLD_W = 20.0` → **`PhantomTracker`**, not a classification event (see §5).

### Physical topology

```
                      ┌──────────── EMS NODE (enclosed) ─────────────┐
                      │                                              │
[ 6A plug ]──L────────┼──[ 5A ceramic ]──[ PZEM in-L ]               │
   to wall            │                       │                      │
   socket             │                  [ PZEM shunt ]              │
   (on 30mA           │                       │                      │
    RCBO)             │                  [ PZEM out-L ]              │
                      │                       │                      │
                      │                  [ Relay COM ]               │
                      │                       │                      │
                      │                  [ Relay NO ]────────────────┼──> L
           ───N───────┼───────────────────────┴──────────────────────┼──> N   [IS 1293
                      │            (also PZEM voltage-sense N)        │        6A socket]
           ───PE──────┼──[ Earth block ]════ UNSWITCHED ═════════════┼──> PE     │
                      │         ║                                     │          │
                      │    (enclosure bond)                           │          ▼
                      │                                               │
                      │   5 V ▲ GND     ← 2-wire DC only, via gland   │   [ ONE socket ]
                      └───────┼───────────────────────────────────────┘ [ 3-pin multi-plug ]
                              │                                          │            │
                   [ USB screw-terminal breakout ]                    laptop      phone
                              │                                        brick     charger
                   [ 5 V 2 A BIS USB charger ]  ← OUTSIDE the enclosure,
                              │                    own wall socket      ↑ BOTH AT ONCE — S5
                          [ mains ]
```

**Load-path order is fixed: fuse → PZEM → relay → socket.** See D7 for why the relay must be downstream.

**Only ONE mains net enters the enclosure now** (the load branch). The PSU tap, its 0.5 A fuse and the MOV all left the box with D6′ — that is the single biggest reduction in hand-built mains work, and it is why this build is appropriate for a first-time builder.

---

## 1. Locked component decisions

### D1 — Sensor: PZEM-004T v3.0, **10 A direct-connect (shunt) variant**

| | 10 A direct (**CHOSEN**) | 100 A CT (rejected) |
| :--- | :--- | :--- |
| Range | 0–10 A / 0–2.3 kW | 0–100 A / 0–23 kW |
| Start current | 0.01 A ≈ **2.3 W** | 0.02 A ≈ 4.6 W |
| Resolution | 0.001 A ≈ **0.23 W/step** | 0.01 A ≈ 2.3 W/step |

Rejected because the 100 A CT's 4.6 W start floor sits *above* a 3–10 W phone charger (reads 0 W or flickers) and its 2.3 W/step quantisation makes a 60 W monitor a 26-step signal — too coarse for NILM feature extraction. The 2.3 kW ceiling of the 10 A part is 3.8× the 600 W envelope.

**Consequences:** no CT clamp, no SCT-013, no burden resistor, no ADC input. Load current physically passes through the module's screw terminals.

### D2 — Switch: **2-channel 5 V opto-isolated relay module with H/L trigger jumper** (SRD-05VDC-SL-C, 10 A / 250 VAC)

Replaces the SLA-05VDC-SL-C 30 A. The 30 A part is 3.8× oversized on contacts and its 180 mA coil is the single largest draw on the 5 V rail; the SRD coil is ~71 mA. Contact margin at the 2.6 A worst-case demo load is 3.8×.

Module must be **opto-isolated**. Buy the **2-channel** variant specifically: 1-channel boards are overwhelmingly fixed low-trigger, whereas 2-channel boards almost always carry the **H/L selector jumper** that D3′ depends on. Channel 1 is the relay; channel 2 is a free spare. ~₹120 either way.

### D3′ — 🔒 Level shifter **DELETED**. GPIO 18 drives a **high-trigger** opto input directly.

> **Supersedes D3 (BSS138 + 1 kΩ + 100 kΩ).** The BSS138 is **SOT-23 surface-mount** — unsolderable by hand for a first-time builder, and the reason S4 forced this revision.

Set the module jumper to **H (high-trigger)**. GPIO 18 then *sources* the optocoupler LED current instead of sinking it:

```
GPIO 18 ──> Relay IN   (module-internal: IN → opto LED → ~1 kΩ → GND)
GPIO 18 ──[ 100 kΩ ]──> GND      ← still mandatory, see below
```

Drive current ≈ (3.3 V − 1.2 V) / 1 kΩ ≈ **2.1 mA**, far inside the ESP32's 40 mA per-pin limit.

**Why a *low*-trigger module cannot be driven directly** — and why the MOSFET existed at all. On a low-trigger board the LED is fed from the module's own 5 V rail and the GPIO must *pull it down*. Driving IN to 3.3 V leaves 5 − 1.2 − 3.3 = 0.5 V across the 1 kΩ series resistor ≈ **0.5 mA still flowing through the LED**. That is not a guaranteed OFF: the relay may fail to release, or chatter. 3.3 V is simply not a valid logic HIGH for a 5 V-referenced opto input. High-trigger inverts the problem away — OFF is a true 0 V, hard off.

> 🔴 **The 100 kΩ IN→GND pull-down is still mandatory** (it was D3's real safety contribution, and it survives unchanged).
> During ESP32 reset and the ~250 ms bootloader window GPIO 18 is high-impedance. Floating → relay state undefined. The pull-down forces IN to 0 V → **OPEN is the only possible boot state**.
> A through-hole resistor's legs push **directly into the screw terminals** — no soldering. Land one leg in the relay `IN` terminal alongside the GPIO 18 wire, the other in `GND`.

### D4 — Relay drive truth table (**high-trigger module, direct drive**)

| GPIO 18 | Relay IN | Opto LED | Relay | Load |
| :--- | :--- | :--- | :--- | :--- |
| LOW | 0 V | off | **OPEN** | dead |
| HIGH (3.3 V) | 3.3 V | ~2.1 mA | **CLOSED** | live |
| Hi-Z (reset/boot) | 0 V via 100 kΩ | off | **OPEN** | dead ✅ fail-safe |

### D5 — 🔒 Firmware polarity **unchanged: `RELAY_ACTIVE_LOW = false`** — and this is now the *direct* reading

Net polarity at GPIO 18 is **active-HIGH**, exactly as before, so `firmware/esp32_node/src/main.cpp:67` is already correct and **no firmware edit is required for this build.**

What changed is *why*. Previously it was active-HIGH because two inversions (inverting MOSFET + active-LOW module input) cancelled. Now there are **zero** inversions: a high-trigger input driven directly is active-HIGH on its face. Same constant, and now it means what it says.

The original defect B-7 is still worth reading: `RELAY_ACTIVE_LOW = true` meant `setRelay(false)` drove GPIO 18 HIGH, so `setup()` at `main.cpp:245` **energised the load at boot** and every overcurrent / dP-dt cutoff in `SafetySamplingTask` **closed** the relay instead of opening it. Fixed; do not reintroduce it.

> ⚠️ **Purchase contingency — the only "it depends" in this BOM.** If the module you receive has no H/L jumper, or refuses to latch at 3.3 V in Stage 2:
> 1. Set module **VCC = 3.3 V** (not 5 V) and remove the **JD-VCC jumper**, feeding JD-VCC from 5 V — the coil stays on 5 V, the opto input becomes 3.3 V-referenced, and 0 V is then a true OFF.
> 2. Flip `main.cpp:67` to `RELAY_ACTIVE_LOW = true`.
> 3. Move the 100 kΩ pull-**down** to a 100 kΩ pull-**up** (IN → 3.3 V), so Hi-Z at boot still means OPEN.
>
> **Re-run Stage 2 and confirm OPEN-at-boot before any mains touches the contacts.** One constant, one resistor move — nothing you purchased becomes wrong.

### D6′ — PSU: **BIS-marked 5 V 2 A USB charger + USB screw-terminal breakout**, mounted **outside** the enclosure

> **Supersedes D6 (HLK-5M05).** The HLK-5M05 is a bare PCB module with **solder pins and exposed AC pads** — it requires soldering, and it puts unguarded mains inside a first-time builder's enclosure. S4 rules it out.

| Rail consumer | Peak |
| :--- | :--- |
| ESP32 WiFi TX burst | 500 mA |
| SRD relay coil | 71 mA |
| PZEM-004T v3.0 | 20 mA |
| **Total** | **≈591 mA** |

A 5 V **2 A** charger leaves **238 % margin** (HLK-PM01's 600 mA left 1.5 %, HLK-5M05's 1 A left 41 %). Feed one 5 V/GND pair through a gland into the shield's `VIN`/`GND` screw terminals and distribute to the relay `VCC` and PZEM `5V` from adjacent terminals.

**This deletes five line items**, because a certified sealed charger already contains all of them: the **10 D471K MOV**, the **0.5 A slow-blow PSU fuse**, the **1000 µF low-ESR bulk cap**, the **100 nF X7R**, and the entire **second mains tap** (B-1's root cause). It is cheaper, solderless, and has reinforced mains isolation certified by a third party rather than by your soldering.

Buy **BIS-marked** (IS 13252). An uncertified ₹60 charger has neither the isolation barrier nor the Y-cap this position assumes.

**D13 still applies, and is unaffected:** its hazard is a *host laptop* USB cable tying ESP32 GND to an earthed chassis. A 2-pin isolated charger has no earth reference, so powering from it while mains-live is exactly the intended configuration.

### D6″ — 🆕 Wiring interface: **ESP32 30-pin screw-terminal expansion shield**

The DevKit plugs in; every GPIO lands on a labelled screw terminal. Its value is **strain relief and non-slip terminations** for the PZEM UART, relay IN and 5 V feed — dupont jumpers on bare header pins work loose, and a GPIO 16 wire that falls off mid-demo reads identically to a dead PZEM.

⚠️ **Match the variant at purchase:** **30-pin**, and the same board *width* as your DevKit. 36/38-pin shields and wide-body shields will not seat. GPIO 16/17/18 are all broken out — no pin remap, so `PZEM_RX_PIN`/`PZEM_TX_PIN`/`RELAY_PIN` at `main.cpp:59-69` are unchanged.

### D7 — Load-path order: fuse → **PZEM** → **relay** → socket

The PZEM's voltage sense and current shunt share one terminal block, so its position decides what a trip looks like:

| Order | On relay trip | Verdict |
| :--- | :--- | :--- |
| **PZEM upstream of relay (CHOSEN)** | 230 V, 0.000 A, 0 W | Proves the load is dead ✅ |
| PZEM downstream of relay | 0 V, 0 A, 0 W | Indistinguishable from a dead sensor ❌ |

### D8 — Protection: **5 A ceramic** on the load branch. Single mains tap.

The original guide put a single 0.5 A fuse in series with *both* branches (blows on any real load — B-1); the second tap is now gone entirely with D6′, so only the load fuse remains. The readiness checklist's **2–3 A** proposal is still rejected — it anti-coordinates with the CRITICAL trip:

```
CRITICAL trip = 1.25 × 250 W = 312 W  →  1.36 A @ 230 V
                                      →  1.74 A @ 180 V brownout
```

The 5 A ceramic is retained even though the trip fell from 3.26 A to 1.36 A. **Do not "right-size" it down to 2 A.** A 2 A fuse sits only 1.15× above the brownout trip current, inside the tolerance band of both the fuse and a USB-PD charger's inrush — it would blow on a cold-start surge *before* the relay ever demonstrates a cutoff, which is the demo failing in the least explicable way possible. 5 A keeps the ladder monotonic with 3.7× margin:

| Stage | Power | Current @ 230 V | Element |
| :-- | :--- | :--- | :--- |
| Nominal prototype load (laptop + charger) | ~185 W | 0.80 A | — |
| WARNING (110 %) | 275 W | 1.20 A | pipeline event |
| **CRITICAL (125 %) → relay opens** | 312 W | **1.36 A** | **functional protection** |
| Load fuse | ~1150 W | 5.00 A | wiring backup only |
| PZEM shunt / relay contacts | 2300 W | 10.0 A | component rating |
| 1.0 mm² conductor | ~3000 W | ~13 A | ampacity |

**The relay is the functional protective element. The fuse protects wiring. The 30 mA RCBO protects people.** Do not conflate them.

> **Reaching the trip:** laptop (65 W) + USB-PD charger (120 W) = 185 W is *below* the 312 W trip, so the cutoff will not fire in normal use — correct, but it means you must force it. Add the **100 W incandescent lamp** to the multi-plug: 285 W crosses WARNING, and a laptop under load (charging + CPU burn) pushes past 312 W. This is the intended trip demonstration, and it is the second reason the lamp is in the BOM.

> **SMPS constant-power caveat:** every prototype load is a switch-mode supply, so as mains sags the current *rises* (250 W is 1.09 A at 230 V but 1.39 A at 180 V). This inverts the intuition from resistive loads and is why the fuse is sized off the brownout case, not the nominal one.

### D9 — Protective earth: bonded, carried through, **never switched**

A **3-way barrier screw block reserved for PE**, bonded to the enclosure and to incoming PE; PE runs straight to the load socket and on through the multi-plug adapter. Resolves B-2. Non-negotiable: laptop bricks are Class I with an earthed chassis, and the earthed enclosure is what makes an insulation fault trip the RCBO instead of energising a case. **The multi-plug adapter must be an earthed 3-pin type** (D12′) — a 2-pin adapter silently breaks this chain at the last inch.

### D10 — MCU: **ESP32-WROOM-32D DevKit V1 only — do NOT substitute ESP32-WROVER**

WROVER modules use **GPIO 16/17 for PSRAM**, which are exactly the PZEM UART pins. A WROVER substitution silently kills metering.

GPIO 18 is confirmed correct for the relay: it is **not** an ESP32 strapping pin (those are GPIO 0, 2, 4, 5, 12, 15), so it has no boot-time pull fighting the drive. This is a further reason the GPIO 5 suggestion in `firmware/esp32_node/src/README_PHASE2.md` was wrong (see B-5).

### D11′ — RC snubber **moved to optional** for this load set

The prototype loads are now *only* SMPS (laptop brick, phone charger) plus a purely resistive incandescent lamp. With the projector and its AC fan deleted by S1, **there is no inductive load left in scope**, and back-EMF contact arcing has nothing to generate it.

> ⚠️ **Reinstate 100 Ω 2 W flameproof + 0.1 µF 275 VAC Class X2 across COM–NO the moment any motor, fan, pump or transformer load enters the rig.** X2 safety grade only; a general-purpose film cap in this position is a fire. It is ₹60 and listed in §2's optional table — buy it now if you think scope may creep.

### D12′ — Load interface: **ONE IS 1293 6 A panel socket + a 3-pin multi-plug adapter**

One socket = one switched point = one PZEM = S2 honoured. The multi-plug adapter in that socket is what makes S5 work: it puts the laptop brick and the phone charger **behind the same shunt at the same time**, so the pipeline sees a genuine superposed signal instead of one appliance at a time.

Inlet is a **pre-moulded** 6 A plug on 1.0 mm² flex through a cable gland. Do not wire your own plug top.

| Adapter choice | Verdict |
| :--- | :--- |
| 3-pin **hard multi-plug**, earthed, 6 A | ✅ Use this. Rigid, no switch, nothing to fail closed |
| Unswitched 2-way earthed converter | ✅ Acceptable |
| **Switched** strip / spike guard | ⚠️ Only if the switch is left ON and taped. A mid-demo switch-off looks identical to a relay trip in the logs |
| Any **unearthed** 2-pin adapter | ❌ Breaks D9. Laptop bricks are Class I |

The old 4-way strip is deleted — there is no fourth load to put on it, and it was the one BOM line that pushed past ₹3,000.

### D13 — 🔴 **USB and mains are mutually exclusive. Never both.**

PZEM-004T v3.0 clones cannot be assumed to galvanically isolate their TTL side from mains. USB ties ESP32 GND to the host laptop's chassis. Hard rule:

* **Flashing / serial monitor:** laptop USB connected, **mains fully disconnected**.
* **Running:** powered from the 5 V charger via the screw-terminal breakout, **laptop USB removed**.
* If you need live serial while on mains, use an isolated USB-serial adapter — not a plain cable.

> **The 5 V charger is not covered by this rule** and is the intended run-time supply. The hazard is a *host laptop*, which bonds ESP32 GND to an earthed chassis. A 2-pin isolated charger has no earth reference. See D6′.

---

## 2. Final BOM — one node, one port, ≤₹3,000, zero solder

Prices are **indicative Indian retail (Aug 2026)** from Robu / Robocraze / Quartz / Amazon.in class vendors — **verify at cart before ordering.** Nothing in the core list needs a soldering iron.

### Core build — order all of this

| # | Item | Exact part / spec | Qty | ₹ | Note |
| :-- | :--- | :--- | :-: | --: | :--- |
| 1 | MCU dev board | ESP32 DevKit V1, **ESP32-WROOM-32D**, **30-pin** | 1 | 400 | **Not WROVER** (D10) |
| 2 | 🆕 Screw-terminal shield | ESP32 **30-pin** GPIO expansion, screw terminals | 1 | 300 | Match pin count **and board width** (D6″) |
| 3 | Energy meter | PZEM-004T **v3.0, 10 A direct-connect** | 1 | 800 | **Not** the CT variant (D1) |
| 4 | Relay module | **2-ch** 5 V opto-isolated, SRD-05VDC-SL-C 10 A, **H/L trigger jumper** | 1 | 120 | Ch.1 live, ch.2 spare (D2/D3′) |
| 5 | 🆕 5 V supply | **BIS-marked 5 V 2 A USB charger** | 1 | 180 | IS 13252 (D6′) |
| 6 | 🆕 DC breakout | USB-A female → screw-terminal adapter | 1 | 70 | No solder on the DC side |
| 7 | **Pull-down resistor** | **100 kΩ 0.25 W** metal film, through-hole | 5 | 15 | 🔴 **Mandatory** — legs into screw terminals (D3′) |
| 8 | **Load fuse + holder** | **5 A ceramic 250 V** ×3 · inline **screw-type** holder | 3 · 1 | 100 | Resolves B-1; **not** 2–3 A, **not** 16 A (D8). Screw holder, not solder-tag |
| 9 | Mains terminal blocks | 3-way barrier screw — L, N **and PE** | 3 | 90 | Every mains joint is a screw joint; one block serves as the PE bond (D9) |
| 10 | Mains wire | 1.0 mm² stranded 300/500 V, brown + blue | 2 m ea | 90 | 13 A ampacity ≫ 5 A fuse |
| 11 | Earth wire | 1.0 mm² green/yellow stranded | 2 m | 40 | |
| 12 | DC wire | 0.5 mm² stranded, assorted | 2 m | 30 | |
| 13 | Enclosure | ABS **UL94-V0** junction box, closes fully | 1 | 220 | Flammability grade is not optional |
| 14 | Cable glands | M16 with locknuts | 3 | 60 | L-in, load-out, 5 V-in |
| 15 | Heat-shrink | Assorted 2–10 mm | 1 pack | 50 | Every mains joint |
| 16 | **Load socket** | **IS 1293 6 A 3-pin, panel mount** — exactly one | 1 | 70 | (D12′) |
| 17 | 🆕 **Multi-plug adapter** | 3-pin **earthed** 6 A hard multi-plug, unswitched | 1 | 50 | 🔴 **S5-critical** — this is what gives NILM real overlap (D12′) |
| 18 | **Mains inlet lead** | 6 A moulded plug + 1.0 mm² flex, 1.5 m | 1 | 150 | Pre-moulded plug — do not wire your own |
| 19 | **100 W incandescent lamp** + batten holder | Plain filament, **not** LED/CFL | 1 | 130 | Calibration reference **and** the only way to reach the 312 W trip (D8, §3) |
| 20 | Warning label | "⚠ 230 VAC INSIDE — ISOLATE BEFORE OPENING" | 1 | 10 | |
| | | | | **₹2,975** | ✅ **under ₹3,000** |

### Optional — deliberately outside the ₹3,000 core, in priority order

| Item | ₹ | Buy it if |
| :--- | --: | :--- |
| **30 mA RCD / RCBO portable adapter** | 700–1200 | 🔴 **Your house DB does not already have a 30 mA RCCB.** This is primary human protection, not an accessory. Check the DB first — if one is there, run the rig from a socket on it and this costs nothing |
| 2nd ESP32 DevKit V1 (spare) | 400 | You want the build to survive a wiring slip. This is the part whose loss stops everything |
| RC snubber: 100 Ω 2 W flameproof + 0.1 µF X2 | 60 | Any motor, fan, pump or transformer load may ever enter the rig (D11′) |
| DIN earth block + TS35 rail | 180 | You prefer a proper earth bar to a barrier block. Electrically equivalent at this scale |

**Honest note on the total:** ₹2,975 is the core rig only, and it assumes the ₹700–800 PZEM price holds. The RCD is the one optional line I would not skip — if your DB has no RCCB, the real all-in floor is ~₹3,700 and the budget cannot absorb it. Say so rather than building without it.

### Deleted from the previous BOM

| Removed | Superseded by |
| :--- | :--- |
| 100 A / 50 mA split-core CT | D1 — 10 A shunt variant needs no CT |
| SLA-05VDC-SL-C 30 A relay | D2 — SRD-05VDC-SL-C 10 A |
| 16 A MCB / 20 A ceramic fuse | D8 — 5 A ceramic |
| 2.5 mm² mains wire | D8 — 1.0 mm² is 2.6× the fuse rating |
| **4-way power strip** | 🆕 **D12′ — one socket + a ₹50 earthed multi-plug. No fourth load exists in scope** |
| **BSS138 / 2N7002 / 2N7000 + 1 kΩ gate resistor** | 🆕 **D3′ — SMD, unsolderable by hand. High-trigger module input replaces the whole inverter** |
| **Hi-Link HLK-5M05** | 🆕 **D6′ — solder-pin module with exposed AC pads → BIS USB charger** |
| **10 D471K MOV** · **0.5 A slow-blow fuse + holder** · **1000 µF low-ESR cap** · **100 nF X7R** | 🆕 **D6′ — all four are inside a certified sealed charger. The second mains tap disappears with them** |
| **Ferrules + crimper kit** (was ~₹500) | 🆕 Screw-terminal blocks throughout. **Still fold each stranded conductor back on itself** before clamping — that is the no-cost substitute for a ferrule, and bare loose strands in a screw terminal still arc |
| DIN rail TS35 · HV fibreglass sleeving | 🆕 Junction box + barrier blocks; no DIN mounting needed at this scale |
| 2 oz copper / >12 mm traces / air isolation slots (Hazard 5) | Not applicable at 5 A. 1 oz copper is ample; **retain the ≥6.3 mm creepage requirement** (IEC 62368) — that is voltage-driven, not current-driven, and 230 V has not changed |

---

## 3. Bench instrumentation — and what ₹3,000 cannot buy

> 🔴 **Read this before ordering.** The ₹3,000 ceiling is a **rig** budget. Bench instrumentation was never inside it, and three items previously listed as "keep" cost more than the entire rig. Pretending otherwise would mean reporting tests as passed that were never runnable.

| Must have | ₹ | Why |
| :--- | --: | :--- |
| **30 mA RCD/RCBO** — portable adapter, **or a house DB RCCB you have verified** | 0–1200 | Non-negotiable primary human protection. **Check the DB first**; if a 30 mA RCCB is present, work from a socket on it and spend nothing |
| VDE 1000 V insulated screwdriver (one is enough) | 250 | |
| Multimeter with continuity + DC volts | 500 | Enough for the B-4 test, the Stage-1 GPIO check and the Stage-2 relay check |

| Out of reach at this budget | ₹ | Consequence — state this, don't hide it |
| :--- | --: | :--- |
| 1:1 isolation transformer ≥300 VA | 2500+ | Work single-handed, one hand behind your back, on an RCD. Slower and less forgiving |
| **Variac 0–260 V** | 4000+ | 🔴 **Test 2 (brownout sag) is NOT RUNNABLE.** Report it as not-run, never as passed. The SMPS constant-power behaviour in D8 stays a calculation, not a measurement |
| Oscilloscope ≥20 MHz + isolated probe | 8000+ | Tests 1, 3, 4, 7 lose their waveform evidence. The relay/UART checks degrade to pass/fail by DMM |
| True-RMS CAT III meter with 10 A AC range | 2000+ | ⚠️ **The ±2 % calibration check in Stage 5 degrades.** A sub-₹1000 DMM cannot measure 2 A **AC** in series. Substitute below |
| IR thermometer | 800 | Stage 7 thermal soak becomes back-of-hand + smell. At 1.4 A that is a genuinely small risk, but it is a downgrade |

**Calibration without an AC ammeter.** Use the **100 W incandescent lamp as the reference standard** instead of a series meter: it is purely resistive, PF = 1, and its nameplate is accurate to about ±5 % at rated voltage. Measure actual mains voltage with the DMM's AC-volts range (cheap meters do handle AC *volts*), scale the expected power by (V/230)², and accept **±5 %** rather than ±2 %. Record the protocol you actually used — a ±5 % lamp check is a real result; a ±2 % claim without a series ammeter is not.

### Test loads

| Load | Purpose | Buy? |
| :--- | :--- | :--- |
| **100 W incandescent lamp** | The single most useful item. Purely resistive, PF = 1, instantaneous clean step edge — calibration reference, NILM ground truth, **and the ballast that makes the 312 W trip reachable** (D8). Every SMPS load has a soft-start that smears the transient | ✅ In BOM |
| **Laptop + its charger** | Primary load. 30–65 W idle, 65–200 W charging under CPU load | Already own |
| **Phone + charger** | Second overlapping load (S5). 3–10 W trickle → `PhantomTracker`; 18–120 W USB-PD → crosses the 20 W threshold and classifies | Already own |
| Monitor · projector · desktop | ❌ Out of scope by S1 — the 250 W ceiling and the single socket have no room for them | No |

---

## 4. Blocking-issue ledger

| ID | Issue | Status |
| :-- | :--- | :--- |
| **B-0** | 100 A CT is the wrong sensor for the demo band | ✅ **Resolved** — D1 |
| **B-1** | 0.5 A fuse in series with the load branch | ✅ **Resolved** — D8; the PSU branch no longer exists at all (D6′), leaving one 5 A load fuse |
| **B-2** | No protective earth anywhere in the design | ✅ **Resolved** — D9 |
| **B-3** | Root `Hardware.md` contradicted the deployment guide | ✅ **Resolved** — rewritten to this spec |
| **B-4** | PZEM TX may be 5 V push-pull; ESP32 absolute max is 3.6 V | ⚠️ **Open by design — bench test required.** Power the PZEM from 5 V with TX unconnected, measure TX-to-GND idle. ≈3.3 V or floating → connect direct (10 kΩ pull-up to 3.3 V if floating). ≈5 V → 1 kΩ/2 kΩ divider on PZEM-TX → GPIO 16. **Two 1 kΩ/2 kΩ resistors cost ₹6 — buy them with the 100 kΩ so you are not blocked mid-bring-up** |
| **B-5** | `firmware/esp32_node/src/README_PHASE2.md` documented an **entirely different sensor chain** — SCT-013-030 CT on GPIO 34 (ADC) + 33 Ω burden + relay on GPIO 5 (a strapping pin) + USB supply | ✅ **Resolved** — corrected in this change set |
| **B-6** | `config/config.demo.yaml` declared **5 nodes as `simulated: false`** while the build is one node | ✅ **Resolved** — the 5 are `simulated: true` (the `make demo` software fleet) |
| **B-7** | `RELAY_ACTIVE_LOW = true` inverted the relay: boot energised the load, and safety cutoffs closed it | ✅ **Resolved** — `main.cpp:67` is `false`, and under D3′ that is now the *direct* reading with zero inversions in the chain |
| **B-8** | 🆕 **3.3 V cannot turn OFF a 5 V low-trigger opto relay module.** IN at 3.3 V leaves ~0.5 mA in the opto LED — the relay may fail to release or chatter. This is the defect the deleted BSS138 was masking | ✅ **Resolved** — D3′ high-trigger direct drive. ⚠️ **Confirm at purchase and at Stage 2**; fallback documented in D5 |
| **B-9** | 🆕 **The 600 W ceiling made the safety cutoff unreachable on the real rig.** `config/config.demo.yaml` put CRITICAL at 750 W (3.26 A) while one port carrying a laptop + phone draws 150–320 W — the relay could never trip, so the headline safety feature was undemonstrable in hardware | ✅ **Resolved** — new `config/config.hardware.yaml` at **250 W → 312 W trip**. `config.demo.yaml` stays at 600 W because its ~1130 W *simulated* fleet needs it; the two files must not be merged |
| **B-10** | 🆕 **Zero-solder claim was false against the old BOM** — BSS138 is SOT-23 SMD and HLK-5M05 is a solder-pin module with exposed AC pads, neither buildable by a first-time builder (S4) | ✅ **Resolved** — D3′ and D6′ |

---

## 5. What this rig can and cannot demonstrate

| ✅ Real | ❌ Not this rig |
| :--- | :--- |
| Live V / I / P / PF / kWh from real appliances at 0.23 W resolution | Any load above 250 W — projector, desktop, kettle, oven, HVAC, EV |
| NILM separating **two overlapping loads** from **one** aggregate meter (S5) | Disaggregating 3+ simultaneous appliances — only two fit the socket and the ceiling |
| Local relay cutoff with the network unplugged (Core 0 only) | Arc-fault detection — dP/dt is quantised to the PZEM's ~500 ms register refresh (Hazard 4); a real AFDD is a separate certified device |
| Threshold + rate-of-change safety events, with the coordination ladder in D8 | Unattended shedding of a critical load |
| `laptop` and `phone_charger` ID as **advisory**, confidence shown | The other 5 trained classes — no socket on this rig can present them (`config.hardware.yaml` lists only the two) |
| Graceful degradation to `HeuristicApplianceClassifier` when `protonet.pt` is deleted | A headline accuracy number without naming the protocol |
| Brownout *reasoning* from D8's SMPS constant-power analysis | Brownout *measurement* — no Variac (§3) |

**Phone charger, stated precisely:** `TRANSIENT_THRESHOLD_W = 20.0` (`src/pipeline/aggregate_nilm.py:29`). A 3–10 W charger is below it and will **not** produce a classification event — correctly, since that step is inside sensor noise. It is tracked as standby draw by `PhantomTracker` (`src/pipeline/phantom_tracker.py:8`, `baseline_threshold_watts=5.0`). Fast USB-PD chargers at 18–120 W do cross the threshold and classify normally. Do not build the narrative around the small charger being *identified*; build it around phantom-load tracking, and use a **USB-PD fast charger** when you want a real classification event.

**On accuracy:** quote the unseen-house figure from `training_results/training_report.json`, never the episodic number. The `projector` caveat no longer applies to hardware claims — that class is out of scope (S1) — but it still applies to any figure you quote from the general model.

---

## 6. Pre-power-on inspection gate

**Wiring**
* ☐ Continuity: plug L → 5 A fuse → PZEM in-L → PZEM out-L → relay COM
* ☐ Continuity: plug PE → PE barrier block → enclosure → socket PE, **unswitched**, <0.1 Ω
* ☐ Isolation ≥10 MΩ: L/N to the 5 V rail, and L/N to enclosure earth
* ☐ Creepage ≥6.3 mm between every mains net and every 3.3 V/5 V net — inspect under magnification
* ☐ **Only one mains branch enters the box** (the load branch). The 5 V charger is external (D6′)
* ☐ Every stranded conductor **folded back on itself** before clamping, every screw torqued, every one tug-tested
* ☐ PZEM is **upstream** of the relay (D7)

**Polarity & orientation**
* ☐ Relay module jumper set to **H (high-trigger)** (D3′)
* ☐ GPIO 18 → relay `IN` **direct**; **100 kΩ from `IN` to `GND`** landed in the screw terminals (D3′)
* ☐ Relay module `VCC` = 5 V, `GND` common with the ESP32 — verified with a DMM, not assumed
* ☐ PZEM UART crossed: GPIO 16 (RX2) ← PZEM TX · GPIO 17 (TX2) → PZEM RX
* ☐ **B-4 logic-level measurement done and recorded** before GPIO 16 is connected
* ☐ Multi-plug adapter is **earthed** and its PE is continuous to the socket (D12′)

**Firmware & config**
* ☐ `RELAY_ACTIVE_LOW == false` at `main.cpp:67` — **unchanged; do not edit** unless the D5 fallback was taken
* ☐ `RELAY_PIN == 18`, `PZEM_RX_PIN == 16`, `PZEM_TX_PIN == 17` at `main.cpp:59-69` — unchanged, the shield only breaks pins out
* ☐ `DEVICE_ID` = `node_bench_agg`, `RATED_WATTS` = **250**, WiFi and MQTT broker set
* ☐ Pipeline launched with **`--config config/config.hardware.yaml`** (250 W ceiling) — **not** `config.demo.yaml` (600 W, simulated fleet) and not the 3500 W default (B-9)

---

## 7. Staged bring-up

Never skip a stage. Each has an abort condition.

| Stage | Connect | Verify | Abort if |
| :-: | :--- | :--- | :--- |
| **1** | Laptop USB only. No mains. Relay module unplugged | Boots, WiFi joins, MQTT connects; **GPIO 18 measures LOW** at idle and through a reset | Bootloop, or GPIO 18 idles HIGH → the load would be live at boot |
| **2** | USB + relay module + the 100 kΩ. **No mains on contacts** | `set_relay(true)` → **DMM continuity COM–NO closes**; `set_relay(false)` → opens. Verify with the meter, **not by ear** — a click is not continuity | Inverted, or will not latch at 3.3 V → take the **D5 fallback** (B-8), then repeat this stage |
| **3** | 5 V charger via the breakout on mains. **Laptop USB removed** (D13). Contacts open | 5 V rail holds ≥4.75 V through a WiFi TX burst | Rail dips below 4.4 V → charger is underrated or the breakout is high-resistance |
| **4** | PZEM in-line, load branch open. **B-4 test passed first** | Voltage ≈230 V, current 0.000 A | No UART response, or TX measured 5 V |
| **5** | 100 W lamp alone through the relay | PZEM power within **±5 %** of the lamp's voltage-scaled nameplate (§3); clean step edge visible in the pipeline | Error >5 % → run `scripts/calibrate_ct.py` |
| **6** | Multi-plug: **laptop + phone charger together** (S5) | Aggregate tracks both; NILM raises a distinct event on **each** plug-in, including the charger plugged in *while the laptop runs*; `laptop` and `phone_charger` both classify | Only one event ever fires, or the second load is invisible → overlap is not working, re-check §5 |
| **7** | Add the lamp to force **>312 W with the broker killed** | **Relay opens on CRITICAL with WiFi and MQTT down** — Core 0 alone | Cutoff needs the network → the safety path is not local |
| **8** | ~250 W continuous, enclosure closed, 1 h | No terminal or conductor warm to the back of the hand; no discolouration, no smell | Any hotspot → stop and re-torque |

**Verify before trusting the software**
* ☐ Relay cutoff works with WiFi and the broker down — Core 0 alone (Stage 7)
* ☐ Relay cutoff works with `backend/models/weights_demo/protonet.pt` deleted — safety trips, classification degrades to `src/pipeline/heuristic_fallback.py`
* ☐ Calibration recorded against the 100 W lamp, **with the protocol and tolerance you actually used** (±5 % by nameplate, or ±2 % only if you had a series AC ammeter)
* ☐ Test 2 (Variac brownout) recorded as **NOT RUN — no Variac** (§3), not as passed

---

## 8. Cross-references

| Topic | Document |
| :--- | :--- |
| Hazard root-cause analysis (6 hazards) | [`HARDWARE_DEPLOYMENT_GUIDE.md`](./HARDWARE_DEPLOYMENT_GUIDE.md) §2 |
| Procurement checkboxes & bring-up gates | [`HARDWARE_READINESS_CHECKLIST.md`](./HARDWARE_READINESS_CHECKLIST.md) |
| 8 physical bench test protocols | [`REAL_WORLD_TESTING_PLAN.md`](./REAL_WORLD_TESTING_PLAN.md) |
| Pinout quick reference | [`../Hardware.md`](../Hardware.md) |
| Physical-rig runtime config | [`../config/config.hardware.yaml`](../config/config.hardware.yaml) |
| Class/method signatures | [`ARCHITECTURE_AND_APIS.md`](./ARCHITECTURE_AND_APIS.md) |
