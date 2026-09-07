# Pin-to-Pin Wiring Guide — Bench Build (38-pin ESP32, Zero Solder)

> **Audience:** first-time hardware builder with an ESP32-38P DevKit, PZEM-004T,
> relay module, and dupont jumpers.
> **Companion to:** [`HARDWARE_FINAL_SPEC.md`](./HARDWARE_FINAL_SPEC.md) (locked
> BOM & safety spec — authoritative where the two disagree).
> **Firmware source of truth:** `firmware/esp32_node/src/main.cpp:80-98`
>
> ⚠️ **This guide covers the LOW-VOLTAGE side only (≤5 V).** The mains 230 V side
> is summarized in §7 for orientation but must be built exactly per
> `HARDWARE_FINAL_SPEC.md` §0/§2 — screw terminals and 1.0 mm² wire, never jumpers.

---

## 0. The one rule everything hangs on

Your build has two separate worlds, and jumper cables are only legal in one of them:

| World | Voltage | Components | Wiring |
| :-- | :-- | :-- | :-- |
| **World 1 — DC/signal** | 3.3 V / 5 V | ESP32 ↔ PZEM ↔ relay module ↔ 5 V supply | ✅ Dupont jumpers fine |
| **World 2 — mains** | 230 VAC | Plug → fuse → PZEM AC → relay contacts → socket | ❌ **Screw terminals + 1.0 mm² wire ONLY. A dupont wire on 230 V is a fire.** |

Good news: with a 38-pin ESP32 and jumpers you can build and test **everything
except the mains side** first (§8).

---

## 1. Parts needed for the bench build

| # | Item | Qty | Note |
| :-: | :-- | :-: | :-- |
| 1 | ESP32 DevKit, **WROOM-32D**, 38-pin | 1 | **Not WROVER** — WROVER uses GPIO 16/17 for PSRAM and silently kills metering (spec D10). Check the metal can label. |
| 2 | PZEM-004T **v3.0, 10 A direct-connect** | 1 | Not the 100 A CT variant (spec D1) |
| 3 | 2-channel 5 V relay module, **H/L trigger jumper** | 1 | Set jumper to **H** before wiring (§3) |
| 4 | Female–Female dupont jumpers, 20 cm | **7** min (buy a 40-pack) | |
| 5 | Female–Male dupont jumpers, 20 cm | 4 (buy a 40-pack) | Runtime power feed + spares |
| 6 | Breadboard, 400 points | 1 | Recommended, not mandatory |
| 7 | **100 kΩ 0.25 W metal film resistor** | 1 (buy 5) | 🔴 **Mandatory** — relay IN → GND pull-down (§3) |
| 8 | 1 kΩ and 2 kΩ resistors | 2 + 2 | Only if the B-4 check fails (§4) |
| 9 | USB cable (laptop → ESP32) for flashing/bench power | 1 | |
| 10 | Multimeter with DC volts | 1 | For the B-4 check and Stage 2 |

Total for the bench side: well under ₹300 in cables/passives.

---

## 2. Pin-to-pin map (World 1)

From `main.cpp`: `RELAY_PIN = 18`, `PZEM_RX_PIN = 16`, `PZEM_TX_PIN = 17`.

| # | ESP32 pin (silkscreen) | Connects to | Cable type |
| :-: | :-- | :-- | :-- |
| 1 | **GPIO 17** (labeled `TX2`) | PZEM **RX** | F–F |
| 2 | **GPIO 16** (labeled `RX2`) | PZEM **TX** ⚠️ *do the B-4 voltage check FIRST (§4)* | F–F |
| 3 | **GPIO 18** (labeled `D18`) | Relay module **IN1** | F–F |
| 4 | **GND** (any GND pin) | Relay module **GND** | F–F |
| 5 | **GND** (a second GND pin) | PZEM **GND** | F–F |
| 6 | **VIN / 5V** | Relay module **VCC** | F–F |
| 7 | **VIN / 5V** | PZEM **5V** | F–F |
| 8 | Relay **IN1** ↔ **GND** | **100 kΩ resistor** (not a cable — §3) | component |

> The UART is **crossed on purpose**: ESP32 RX (GPIO 16) ← PZEM TX, and ESP32 TX
> (GPIO 17) → PZEM RX. That is how UART works, not a typo.

### Finding the pins on a 38-pin DevKit

All of `TX2/17`, `RX2/16`, `D18` are on the same header column, toward the USB
connector end. Reading down that column the order is roughly:
…`D19`, **`D18`**, `D5`, **`TX2`**, **`RX2`**, `D4`, `D2`, `D15`…

- Trust the **silkscreen label** over any description — board variants differ.
- **Never use `TX0` / `RX0`** — those are the USB flashing pins.

---

## 3. The 100 kΩ pull-down is NOT optional

Wire it from relay **IN1** to **GND**:

- On a breadboard: one leg in the same strip as the GPIO 18 wire, other leg in the GND rail.
- Without a breadboard: the through-hole legs push directly into the same
  dupont socket as the GPIO 18 wire / a GND wire.

**Why (spec D3′/D4):** during ESP32 reset and the ~250 ms bootloader window,
GPIO 18 is high-impedance (floating). Without the pull-down, the relay state at
boot is **undefined** — it could energize your load. With it, IN is held at 0 V
and **OPEN is the only possible boot state**. It costs ₹3.

### Relay module setup BEFORE wiring

- Set the module's **H/L trigger jumper to H (high-trigger)**.
- Firmware is written for this: `RELAY_ACTIVE_LOW = false` (`main.cpp:96`).
- A low-trigger module driven at 3.3 V cannot reliably turn OFF (~0.5 mA keeps
  flowing in the opto LED — spec B-8). **If your board has no H/L jumper, STOP**
  and read the D5 purchase-contingency fallback in `HARDWARE_FINAL_SPEC.md`
  before connecting anything.

---

## 4. B-4 check — do this BEFORE connecting PZEM TX to GPIO 16

The PZEM's TX pin may push 5 V; the ESP32 absolute max is 3.6 V.

1. Wire only **5 V and GND** to the PZEM (map rows 5 and 7). Leave **TX unconnected**.
2. Multimeter: black probe on GND, red probe on the PZEM **TX** pin, DC volts.
3. **≈3.3 V or floating** → connect TX direct (map row 2). ✅
4. **≈5 V** → do NOT connect it directly. Build a divider first:

```
PZEM TX ──[ 1 kΩ ]──┬──> GPIO 16
                    │
                 [ 2 kΩ ]
                    │
                   GND
```

5 V × 2/3 ≈ 3.3 V at the GPIO. Then connect GPIO 16 to the divider midpoint.

**Buy the 1 kΩ / 2 kΩ resistors with the 100 kΩ so you are not blocked mid-bring-up.**

---

## 5. Power — two modes, one rule (D13)

| Mode | Power source | When |
| :-- | :-- | :-- |
| **Bench / flashing** | Laptop USB → ESP32; 5 V appears on VIN and feeds relay + PZEM via jumpers 6 & 7 | Bring-up, firmware tests |
| **Runtime (mains involved)** | BIS-marked 5 V 2 A USB charger → USB screw-terminal breakout → 2× F–M jumpers to breadboard 5 V/GND rails | Final rig |

> 🔴 **Never USB and mains at the same time.** The PZEM's TTL side is not
> guaranteed isolated from mains, and USB ties ESP32 GND to your laptop's
> chassis. Flash with USB and mains off; run with the charger and USB removed.
> If you need live serial while on mains, use an isolated USB-serial adapter.

---

## 6. Relay truth table (high-trigger module, direct drive — spec D4)

| GPIO 18 | Relay IN | Opto LED | Relay | Load |
| :-- | :-- | :-- | :-- | :-- |
| LOW | 0 V | off | **OPEN** | dead |
| HIGH (3.3 V) | 3.3 V | ~2.1 mA | **CLOSED** | live |
| Hi-Z (reset/boot) | 0 V via 100 kΩ | off | **OPEN** | dead ✅ fail-safe |

---

## 7. The mains side (World 2) — orientation only, build per the spec

**Screw terminals and 1.0 mm² wire only. No jumpers ever.** Fixed load-path
order (spec D7/D8/D9):

```
Moulded plug L → 5 A ceramic fuse → PZEM AC in-L → PZEM AC out-L → Relay COM → Relay NO → socket L
Plug N          → PZEM AC in-N   → PZEM AC out-N → socket N
Plug PE         → PE barrier block → enclosure bond → socket PE   (NEVER switched)
```

Full BOM, creepage/torque/inspection gates: `HARDWARE_FINAL_SPEC.md` §2 and §6.
**Don't buy the mains parts until the bench build passes §8.**

---

## 8. What you can build and test tonight (no mains anywhere)

With just: ESP32 + relay module + PZEM + breadboard + jumpers + one 100 kΩ.

1. Wire map rows 1–8 (after the B-4 check on row 2).
2. Create `firmware/esp32_node/include/secrets.h` from `secrets.h.example`
   (Wi-Fi SSID/password, broker IP/port, broker user/password, device ID).
3. Flash: `platformio run -t upload` in `firmware/esp32_node/`.
4. Verify:
   - Board boots, Wi-Fi joins, MQTT connects (spec Stage 1).
   - **GPIO 18 measures LOW at idle and through a reset** (Stage 1 abort gate).
   - `set_relay(true)` → continuity COM–NO **closes**; `set_relay(false)` → opens,
     verified with a multimeter, **not by ear** (Stage 2).
   - PZEM responds on UART; voltage reads ~230 V **only once the AC side is
     later wired** — on the bench it reads ~0 V / no reply with AC absent, which
     is expected.
   - 1 Hz telemetry appears on `home/sensor/{DEVICE_ID}/power`.

This covers bring-up Stages 1, 2 and most of 4 of `HARDWARE_FINAL_SPEC.md` §7
with nothing dangerous energized.

---

## 9. Two honest caveats

1. **Dupont jumpers are fine for bring-up, not for the final enclosed build.**
   Get the ESP32 screw-terminal expansion shield for the final rig — but buy the
   **38-pin** variant to match a 38-pin DevKit (the spec's 30-pin shield will
   not seat). A GPIO 16 jumper that falls off mid-demo reads identically to a
   dead PZEM (spec D6″).
2. **Nothing here is "hardware validated"** until it runs with real loads through
   the full staged bring-up (spec §7 Stages 3–8) — bench wiring is simulation of
   the wiring, not physical validation.
