# Zero-Trust Pre-Power Hardware Audit and Multimeter Checklist

**Status: MAINS RELEASE BLOCKED.** This document is a pre-power gate, not
permission to energize a 230 V assembly. A qualified electrician must review
and perform the mains-side work.

## Audit conclusion

The active design has **no ESP32 ADC sensor path**. It reads the available
PZEM-004T-100A(V4.0)-type external-CT meter digitally over UART. Therefore:

- no SCT-013, ACS712, ZMPT101B, ADC midpoint, ADC RMS window, or external CT
  burden resistor is part of this build;
- GPIO16/17/18 are the only application signal pins;
- ADC2/Wi-Fi conflict and input-only output violations are absent in the active
  firmware because no ADC pin is used;
- do not create a 1.65 V bias on unused ADC pins just to satisfy an old analog
  design—the pins must remain unused;
- PZEM TX level, relay H/L mode, relay VCC/JD-VCC arrangement, CT polarity,
  exact PZEM markings, and mains isolation remain physical gates.

## Canonical hardware decisions

| Item | Final current truth | Status |
|---|---|---|
| MCU | 38-pin ESP32-WROOM-32 / D0WD-V3 | Confirm board silkscreen before wiring |
| PZEM | PZEM-004T-100A(V4.0)-type external CT | Exact manual/marking still to be recorded |
| PZEM TTL | `GND, TX, RX, 5V`; crossed UART | TX level must be measured first |
| PZEM AC side | `CT, CT, N, L`; CT around one live conductor only | Mains electrician gate |
| Relay | Available 4-channel 5 V board, channel 1 only; 2-channel H/L equivalent is acceptable | H/L, VCC/JD-VCC and contacts unverified |
| Relay logic | 4-channel active-LOW (GPIO 26, 27, 14, 32); LOW = closed/ON, HIGH = open/OFF | Active-LOW optocoupled; HIGH preloaded at boot |
| Level shifter | Four-channel board is unused | Do not insert an unverified converter into GPIO18 or UART |
| Power | External isolated/BIS-marked 5 V 2 A supply for runtime | Laptop USB is bench/flashing only |
| Load | One earthed 6 A socket; laptop + phone charger; 250 W profile | Projector is software-only |
| Protection | 5 A ceramic fuse, 1.0 mm² mains wire, unswitched PE, upstream 30 mA RCBO/RCD | Physical inspection required |

## Master ESP32 pin and electrical safety table

| ESP32 pin | Component | Direction | Voltage | Boot behavior | Status |
|---|---|---:|---:|---|---|
| `VIN/5V` | External 5 V rail; relay VCC; PZEM TTL 5V | Power in | 5 V | Not used until supply check | SAFE when polarity/rail checked |
| `3V3` | No external component | Power out | 3.3 V | Leave unused | SAFE / do not feed 5 V |
| `GND` | Relay GND and PZEM TTL GND | Reference | 0 V | Common low-voltage ground | SAFE; not a mains isolation barrier |
| `GPIO16/RX2` | PZEM TX | Input | 0–3.3 V maximum | UART2 input; no unverified TX connection | **BLOCKED until TX voltage measured** |
| `GPIO17/TX2` | PZEM RX | Output | 3.3 V logic | UART2 output | SAFE after PZEM RX label check |
| `GPIO18/D18` | Relay IN1 | Output | 0/3.3 V | Output latch preloaded OFF; external 100 kΩ pulls IN1 low while Hi-Z | SAFE only with verified H-trigger board |
| `GPIO1/TX0` | USB serial logging | Output | 3.3 V logic | USB/boot logging | Do not connect PZEM or relay |
| `GPIO3/RX0` | USB serial logging | Input | 3.3 V logic | USB/boot logging | Do not connect PZEM or relay |
| `GPIO0, 2, 5, 12, 15` | None | Unused | Boot-strapping | Leave unconnected | SAFE when unused |
| `GPIO32–39` | None | Unused | No analog signal | No bias circuit | SAFE when unused; GPIO34–39 are input-only |

### Relay truth table

This table is valid only after the physical board is confirmed H/high-trigger,
its 5 V/VCC/JD-VCC arrangement is verified, and the 100 kΩ resistor is fitted:

| GPIO18 | Relay contact CH1 `COM–NO` |
|---|---|
| LOW | Open |
| HIGH | Closed |
| Reset/Hi-Z | Open via IN1 pull-down |

Use CH1 only. Leave relay channels 2–4, `NC`, and the four-channel level
shifter disconnected.

## Do not plug in until these tests pass

### Step 1 — Power-off continuity and short checks

1. Disconnect wall mains, external 5 V, laptop USB, PZEM AC/CT wires, relay
   contact wires and all loads.
2. Put the DMM in continuity/ohms mode. Never use the current jack for these
   checks, and never measure resistance on an energized circuit.
3. Short the probes together and record the lead resistance.
4. Verify end-to-end continuity only for the intended low-voltage wires:
   - ESP32 VIN/5V → relay VCC;
   - ESP32 VIN/5V → PZEM TTL 5V;
   - ESP32 GND → relay GND and PZEM TTL GND;
   - GPIO18 → relay IN1;
   - GPIO17 → PZEM RX;
   - GPIO16 → PZEM TX, only through the approved direct/divider/converter path.
5. Measure the 100 kΩ resistor out of circuit. Record its actual value.
6. Reject any stable near-zero short between the separate DC rail and GND,
   DC rail and GPIO signal, or signal and GND. Do not demand infinity from an
   assembled circuit containing capacitors, LEDs, or regulators.
7. With the relay unpowered and labels confirmed, `COM–NO` must be open. Do not
   rely on relay sound.
8. With all mains conductors disconnected from DC electronics, inspect for no
   conductive path from mains L/N to the 5 V/GND rail. An ordinary DMM is not
   an insulation-resistance certification instrument.

### Step 2 — Low-voltage-only voltage checks

Use laptop USB for the ESP32-only stage. For ESP32 + relay + PZEM, prefer the
external regulated 5 V supply; a nominal laptop USB port is not a guaranteed
591 mA transient supply.

1. With relay and PZEM disconnected, measure ESP32 `VIN` to `GND`: expected
   approximately 5 V, correct polarity.
2. Measure GPIO18 to GND during reset and idle: expected LOW. Stop if it rises
   HIGH.
3. Connect the verified relay low-voltage pins and 100 kΩ pull-down. Measure:
   - relay VCC to GND: approximately 5 V;
   - relay IN1 to GND at reset/idle: approximately 0 V;
   - ESP32 3V3 pin: approximately 3.3 V, with no external 5 V applied to it.
4. Connect PZEM TTL `5V` and `GND` only. Do not connect PZEM TX yet.
5. Measure PZEM TX to PZEM GND with DC volts:
   - stable approximately 3.3 V: direct connection may be considered;
   - floating/undefined: use the documented pull-up arrangement and recheck;
   - approximately 5 V: **never direct-connect**; use `1 kΩ` series and `2 kΩ`
     from GPIO16 to GND, or a verified 5 V-to-3.3 V converter.
6. After the selected protection is installed, measure the GPIO16-side voltage
   before attaching it. It must stay within 0–3.3 V.
7. Under Wi-Fi transmit and relay pickup, measure the 5 V rail at ESP32 VIN,
   relay VCC and PZEM 5V. Target is ≥4.75 V; stop and correct brownout/sag if
   it falls below 4.4 V.
8. **ADC expectation:** no active ADC pins exist in this design. GPIO32–39 have
   no required 1.65 V rest voltage. Do not connect them or add a midpoint bias.

### Step 3 — Relay dry continuity test

Use the dedicated bench firmware, not production MQTT:

```text
cd firmware/esp32_node
pio run -e bench -t upload --upload-port COM3
pio device monitor -e bench -b 115200
```

The `bench` environment disables Wi-Fi/MQTT and accepts only:

```text
DRY_OFF
DRY_ON
STATUS
```

With mains and every load disconnected:

1. Boot/reset: `COM–NO` must be open.
2. Send `DRY_ON`: `COM–NO` must close.
3. Send `DRY_OFF`: `COM–NO` must open.
4. Verify CH1 only with the continuity meter. Leave channels 2–4 and `NC`
   open. If the result is inverted, intermittent, or cannot release, stop.
5. Reflash the normal `esp32dev` production environment after the dry test;
   never use the bench firmware on a mains-connected assembly.

### Step 4 — First 230 V connection and load test

This step is **qualified-electrician-only** and remains blocked until the
actual PZEM manual, CT polarity, relay labels, enclosure, PE, fuse holder,
creepage/clearance and RCBO/RCD have all been inspected.

For the actual CT board, the orientation is:

```text
Wall L → 5 A fuse → branch to PZEM L sense and relay COM
                  → relay NO → CT around one live conductor → socket L
Wall N → PZEM N sense and socket N
Wall PE → bonded PE barrier/enclosure → socket PE, never switched
```

The PZEM CT secondary must be connected to both marked CT terminals before
primary current is present. Never clamp around both L and N. Never use a
breadboard or Dupont wire for mains.

After de-energized inspection and an insulation-resistance test with sensitive
electronics protected as required by their manuals:

1. Use the verified upstream 30 mA RCBO/RCD.
2. Keep laptop USB disconnected; use the isolated external 5 V supply.
3. Energize with the relay commanded/open and no load.
4. Verify the PZEM voltage reading against a reference meter.
5. Test one 100 W incandescent lamp first, if available, and compare the
   PZEM reading with the reference meter.
6. Do not connect kettle, oven, HVAC, washing machine, EV charger, desktop PC,
   projector, or other unapproved loads.
7. Do not deliberately exceed the 250 W physical profile for a trip test until
   the relay, socket, wiring and protection ratings are explicitly reviewed.
   The firmware threshold is 312.5 W (250 × 1.25), which is above the nominal
   profile and is not by itself authorization for a 312.5 W load.

## Four-channel relay and four-channel level shifter disposition

- **Four-channel relay:** acceptable only if the actual board's CH1 input is
  verified H/high-trigger, 5 V/VCC/JD-VCC arrangement is understood, and
  `COM/NO/NC` labels are confirmed. Use CH1; leave CH2–CH4 unused. Firmware
  remains GPIO18 active-HIGH.
- **Two-channel relay replacement:** acceptable if it meets the same verified
  H/high-trigger contract. It is not required merely because old diagrams show
  two channels.
- **Four-channel logic-level shifter:** spare/not connected in the approved
  path. The relay input is driven directly only for a verified H-trigger board.
  A generic bidirectional shifter is not automatically a safe relay driver or
  PZEM TX protector. Use it only after its exact rail direction, pull-ups,
  voltage levels and boot behavior are documented and re-audited.

## Audit evidence and scope

Static code/doc audit found no active ADC, ADC2, input-only GPIO output, or
strapping-pin assignment. It did find stale 10 A shunt, 30-pin, active-LOW,
MOSFET and analog-CT references in historical documents. The current files
identify those as historical; do not use them for physical assembly.

Physical validation is still **NOT RUN** until actual meter readings, board
labels, relay truth table, rail sag, CT polarity, PE continuity, isolation,
enclosure and RCBO/RCD evidence are recorded.
