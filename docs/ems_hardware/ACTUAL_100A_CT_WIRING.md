# Actual Hardware Wiring — 38-pin ESP32 + PZEM-004T-100A CT

![Actual 100A CT wiring diagram](./actual_100a_ct_wiring.svg)

![Annotated low-voltage connection photo](./actual_low_voltage_pin_photo.svg)

**Applies to:** the hardware shown for this bench: 38-pin ESP32-WROOM-32,
PZEM-004T-100A(V4.0)-type meter with external CT, and the available relay
module. The relay board may have four channels; this design uses channel 1
only. A verified two-channel H/L equivalent is also acceptable.

**Status:** the low-voltage pin map below is the wiring target. Mains wiring is
still blocked until the relay labels/trigger mode, CT polarity, enclosure,
protective earth, RCD/RCBO, fuse holder, and isolation are physically checked.

> **Never put 230 VAC on the breadboard or in Dupont jumpers.** The diagrams in
> Sections 1–4 are the only part that may be assembled for the first bench test.

## 1. Exact low-voltage pin-to-pin map

Use the labels printed on the boards. Do not substitute `TX0/RX0`, GPIO 5, or a
30-pin pinout.

| ESP32 38-pin label | Connect to | Wire / condition |
|---|---|---|
| `VIN` / `5V` | Relay module low-voltage `VCC` or `+5V` | 5 V supply; only after the relay board label is verified |
| `VIN` / `5V` | PZEM TTL `5V` | 5 V supply |
| `GND` | Relay module `GND` | Shared low-voltage ground |
| `GND` | PZEM TTL `GND` | Shared low-voltage ground |
| `GPIO18` / `D18` | Relay `IN1` | Direct signal for the approved H-trigger configuration |
| `GPIO17` / `TX2` | PZEM TTL `RX` | UART transmit, crossed |
| `GPIO16` / `RX2` | PZEM TTL `TX` | **Only after the TX voltage check in Section 3** |
| Relay `IN1` | Any shared `GND` | `100 kΩ` pull-down resistor; mandatory |

```text
                         LOW-VOLTAGE BENCH ONLY

 ESP32 5V/VIN  ───────────────┬──────── Relay +5V/VCC
                               └──────── PZEM TTL 5V

 ESP32 GND     ───────────────┬──────── Relay GND
                               └──────── PZEM TTL GND

 ESP32 GPIO18 / D18 ─────────────────── Relay IN1
                         │
                         └──[ 100 kΩ ]── ESP32 GND

 ESP32 GPIO17 / TX2 ─────────────────── PZEM RX
 ESP32 GPIO16 / RX2 ◄──── PZEM TX  (direct or divider, Section 3)

 PZEM AC/CT screw terminals: DISCONNECTED during this bench stage.
 Relay COM/NO/NC contacts:   DISCONNECTED during this bench stage.
```

### Relay-board identity gate

The relay photograph does not make the channel count and all terminal labels
reliably readable. Before connecting `5V`, `GND`, or `GPIO18`, confirm on the
actual board or its exact manual:

1. `IN1`, `GND`, and the 5 V input labels.
2. The H/L selector position and which position is **H/high-trigger**.
3. Whether the board has `VCC` and `JD-VCC`, and whether its jumper joins them.
4. Channel-1 contact labels `COM`, `NO`, and `NC`.

For the approved target configuration, set the selector to **H**, keep the
documented 5 V arrangement, connect only `IN1` for channel 1, and leave channels
2–4 (or any channels beyond it) unused. If the actual module has no verifiable H
mode or its labels differ, stop; do not infer the wiring from the relay-coil
marking alone.

The separate four-channel logic-level shifter is **not** in the approved path.
Leave it disconnected. A generic bidirectional shifter is not automatically a
safe relay driver or PZEM TX protector.

With the approved H-trigger arrangement and `RELAY_ACTIVE_LOW = false`:

| GPIO18 state | Relay result |
|---|---|
| LOW | Channel 1 open |
| HIGH (3.3 V) | Channel 1 closed |
| Hi-Z during reset/boot | 100 kΩ holds `IN1` low; channel 1 must be open |

Do not connect any relay contact terminal to the breadboard or to mains for the
first test. Verify `COM–NO` with a continuity meter while toggling the relay.

## 2. PZEM TTL header

Use the manual's header order, not a wire-color assumption. The supplied
PZEM-004T-100A(V4.0)-type board header is:

| PZEM TTL header position | Function | ESP32 connection |
|---:|---|---|
| 1 | `GND` | Any ESP32 `GND` |
| 2 | `TX` | ESP32 `GPIO16/RX2`, after Section 3 |
| 3 | `RX` | ESP32 `GPIO17/TX2` |
| 4 | `5V` | ESP32 `VIN/5V` |

The UART is crossed intentionally: PZEM `TX → ESP32 RX`, and ESP32 `TX → PZEM
RX`. Do not use ESP32 `TX0/RX0`; those pins are reserved for USB flashing and
serial logging.

## 3. Mandatory PZEM-TX voltage check

The ESP32 GPIO maximum is 3.6 V. Do this with the PZEM AC/CT terminals still
disconnected:

1. Power only the ESP32 and PZEM TTL side from the low-voltage 5 V rail.
2. Connect PZEM `5V` and `GND`; leave PZEM `TX` disconnected from GPIO16.
3. Measure PZEM `TX` to PZEM `GND` with a DC multimeter.
4. Use exactly one result:

| Measured PZEM TX | Connection to GPIO16 |
|---|---|
| Approximately 3.3 V | Direct connection is acceptable after confirming the reading is stable |
| Floating / high impedance | Do not guess; use the documented pull-up arrangement and recheck |
| Approximately 5 V | **Never direct-connect.** Use a divider: `TX → 1 kΩ → GPIO16`, and `GPIO16 → 2 kΩ → GND` |

```text
For a measured 5 V PZEM TX:

 PZEM TX ──[ 1 kΩ ]──┬──── ESP32 GPIO16 / RX2
                     │
                  [ 2 kΩ ]
                     │
                    GND
```

Do not connect PZEM `TX` directly to GPIO16 before this measurement.

## 4. PZEM-004T-100A(V4.0) AC/CT terminals

This is the important difference from the project's old 10 A direct-connect
diagram. On the actual CT meter, the load conductor does **not** pass through a
PZEM shunt. The external CT measures one conductor.

Using the manual's stated top-to-bottom order:

| PZEM AC/CT terminal | Function | Connection |
|---:|---|---|
| 1 | `CT` | One CT secondary lead |
| 2 | `CT` | The other CT secondary lead |
| 3 | `N` | Neutral voltage-sense connection; mains work only |
| 4 | `L` | Fused line voltage-sense connection; mains work only |

The CT must surround **one conductor only**, the live conductor. Do not put a
complete two-wire cable through the CT because live and neutral cancel. If the
CT has an arrow, orient it from source/inlet toward the socket/load. Confirm
polarity with a qualified person and a controlled reference load; a reversed CT
can produce a wrong-sign or invalid power result.

```text
 CT terminal 1 ───────── CT secondary lead 1
 CT terminal 2 ───────── CT secondary lead 2

 CT clamp around ONLY this conductor:
       relay NO ────────( CT around live wire )────── socket L
                          arrow → socket/load

 PZEM terminal 3 (N) ─── voltage-sense N
 PZEM terminal 4 (L) ─── fused voltage-sense L
```

The PZEM `L` and `N` terminals are voltage-sense/mains-side connections; they
are not an `L in` / `L out` current path for this 100 A CT version.

## 5. Mains-side topology — orientation only, not a self-build instruction

Have a qualified electrician make and test this inside a closed, flame-rated
enclosure. The exact board labels and CT direction must be checked against the
physical unit before energizing it.

```text
 Wall Live (L)
      │
      ├── 5 A fuse ──┬── PZEM L voltage sense
      │              │
      │          Relay COM ── Relay NO ──(CT around this live wire)── Socket L
      │
 Wall Neutral (N) ───┬── PZEM N voltage sense
                     └──────────────────────────────────────────── Socket N

 Wall PE ─────────── PE barrier / enclosure bond ───────────────── Socket PE
                     (protective earth is never switched and does not pass
                      through the CT or relay contacts)
```

This topology keeps PZEM voltage present when the relay opens, while the CT
reads zero current after the socket is disconnected. Do not reuse the old
10 A diagram showing `L in → PZEM → L out`; that is the wrong current topology
for this actual 100 A CT board.

Required before any mains energization:

- upstream verified 30 mA RCD/RCBO;
- 5 A fuse and suitable holder;
- 1.0 mm² mains conductors and screw terminals, never Dupont wires;
- closed enclosure, strain relief, creepage/clearance, and enclosure PE bond;
- relay `COM/NO` continuity and open-at-boot verified with mains disconnected;
- CT terminals, CT orientation, PZEM `L/N`, and relay labels checked by a
  qualified person;
- laptop USB removed before mains is connected.

## 6. Power-mode rule

| Mode | Power | Mains |
|---|---|---|
| Flash / bench | Laptop USB → ESP32 USB; VIN supplies the low-voltage modules | Fully disconnected |
| Runtime | Isolated/BIS-marked 5 V 2 A charger → USB screw-terminal breakout → 5 V/GND rail | Laptop USB removed |

Never use laptop USB and a live mains-connected PZEM assembly together.

## 7. First safe assembly order

1. ESP32 alone on laptop USB; confirm serial output and GPIO18 LOW during reset.
2. Add the relay low-voltage connections and 100 kΩ pull-down; verify open at
   boot and `COM–NO` continuity changes with a meter.
3. Power the PZEM TTL side only; perform the TX voltage check.
4. Add the PZEM UART connection, using the divider if required.
5. Leave all PZEM AC/CT terminals and relay contact terminals disconnected.
6. Only after the low-voltage checks pass should a qualified electrician review
   the CT/mains enclosure wiring.

For dry relay continuity, use the dedicated PlatformIO `bench` environment and
its `DRY_ON`/`DRY_OFF` serial commands with relay contacts disconnected. Do not
use production MQTT `ON` for a no-mains test; production firmware requires a
valid 80–260 V PZEM frame before it can close the relay.

## References

- Firmware pins: `firmware/esp32_node/src/main.cpp:82-95`.
- PZEM-004T manual/datasheet mirror: <https://arduino.ua/files/PZEM-004T-V3_0-Datasheet-User-Manual.pdf>
- Additional datasheet mirror: <https://www.mantech.co.za/Datasheets/Products/pzem-004t-250109a.pdf>
- Project safety gates: [`SAFETY_GATES.md`](./SAFETY_GATES.md) and
  [`VALIDATION_STATUS.md`](./VALIDATION_STATUS.md).
