# Current Low-Voltage Wiring — Actual 100A CT Build

**Board:** 38-pin ESP32-WROOM-32 DevKit. **Mains disconnected.** Dupont wires
are permitted only on this low-voltage side.

The available meter is the **PZEM-004T-100A(V4.0)-type external-CT variant**.
Its external CT measures one live conductor; its `L/N` terminals are voltage
sense. Do not use the retired 10 A shunt series path.

## Pin map

| ESP32 | Connect to | Required condition |
|---|---|---|
| GPIO17 / TX2 | PZEM RX | UART transmit |
| GPIO16 / RX2 | PZEM TX | Measure TX idle voltage first |
| GPIO18 / D18 | Relay IN1 | Direct connection; no MOSFET |
| GND | Relay GND and PZEM GND | One common low-voltage ground |
| VIN / 5 V | Relay VCC and PZEM 5 V | Use an isolated 5 V supply |

Relay hardware contract: the available board may have four channels; use only
channel 1. Channels 2–4 and the separate four-channel level shifter are unused.
Direct GPIO18 drive is permitted only after the actual relay is verified as
H/high-trigger with its VCC/JD-VCC arrangement and contact labels identified.

## PZEM TX safety check

1. Connect only PZEM 5 V and GND; leave TX disconnected.
2. Measure PZEM TX to GND with a DC multimeter.
3. For approximately 3.3 V, connect directly; if the output floats, add the
   documented 10 kΩ pull-up to 3.3 V.
4. For approximately 5 V, use either a confirmed level converter (`HV=5 V`,
   `LV=3.3 V`, common GND) or this divider:

```text
PZEM TX ── 1 kΩ ──┬── GPIO16
                  │
             2 kΩ to GND
```

Two 1 kΩ resistors in series may form the 2 kΩ leg. Never connect an unverified
PZEM TX directly to GPIO16.

## Relay safety state

- Set the relay module jumper to **H/high-trigger**.
- GPIO18 connects directly to IN1.
- Add a 100 kΩ resistor from IN1 to GND.
- `RELAY_ACTIVE_LOW = false`: LOW or reset/Hi-Z means relay open; HIGH means
  relay closed.
- Verify the actual module's VCC/JD-VCC arrangement, input current and COM/NO
  labels before connecting it.

## Power and USB separation

- Runtime: BIS-marked isolated 5 V 2 A charger → USB-A-female-to-screw-terminal
  breakout → 5 V/GND rail.
- The micro-USB adapter/cable is not a substitute for that breakout.
- Flash with laptop USB and mains disconnected. For runtime, remove laptop USB.
- Never use Dupont wires, breadboard tracks or loose USB wires for 230 V.

## Stop conditions

Stop before powering the assembly if the board is WROVER, a 30-pin shield is
being used on the 38-pin board, the relay has no verified H/L behavior, the
PZEM TX voltage has not been measured, or any PZEM AC/CT or relay contact wire
is connected during the low-voltage bench stage.
