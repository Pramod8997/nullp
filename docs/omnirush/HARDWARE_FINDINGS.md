# Hardware Findings — Agent 3 Consolidation

**Date:** 2026-10-04  
**Physical validation:** **NOT RUN**  
**Mains authorization:** **BLOCKED** pending an approved, single hardware contract and qualified review.

## Current assumptions and evidence status

| Area | Current software/document assumption | Evidence status |
|---|---|---|
| Board | ESP32-WROOM-32D DevKit V1; GPIO 16/17 PZEM UART; GPIO 18 relay | Documentation/code only; actual board and pin count unknown |
| Board format | Final spec references 30-pin; wiring guide references 38-pin | Contradictory; purchase/build identity unknown |
| Relay | 5 V opto-isolated H-trigger module, direct active-HIGH GPIO 18, 100 kΩ pull-down | Part, jumper, polarity, and contact behavior unknown |
| PZEM | PZEM-004T v3.0 10 A direct shunt | Actual PZEM variant and TX level unknown; historical docs use 100 A CT |
| Power | External BIS-marked isolated 5 V 2 A USB charger | Actual PSU, isolation, rail sag, and placement unknown |
| Mains protection | 5 A fuse, 1.0 mm² wiring, 6 A socket, upstream 30 mA RCBO, PE unswitched | No inspection, continuity, isolation, or rating evidence |
| Loads | Laptop + phone charger on one earthed multi-plug; projector appears in current software scope | Actual load nameplates and approved scope unknown |
| ML scope | phone/laptop/projector; LED bulb phantom tracked; fan demo-only/drop scope | Software decision exists; physical class availability unverified |

## Blocking contradictions

1. `claude_debug/HARDWARE_FINAL_SPEC.md` is marked authoritative and locks a single-socket laptop + phone-charger prototype at ≤250 W while explicitly excluding projector.
2. `config/config.hardware.yaml`, root scope lock, and the 2026-09-10 plan advertise projector as a physical class.
3. The final spec and wiring guide disagree on 30-pin versus 38-pin board/shield.
4. Current firmware uses active-HIGH direct GPIO 18 semantics; older documents describe active-LOW/MOSFET designs. Only the actual purchased relay module and measured truth table can resolve this.
5. Current firmware/config use a 250 W ceiling, but projector-related planning proposes a safety-reviewed 400/500 W amendment. No nameplate or safety review exists.
6. The hardware profile has no physical registry artifact; current ML recognition is not hardware evidence.

## Required human decision

Before changing thresholds, wiring instructions, or projector claims, select one:

- retain the 250 W laptop/phone-only build and remove projector from the physical claim;
- approve a safety-reviewed projector-capable BOM/rating amendment with matching firmware/config/tests; or
- declare the current physical hardware unsupported for projector recognition.

Do not infer this decision from simulator tests or documentation prose.

## Safe evidence checklist

- Photograph/record board marking and pin count, relay part/jumper/polarity, PZEM variant, PSU, fuse, socket, RCBO, enclosure, PE path, and multi-plug.
- With mains disconnected, verify GPIO 18 idle/reset level, relay COM–NO behavior, pull-down, and low-voltage wiring.
- Measure PZEM TX idle voltage before connecting it to ESP32 GPIO 16; use the documented level interface only if the measurement requires it.
- Have a qualified person verify live/conductor path, fuse/wire ratings, PE continuity, isolation, creepage, strain relief, terminal torque, and emergency shutdown before mains.
- Record firmware hash, active config path, credentials identity, raw PZEM/reference-meter readings, load nameplates, and calibration artifacts.
- Keep all resulting tests classified as physical only when raw measurements and operator/equipment evidence are retained.

## Disposition

Hardware consistency: **FAILED/BLOCKED**.  
Physical validation: **NOT RUN**.  
No simulator/HIL/static result promotes any item to `PHYSICAL VERIFIED`.
