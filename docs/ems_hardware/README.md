# EMS Hardware Docs — Current Build

**Status:** current documentation bundle for the 38-pin ESP32-WROOM-32 bench
prototype. No hardware is assembled and mains authorization is **blocked**.

## Use these documents

1. [`ACTUAL_100A_CT_WIRING.md`](./ACTUAL_100A_CT_WIRING.md) — actual-board pin map and CT-meter correction, with visual diagrams [`actual_100a_ct_wiring.svg`](./actual_100a_ct_wiring.svg) and [`actual_low_voltage_pin_photo.svg`](./actual_low_voltage_pin_photo.svg).
2. [`WIRING.md`](./WIRING.md) — compact current low-voltage reference for the available CT meter.
3. [`PRE_POWER_MULTIMETER_CHECKLIST.md`](./PRE_POWER_MULTIMETER_CHECKLIST.md) — zero-trust audit, master pin table and physical test protocol.
4. [`SAFETY_GATES.md`](./SAFETY_GATES.md) — fixed mains boundary and release gates.
5. [`PROCUREMENT.md`](./PROCUREMENT.md) — current BOM, order gates and missing interface.
6. [`VALIDATION_STATUS.md`](./VALIDATION_STATUS.md) — physical evidence status.

## Canonical project references

- Full current low-voltage guide: [`../archive/claude_debug/WIRING_STEP_BY_STEP.md`](../archive/claude_debug/WIRING_STEP_BY_STEP.md)
- Firmware/config parity: [`../archive/claude_debug/HARDWARE_ALIGNMENT_CONTRACT.md`](../archive/claude_debug/HARDWARE_ALIGNMENT_CONTRACT.md)
- Staged bring-up: [`../archive/claude_debug/BRINGUP_RUNBOOK.md`](../archive/claude_debug/BRINGUP_RUNBOOK.md)
- Physical validation ledger: [`../omnirush/PHYSICAL_VALIDATION.md`](../omnirush/PHYSICAL_VALIDATION.md)
- Procurement artifact: [`../../reports/ems_procurement_bom_under_2500.txt`](../../reports/ems_procurement_bom_under_2500.txt)

## Current scope

- Low-voltage bench bring-up only until the physical safety gates pass.
- ESP32 DevKit V1 (30-pin); boot strap pin GPIO 12 strictly avoided; GPIO 16/17 reserved exclusively for PZEM UART2.
- GPIO 16/17 are the PZEM UART (9600 baud, 3.3V logic); GPIO 26 (Relay 1: Laptop charger) and GPIO 27 (Relay 2: Phone charger) are the 2-channel active-LOW relay inputs.
- GPIO 25 is configured as an output driven HIGH to power the relay optocoupler LEDs (3.3V source, few mA), eliminating boot chatter.
- The physically available PZEM is the 100 A CT variant; use
  [`ACTUAL_100A_CT_WIRING.md`](./ACTUAL_100A_CT_WIRING.md), not the retired
  10 A shunt topology.
- The physical socket envelope is the documented laptop + phone-charger,
  250 W prototype. Projector recognition remains a software
  scope item and is not physical authorization for this socket.

## Removed from the active build set

The old CT/MOSFET/HLK/30 A and high-current test-plan documents were retired.
Do not recreate their wiring, protection values or test procedures from logs.
