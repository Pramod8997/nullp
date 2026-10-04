# ASTRA Hardware State and Consistency Record

**Physical validation:** `NOT RUN`  
**Mains authorization:** `BLOCKED` pending a single authoritative build definition and qualified review.

## Specifications found

| Source | Declared hardware/profile | Status |
|---|---|---|
| `claude_debug/HARDWARE_FINAL_SPEC.md` | ESP32-WROOM-32D DevKit, PZEM-004T v3.0 10 A direct, high-trigger relay, 5 V 2 A USB supply, 250 W, one socket, laptop + phone charger; projector explicitly out of scope | Marked authoritative in document |
| `claude_debug/WIRING_STEP_BY_STEP.md` | 38-pin DevKit, GPIO 16/17 UART, GPIO 18 relay, high-trigger module, 100 kΩ pull-down; low-voltage guide only | Companion; must match actual purchased board |
| `firmware/esp32_node/src/main.cpp` | PlatformIO `esp32doit-devkit-v1`; GPIO/constants and relay/PZEM behavior in source | Code source of truth for shipped binary, uncommitted changes present |
| `config/config.hardware.yaml` | One node `node_bench_agg`, 250 W, classes phone/laptop/projector, delta overlap enabled, no physical registry path | Contradicts the 250 W spec's explicit projector exclusion |
| `claude_debug/GOD_TIER_PLAN_2026-09-10.md` | Projector included; proposes 400/500 W ceiling decision from nameplate | Plan, not physical evidence |
| `config/config.demo.yaml` | Simulator/demo profile with nonphysical classes/loads | Simulation only |
| `docs/PRODUCTION_TESTING_AND_HARDWARE_AUDIT.md` | Older active-low/30 A/HLK/CT design | Historical and superseded; unsafe as current build instruction |

## Required actual-hardware inventory (not supplied/verified)

| Field | Required evidence | Current state |
|---|---|---|
| Board and MCU | photo/marking and serial | UNKNOWN |
| Relay module and trigger polarity | part marking, jumper position, measured GPIO truth table | UNKNOWN |
| PZEM variant | 10 A direct vs 100 A CT, part marking | UNKNOWN |
| UART pins/levels | continuity and measured TX voltage | UNKNOWN |
| DC power supply | nameplate, isolation/rating, rail measurement | UNKNOWN |
| Fuse/RCBO/PE/enclosure/wiring | qualified inspection and build record | UNKNOWN |
| Loads/nameplates | photographed ratings and safe test plan | UNKNOWN |
| Calibration reference | instrument identity and readings | UNKNOWN |

## Blocking decision

Do not change `RATED_WATTS`, relay thresholds, or safety assumptions to make the projector fit. A human must choose one of:

1. retain the 250 W laptop/phone-only build and remove projector from the physical claim;
2. approve a safety-reviewed projector-capable hardware/rating amendment with matching BOM, firmware, config, wiring, and tests; or
3. declare the current physical hardware unsupported for projector recognition.

Until selected, hardware consistency is `FAILED` and all mains/physical gates are `BLOCKED`.
