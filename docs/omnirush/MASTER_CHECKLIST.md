# ASTRA Omnirush Master Checklist

**Created:** 2026-10-04  
**Baseline commit:** `4e07b658`  
**Status:** BASELINE IN PROGRESS — no remediation has been applied by this execution loop.

Statuses: `[ ] NOT STARTED` · `[~] IN PROGRESS` · `[✓] VERIFIED` · `[!] FAILED` · `[?] BLOCKED` · `[N/R] NOT RUN` · `[S] SIMULATED ONLY` · `[P] PHYSICAL VERIFIED`

## Machine-readable checklist

```yaml
schema: astra-release-checklist/v1
last_verified: 2026-10-04
physical_validation: NOT RUN
release_recommendation: NOT READY
items:
  - id: G1
    priority: P0
    description: Reproducible software baseline
    status: IN PROGRESS
    evidence: 641 Python passed/3 warnings and 20 frontend passed; default venv command is broken
    tests: BASE-PY-MATCH, BASE-FE
    files: [docs/omnirush/CURRENT_STATE.md]
    hardware_requirement: none for software baseline
    blocking: true
    last_verified: 2026-10-04
  - id: H1
    priority: P0
    description: Dead-at-boot measurement state cannot energize relay
    status: IN PROGRESS
    evidence: Supplied audit reports a reproduction; current dirty firmware/twin changes require independent rerun
    tests: tests/test_relay_safety_boot_brownout.py
    files: [firmware/esp32_node/src/main.cpp, src/hardware/esp32_firmware_sim.py]
    hardware_requirement: physical boot test required for physical PASS
    blocking: true
    last_verified: 2026-10-04
  - id: H2
    priority: P0
    description: Safety trip wins atomically over ON and has one relay owner
    status: IN PROGRESS
    evidence: Supplied audit reports a race; current dirty changes require independent rerun
    tests: tests/test_relay_safety_boot_brownout.py
    files: [firmware/esp32_node/src/main.cpp, src/hardware/esp32_firmware_sim.py]
    hardware_requirement: physical relay/lockout test required for physical PASS
    blocking: true
    last_verified: 2026-10-04
  - id: H3
    priority: P0
    description: Freshness is based on successful measurement age and bounded blind time
    status: IN PROGRESS
    evidence: Supplied audit reports getter-count/timeout issue; current implementation pending trace
    tests: tests/test_relay_safety_boot_brownout.py, tests/test_hil_uart_corruption.py
    files: [firmware/esp32_node/src/main.cpp]
    hardware_requirement: PZEM timeout/measurement timing bench evidence
    blocking: true
    last_verified: 2026-10-04
  - id: H4
    priority: P0
    description: Prototype protection claims are separated from certified electrical safety
    status: IN PROGRESS
    evidence: Conflicting audit/spec/UI wording located; physical certification not supplied
    tests: docs/UI claim audit pending
    files: [Hardware.md, claude_debug/HARDWARE_FINAL_SPEC.md, frontend/src/App.jsx]
    hardware_requirement: qualified hardware review for safety claims
    blocking: true
    last_verified: 2026-10-04
  - id: H5
    priority: P1
    description: One authoritative hardware/profile/calibration contract
    status: FAILED
    evidence: HARDWARE_FINAL_SPEC.md says 250 W laptop/phone only; current config includes projector
    tests: tests/test_hardware_alignment.py pending baseline
    files: [claude_debug/HARDWARE_FINAL_SPEC.md, config/config.hardware.yaml]
    hardware_requirement: actual BOM/nameplates/wiring required
    blocking: true
    last_verified: 2026-10-04
  - id: B1
    priority: P1
    description: Broker identities and ACL grants match API/pipeline/firmware use
    status: IN PROGRESS
    evidence: ACL and compose identities inspected; real broker test pending
    tests: MQTT/API integration pending
    files: [mosquitto/config/acl, docker-compose.yml, src/api/main.py]
    hardware_requirement: no, real broker deployment required
    blocking: true
    last_verified: 2026-10-04
  - id: B2
    priority: P1
    description: Malformed MQTT events are rejected without killing bridge/readiness
    status: IN PROGRESS
    evidence: Supplied audit reports ValidationError escape; live reproducer pending
    tests: tests/test_api.py, tests/test_api_extended.py
    files: [src/api/main.py]
    hardware_requirement: no
    blocking: true
    last_verified: 2026-10-04
  - id: P1-PARSER
    priority: P1
    description: Empty, object, negative, NaN, and infinite power payloads are rejected
    status: IN PROGRESS
    evidence: Supplied audit reports empty/{} become 0 W and negative remains negative
    tests: tests/test_mqtt.py, tests/test_hil_uart_corruption.py
    files: [src/api/main.py, src/hardware/mqtt.py, scripts/run_pipeline.py]
    hardware_requirement: no
    blocking: true
    last_verified: 2026-10-04
  - id: M1
    priority: P1
    description: Physical ML registry is enrolled from held-out physical data and partial checkpoints fail closed
    status: [?] BLOCKED
    evidence: config.hardware.yaml has no registry_path; no physical enrollment evidence found
    tests: tests/test_ml_pipeline_recognition.py
    files: [config/config.hardware.yaml, src/models/protonet.py, scripts/enroll_demo_devices.py]
    hardware_requirement: physical captures and registry artifact
    blocking: true
    last_verified: 2026-10-04
  - id: M2
    priority: P1
    description: Unknown/out-of-envelope devices abstain rather than confidently misclassify
    status: IN PROGRESS
    evidence: Supplied audit reports envelope survivor renormalization risk
    tests: tests/test_ml_pipeline_recognition.py, tests/test_e2e_five_class_recognition.py
    files: [scripts/run_pipeline.py, src/models/protonet.py]
    hardware_requirement: physical unknown rehearsal for physical PASS
    blocking: true
    last_verified: 2026-10-04
  - id: M3
    priority: P1
    description: Aggregate disaggregation maintains active appliance state and handles deltas
    status: IN PROGRESS
    evidence: Supplied audit reports one classification per aggregate node; current delta path requires verification
    tests: tests/test_overlap_delta.py, tests/test_detector_path_e2e.py
    files: [scripts/run_pipeline.py, src/pipeline/aggregate_nilm.py]
    hardware_requirement: physical overlap validation for shipping claim
    blocking: true
    last_verified: 2026-10-04
  - id: U1
    priority: P1
    description: Dashboard distinguishes live, simulated, stale, inferred, unknown, rejected, and safety-tripped state
    status: IN PROGRESS
    evidence: Frontend contains random EnergyChart data; provenance audit pending
    tests: frontend/src/__tests__/test_all.jsx
    files: [frontend/src/App.jsx, frontend/src/components/EnergyChart/EnergyChart.jsx]
    hardware_requirement: physical telemetry needed for live-truth PASS
    blocking: true
    last_verified: 2026-10-04
  - id: G11
    priority: P1
    description: Soak, memory, storage, reconnect, and deterministic rollover bounds
    status: NOT STARTED
    evidence: no current execution evidence in this loop
    tests: targeted soak/capacity matrix pending
    files: [src/database/session.py, scripts/run_pipeline.py]
    hardware_requirement: no for software bounds; physical uptime optional
    blocking: true
    last_verified: 2026-10-04
  - id: G12
    priority: P0
    description: Final adversarial release audit with P0=0 and explicit physical status
    status: NOT STARTED
    evidence: release audit not run
    tests: full regression plus adversarial matrix pending
    files: [docs/omnirush/RELEASE_READINESS_REPORT.md]
    hardware_requirement: physical gates cannot be marked PASS without evidence
    blocking: true
    last_verified: 2026-10-04
```

## Human-readable gate summary

| ID | Priority | Gate | Status | Evidence / next proof |
|---|---:|---|---|---|
| G1 | P0 | Reproducible baseline | `[~]` | Run exact Python/frontend/version commands |
| H1 | P0 | Fail-safe boot measurement gate | `[~]` | Reproduce against current dirty tree, then test-first fix |
| H2 | P0 | Atomic safety/relay ownership | `[~]` | Reproduce safety-trip + ON race |
| H3 | P0 | Measurement freshness/liveness | `[~]` | Trace actual PZEM transaction timing |
| H4 | P0 | Truthful prototype safety claims | `[~]` | Reconcile claims and mark certification absent |
| H5 | P1 | Hardware/profile consistency | `[!]` | Resolve 250 W-only spec versus projector configuration |
| B1/B2 | P1 | MQTT identity and malformed-input resilience | `[~]` | Real broker and bridge tests |
| M1/M2/M3 | P1 | ML enrollment, abstention, disaggregation | `[?]/[~]` | Physical registry/evaluation unavailable in this environment |
| U1 | P1 | Dashboard truth/provenance | `[~]` | Remove or explicitly label synthetic data |
| G11 | P1 | Soak/capacity/storage bounds | `[ ]` | Deterministic boundedness tests |
| G12 | P0 | Final release audit | `[ ]` | Only after preceding gates |

**Overall:** `NOT READY` · **Physical validation:** `NOT RUN` · **No item may be promoted to `[P]` without physical evidence.**
