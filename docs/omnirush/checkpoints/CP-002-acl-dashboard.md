# CP-002 — ACL and Dashboard Truth

**Date:** 2026-10-05  
**Recommendation:** **NOT READY**  
**Physical validation:** **NOT RUN**

## Scope

This checkpoint follows CP-001-SAFETY-PARSER and covers the remaining directly actionable software defects in the broker contract, physical-profile ML behavior, firmware liveness guards, and dashboard truth boundary.

## Changes

- Added `ems_pipeline` ACL read permissions for `home/ui/events` and `home/plug/+/command`.
- Added `tests/test_mqtt_acl_contract.py` to prevent regression of the static ACL contract.
- Added physical-profile fail-closed ML behavior when model/registry/class/envelope artifacts are missing or incomplete.
- Added firmware source guards for successful PZEM-read age, bounded blind time, safety-task liveness, and task-creation failure.
- Removed random historical energy generation from `EnergyChart`.
- Removed fixed-assumption energy, cost, and savings values from `SummaryCards`.
- Added truthful unavailable states until backend analytics/history are supplied.
- Passed analytics into overview and analytics pages rather than deriving historical values from live wattage samples.

## Verification

Commands:

```bash
PYTHONPATH=venv/lib/python3.10/site-packages:. /usr/bin/python3.10 -m pytest tests/test_mqtt_acl_contract.py tests/test_physical_ml_fail_closed.py tests/test_hardware_alignment.py -q --tb=short
git diff --check
PYTHONPATH=venv/lib/python3.10/site-packages:. /usr/bin/python3.10 -m pytest tests/ -q --tb=short
cd frontend && npm test -- --run
```

Results:

- Focused CP-002 gate: **23 passed**.
- Full Python suite: **650 passed, 4 warnings**.
- Frontend Vitest: **22 passed**.
- Whitespace check: **PASS**.
- Firmware PlatformIO compile: **NOT RUN — PlatformIO unavailable**.
- Authenticated Mosquitto deployment/replay test: **NOT RUN**.
- Physical validation: **NOT RUN**.

## Open gates

- H3: installed PZEM library transaction timing and measured blind interval.
- H4: qualified review of prototype-only safety claims.
- H5: authoritative hardware/projector/profile contradiction.
- B1: authenticated broker identity, ACL, retained, replay, duplicate, stale, and reconnect behavior.
- M1–M3: physical enrollment, held-out evaluation, open-set rejection, and overlap evidence.
- U1: measurement age, provenance, offline/stale state, and physical relay-contact mapping.
- G11: soak, storage, reconnect, and capacity bounds.
- Physical mains, EMI, brownout, thermal, calibration, and relay/PZEM validation.

## Final decision

CP-002 improves software defensibility but does not authorize deployment or mains operation. The release remains **NOT READY**.
