# ASTRA Current State

**Checkpoint:** CP-001-SAFETY-PARSER (software partial)
**Date:** 2026-10-04  
**HEAD:** `302e1b214c2a23e9d4061d847eb4abfc19baebf7` (`Updated Docs`)
**Working tree:** dirty with CP-001 safety/parser changes and the untracked CP-000 audit artifact; unrelated work must be preserved.

## Baseline facts

- The current checkout contains prior documentation commits; CP-001 changes are uncommitted in firmware, simulator, parser/API, and regression tests. `Untitled Document 1` and `__agent__/` are tracked at the current HEAD.
- The supplied audit in `Untitled Document 1` reports a prior NO-GO assessment: P0 safety issues H1/H2/H3, safety-claim issue H4, hardware contradiction H5, MQTT/parser failures, ML enrollment/open-set gaps, and dashboard truth gaps.
- Existing project documents claim different historical baselines (224, 433, 549, 626, and 641 Python tests). The live count will be established by the current commands, not documentation.
- `graphify-out/` exists and reports a graph built from `83ed9c1b`, but the installed `graphify` launcher currently fails with `ModuleNotFoundError: No module named 'graphify'`; the graph is therefore stale/unqueryable in this environment until tooling is repaired.
- Physical mains, brownout waveform, EMI, relay contacts, PZEM timing, calibration, and physical ML enrollment have not been performed by this execution loop. Physical status is `NOT RUN`.

## Live baseline execution

- Initial `venv/bin/python -m pytest tests/ -q`: **BLOCKED** — the venv Python symlink resolves to Python 3.12 while its populated packages are under `lib/python3.10`; `pytest` is not importable.
- Initial matching-runtime baseline before CP-001: **641 passed, 4 intentional audit failures, 3 warnings**.
- CP-001 full regression: `PYTHONPATH=venv/lib/python3.10/site-packages:. /usr/bin/python3.10 -m pytest tests/ -q --tb=short` → **646 passed, 3 warnings, 18.84 s**.
- Runtime packages used: Python 3.10.12, pytest 9.0.3, torch 2.11.0+cu130, numpy 2.2.6, FastAPI 0.136.0.
- Frontend command: `cd frontend && npm test -- --run` → **20 passed, 1 file, 1.45 s**; Node v24.14.0, npm 11.9.0, Vitest 4.1.10.
- CP-001 focused safety/protocol gate → **163 passed, 0 failed**; frontend remains **20 passed**.
- Firmware compilation was **NOT RUN** because PlatformIO/`pio` is unavailable.

## Runtime boundaries

1. **Firmware:** `firmware/esp32_node/src/main.cpp`; Core 0 safety sampling/PZEM/relay path, Core 1 MQTT/telemetry/command path.
2. **Digital twin:** `src/hardware/esp32_firmware_sim.py`; simulator contract used by HIL/chaos tests.
3. **Transport:** `src/hardware/mqtt.py`; in-memory test client and production MQTT integrations.
4. **Pipeline:** `scripts/run_pipeline.py` and `src/pipeline/*`; telemetry ingestion, safety monitor, transient/delta detection, classification, phantom tracking, analytics, persistence, and event publication.
5. **ML:** `src/models/protonet.py`, `src/pipeline/heuristic_fallback.py`, `scripts/enroll_demo_devices.py`, profile/config registry artifacts.
6. **API:** `src/api/main.py`; REST, MQTT bridge, label flow, health/readiness, WebSocket broadcasts.
7. **Frontend:** `frontend/src/App.jsx` and components; WebSocket/MQTT-derived dashboard state and energy views.
8. **Deployment/config:** `config/*.yaml`, `docker-compose.yml`, `mosquitto/config/*`, Makefile, PlatformIO.

## Current execution phase

`VERIFY → CHECKPOINT → REASSESS`. CP-001 closes the reproduced H1/H2/parser/B2 software regressions in simulator/API paths. H3 freshness, H4 claims, H5 hardware consistency, real broker ACL behavior, physical ML enrollment, dashboard truth, and physical validation remain open or blocked.

## Known contradiction requiring human/hardware decision

`claude_debug/HARDWARE_FINAL_SPEC.md` is marked authoritative and says single-socket laptop + phone charger, ≤250 W, explicitly not projector. `config/config.hardware.yaml`, `claude_debug/GOD_TIER_PLAN_2026-09-10.md`, and root `CLAUDE.md` scope lock include projector as a required physical class. The actual board/relay/PZEM/nameplate and intended physical build must decide whether code, config, or documentation changes; no release claim can proceed while this is unresolved.

## Hardware status

`PHYSICAL VALIDATION: NOT RUN`  
`Mains test authorization: BLOCKED pending authoritative hardware configuration and qualified safe procedure.`

## Next exact actions

1. Human-review the CP-001 firmware safety diff and perform qualified low-voltage/bench verification before any mains claim.
2. Resolve H3 with production PZEM transaction-age/liveness implementation and measured dependency timing.
3. Resolve H5's authoritative hardware/projector contradiction before ceiling or physical ML claims.
