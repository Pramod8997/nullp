# ASTRA Current State

**Checkpoint:** CP-000-BASELINE (subagent review pending)  
**Date:** 2026-10-04  
**HEAD:** `4e07b658 Fixed minor firmware issues and Ml pipeline issues`  
**Working tree:** dirty; pre-existing user changes are present and must be preserved.

## Baseline facts

- Repository is a git worktree with uncommitted modifications in firmware, simulator, and tests, plus untracked `Untitled Document 1` and `__agent__/`.
- The supplied audit in `Untitled Document 1` reports a prior NO-GO assessment: P0 safety issues H1/H2/H3, safety-claim issue H4, hardware contradiction H5, MQTT/parser failures, ML enrollment/open-set gaps, and dashboard truth gaps.
- Existing project documents claim different historical baselines (224, 433, 549, 626, and 641 Python tests). The live count will be established by the current commands, not documentation.
- `graphify-out/` exists and reports a graph built from `83ed9c1b`, but the installed `graphify` launcher currently fails with `ModuleNotFoundError: No module named 'graphify'`; the graph is therefore stale/unqueryable in this environment until tooling is repaired.
- Physical mains, brownout waveform, EMI, relay contacts, PZEM timing, calibration, and physical ML enrollment have not been performed by this execution loop. Physical status is `NOT RUN`.

## Live baseline execution

- Initial `venv/bin/python -m pytest tests/ -q`: **BLOCKED** — the venv Python symlink resolves to Python 3.12 while its populated packages are under `lib/python3.10`; `pytest` is not importable.
- Reproducible matching-runtime command: `PYTHONPATH=venv/lib/python3.10/site-packages:. /usr/bin/python3.10 -m pytest tests/ -q --tb=short` → **641 passed, 3 warnings, 22.70 s**.
- Runtime packages used: Python 3.10.12, pytest 9.0.3, torch 2.11.0+cu130, numpy 2.2.6, FastAPI 0.136.0.
- Frontend command: `cd frontend && npm test -- --run` → **20 passed, 1 file, 1.45 s**; Node v24.14.0, npm 11.9.0, Vitest 4.1.10.
- New focused reproductions in `tests/test_audit_reproductions.py`: **4 failed as intended** against current code (H1, H2, parser, B2). These are now the first red regression gate and must not be weakened.

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

`DISCOVER → BASELINE`. No source fix is authorized until the baseline suite and defect reproduction matrix are recorded. The first implementation target, if reproduced, is the highest-severity P0 safety gate; the existing dirty safety changes are not assumed correct merely because tests were added.

## Known contradiction requiring human/hardware decision

`claude_debug/HARDWARE_FINAL_SPEC.md` is marked authoritative and says single-socket laptop + phone charger, ≤250 W, explicitly not projector. `config/config.hardware.yaml`, `claude_debug/GOD_TIER_PLAN_2026-09-10.md`, and root `CLAUDE.md` scope lock include projector as a required physical class. The actual board/relay/PZEM/nameplate and intended physical build must decide whether code, config, or documentation changes; no release claim can proceed while this is unresolved.

## Hardware status

`PHYSICAL VALIDATION: NOT RUN`  
`Mains test authorization: BLOCKED pending authoritative hardware configuration and qualified safe procedure.`

## Next exact actions

1. Run and record current Python suite, frontend suite, versions, and targeted audit reproducers.
2. Create CP-000 with exact counts/failures.
3. Launch narrow domain reviewers against these artifacts; reviewers must not patch safety code before reporting.
4. Cross-review findings and choose one P0 root cause for test-first remediation.
