# Verification Ledger — 2026-09-10 (Run 0.0)

> **What this is:** the persisted evidence base of the twelve-specialist verification that
> preceded the Wave-1 execution plan (`.claude/plans/also-make-sure-the-gleaming-sparkle.md`).
> Twelve read-only specialist agents ran against live code with measurements: six in the first
> round (firmware/embedded-safety, twin/HIL parity, contract-integrity, ML/meta-learning, math,
> twin-signal-fidelity/plan-completeness), three as the final pre-implementation checkpoint
> (actuation/RL safety, API/frontend demo flow, runtime environment/CI), three as the last
> gap-hunt (plan-integrity audit, security/penetration for the LAN threat model,
> async-concurrency race hunt).
>
> **Raw reports persisted:** async/concurrency at
> `.claude/plans/also-make-sure-the-gleaming-sparkle-agent-a32a8380a1bf6fd51.md`, security at
> `.claude/plans/also-make-sure-the-gleaming-sparkle-agent-ae10ecc33b4a7e034.md`. The other ten
> are consolidated here and in the plan file. This ledger exists so the final gate
> (plan Verification item 8) is executable without the originating conversation.
>
> **Status vocabulary:** **WAVE 1** = code agents applying now (Runs 0.1–0.10, 1.1–1.6) or this
> documentation wave (Runs 0.0/0.4/0.5). **BENCH-DAY** = requires the physical rig. **HUMAN/SUDO**
> = operator action no agent may take.

---

## 1. Verified aligned — stands as built (no change required)

| Item | Verified state |
| :-- | :-- |
| **Firmware ↔ rig safety contract** | **12/12 clauses confirmed**: pins, polarity, constants, boot order (open-at-boot via pre-Wi-Fi `setRelay(false)` + 100 kΩ pull-down), lockout, heartbeat fail-passive, payload caps, exact-case commands. |
| **MQTT topic symmetry** | Firmware `snprintf` topics ↔ pipeline subscriptions match end-to-end (power/telemetry/command/status/ack). |
| **DEVICE_ID chain** | `secrets.h` `EMS_DEVICE_ID` → firmware → `config.hardware.yaml` `devices.node_bench_agg` — one consistent chain; mismatch = silently ignored readings (documented). |
| **250 W / 1.25 ladder** | `RATED_WATTS 250` × `CRITICAL_PCT 1.25` → trip 312.5 W; config `max_aggregate_wattage`/`critical_pct` agree; coordination ladder monotonic (relay → 5 A fuse → 10 A parts → 13 A wire). |
| **Safety ladder at the 400 W candidate (WS-A)** | Trip 500 W = 2.17 A @ 230 V / 2.78 A @ 180 V; fuse margin 1.8–2.3×; all four loads = 474 W → WARNING band only, no trip; +100 W lamp → 574 W trips overcurrent; arc-fault not involved at 625 W/s. The 500 W candidate's fuse margin is corrected in C10 below. |
| Secrets hygiene | `.pio/`, `secrets.h`, `.env`, `frontend/dist` untracked; `#error` guard on missing secrets.h. |
| Wiring-doc pin table | `WIRING_STEP_BY_STEP.md` §2 matches `RELAY_PIN 18` / `PZEM_RX_PIN 16` / `PZEM_TX_PIN 17`. |
| **The retraining question (user's explicit ask)** | **Retraining is NOT the path** for phone/laptop/projector. For enrolled classes the encoder is bypassed — the decision is PZEM watts + enrolled envelope band, confidence 1.0 by construction on single-band survival. The three bands (≈45/120/300 W) are disjoint by 2.4–2.6×. Retraining cannot fix the real constraint (zero USB-PD-era phone data exists in any dataset; UK-DALE laptop = a 21–92 W netbook). A contingency protocol exists and is built **only if** the Run-2 enrollment gate fails. |

## 2. Findings ledger

### 2a. First consolidated round — C1–C10 (all measured; file:line in the plan)

| # | Finding (abridged) | Consequence |
| :-: | :-- | :-- |
| **C1** | Classifier sees **97–99 % PRE-event data** at a plug-in (`aggregate_nilm.py:136-158`: last 128 ending at detection; measured: 60 W laptop running + phone plugs in → "bulb" @ conf 1.000, 10/10). | "Sequential one-at-a-time" is only proven for **idle-socket** plug-ins. On a loaded socket the current code names the OLD load. Overlap (WS-D) is the fix for a broken path, not a layer above a proven one. |
| **C2** | Demo/e2e evidence **bypasses the detector** (`test_e2e_five_class_recognition.py:152-168` feeds steady windows straight to `_classify_device`; simulator noise σ≥15 W re-fires the detector every 5 s, masking the skew — measured 1/10 correct at t=0 for phone). This is the **train/serve window skew**: what the e2e proves (gate plumbing) is not what the demo claims (physical recognition). | "549 green" does not cover the transient path. A detector-path e2e routing plug-ins through `push()` is required for any physical claim. |
| **C3** | **Arc-fault kills wrong-ordered overlap**: firmware dt = loop period ≈ 0.134–0.16 s (PZEM 200 ms cache), so projector step (300 W) → ROC ≈ 1875–2239 W/s > 1000, **unsuppressed whenever baseline ≥ 50 W** → relay opens + 5-min lockout. Laptop (+120 W → 750–896 W/s) and phone (+336 W/s max) never trip on hardware. Cold projector start is suppressed. | Overlap demo viable **projector-first only** (cold → laptop → phone → bulb). "All ordered pairs" gate physically unachievable for (projector\|laptop) as built. **Do NOT tune `EDGE_ROC_THRESHOLD`/suppression** — safety decision; sequence the demo instead. |
| **C4** | **Twin ROC divergence**: twin `sim_dt=0.1` + instantaneous PZEM computes 10·ΔW — trips on the 120 W laptop step where hardware does not. | Twin must model the ~0.134–0.16 s effective dt (ROC dt-rescale clamp, PZEM reads stay instantaneous). |
| **C5** | **LED bulb dead zone 5–20 W**: 9 W is above `baseline_threshold_watts=5.0` (`phantom_tracker.py:45-64`), below the 20 W transient threshold, and `track()` is only called from the classified branch (`run_pipeline.py:1205-1209`). The bulb is invisible, not phantom-tracked. | Fix: raise `baseline_threshold_watts` → ~15 W (config-read) + call `track()` in the no-transient branch after `run_pipeline.py:1030`. **Do NOT lower the 20 W transient threshold** (noise floor argument stands). Loads in (10, 15] W stay unattributed — accepted, documented. |
| **C6** | **Confidence is vacuous for enrolled classes**: single-survivor renormalization → exactly 1.0; the 0.45 gate is unreachable with ≤2 survivors. In-band unknowns confidently misnamed (measured: 45 W fan → "phone" @ 1.000; 110 W amp → "laptop" @ 1.000; 290 W heater → "projector" @ 1.000). | Success criteria drop "confidence ≥ 0.65"; use envelope-band correctness + zero-confident-wrong counts; add an in-band unknown to the rehearsal. |
| **C7** | **Enrollment math / state-scoped capture**: 8 windows @ 16 s stride = **1.88 effective independent windows** (87.5 % overlap, 4.27× variance inflation). Envelope = state fingerprint. Multi-state capture actively harmful (laptop 40+120 W enrollment nests the phone band → coin-flip; measured 44 W laptop-idle → "phone" @ 0.762). | Capture protocol: single-state per class, state-scoped — phone ≤55 W states, laptop at demo state ~110–130 W (enrolled hi <172 W to keep projector separation), projector any. Other states → label loop (self-healing merge). Non-overlapping 128 s stride ≈ 17 min/class if statistical claims are wanted. |
| **C8** | **Zone map with real variability**: laptop correctly banded only 25 % of its 30–200 W span; 37–55 W laptop draw confidently "phone"; phone+projector together (345 W) lands inside the projector band with 2.2 W margin. | Zero-confident-wrong validation must include a light-load laptop case; simultaneous-switch-on documented as sequencing-avoided. |
| **C9** | **Unplug events unclassifiable**: negative deltas match no positive band → unrecognized. | Decision: accept + document (aggregate drops, no event); test pins it. |
| **C10** | **Errata**: 500 W fuse margin is **1.44–1.84×** (not 1.6–1.8×); 500 W trip demo needs laptop-under-load + lamp ≈ **654 W**, not +100 W lamp (574 W < 625 W). | GOD_TIER §1a corrected (amendment X5). 400 W column stands as verified. |

First-round ledger remainder (all confirmed, all run-assigned): twin/HIL 250 W-cutoff divergence at 4 sites; twin missing status strings (`SERVER_TIMEOUT`/`ONLINE`/`OFFLINE`), command casing, lock-set core; broker collision; 40 stale doc refs; dead config keys.

### 2b. Wave-3/4/5 additions (final checkpoint + security + concurrency specialists)

| # | Finding | Consequence |
| :-: | :-- | :-- |
| **RL-1** | **RL can open the physical relay uncommanded (CRITICAL)**: `run_pipeline.py:281` constructs `TabularQLearningAgent()` without config — loads `config/config.yaml` regardless of `--config config.hardware.yaml`. `PolicyPromotionGate` self-promotes after ~50 known+confident ticks and is sticky; empty Q-table + epsilon 0.3 returns SHED ~10 %/call and the reward teaches SHED. Worst case: one uncommanded relay-open every ~5 min, indefinitely, mid-demo (firmware honors OFF unconditionally). | Fix: pass `config=self.config`; `devices.node_bench_agg.tier0: true` (NEVER_SHED blacklist, defense-in-depth at `run_pipeline.py:1302-1306`); `rl: {cooldown_seconds: 300.0}`; NEVER_SHED regression. |
| **SEC-1** | **MQTT ACL gap**: no ACL; ESP32 + pipeline + API share the single `ems_pipeline` user — any holder of the one password can publish `home/plug/node_bench_agg/command` (mains toggle) and spoof telemetry/UI events. | Fix: `acl_file` + dedicated `esp32` user (read command; write sensor/ack only) with the **full grant matrix** (a sketchy ACL breaks the stack — verified: without `ems_pipeline`'s `$SYS/#` read the docker healthcheck fails and blocks ems-pipeline AND ems-api; without its `home/ml/label` write the label loop dies). `ems_api` user: wire or delete, never zero-permission. |
| **CONC-1** | **`/api/submit-label` 200-lie**: publishes `home/ml/label` QoS 0, no retry, and on publish failure still returns `200 {"status":"ok"}` (`src/api/main.py:683-694`); dashboard shows "successfully enrolled" when nothing was enrolled. The lie is test-locked (`tests/test_api.py:133`). | Fix: HTTP 503 on publish failure (+ update the test). |
| **CONC-2** | **Non-atomic registry save**: `PrototypeRegistry.save()` writes straight onto the live file (`protonet.py:648-653`). Crash mid-write → corrupt `.pt` → silent heuristic-only mode, all future labels refused. | Fix: tmp + `os.replace`. Also: `enroll_demo_devices.py` default `--out` collides with the pipeline's live registry — change it. |
| **CONC-3** | `DatabaseSession.close()` awaits (not cancels) the flush task → ~10 s Ctrl-C stall. Cosmetic. | Fix: cancel before await (drain handler already exists). |
| **BROKER-1** | **Broker collision**: system mosquitto vs docker-compose vs `start_broker.py` (anonymous amqtt on 0.0.0.0:1883 — silent security downgrade); demo launcher proceeds brokerless when `mosquitto -c` dies silently; repo passwd rejects the compose-default password (rc=5) so the stack never goes healthy without real `MQTT_PASSWORD`. | Runbook Gate 0 pre-step (single owner) + esp32 user (this wave); `start_broker.py` auth-or-fail-fast + launcher fail-loud (code); untrack `mosquitto/config/passwd` (this wave). |
| **DOC-1** | **40 stale `main.cpp:<line>` refs** across docs/tests/scripts/config (line numbers drifted; several describe pre-fix behavior). | Doc files converted to symbol anchors in this wave (Run 0.5); code/config files deferred to wave 2 (see §4). |
| **ENV-1** | **Host pre-flight (live machine state)**: `mosquitto-clients` NOT installed (Gates 0/5/6/7 stop cold); user NOT in `dialout` (flash will fail on `/dev/ttyUSB0`); `/tmp/pio-venv` already gone (volatility has bitten — recreate at persistent `~/.pio-venv`, update 4 runbook refs); `slowapi` missing locally (rate limiting silently off) + `tqdm`; `secrets.h` contains placeholders (green build, WiFi never associates — real values at Gate 1b). | HUMAN/SUDO, before Gate 0 / any bench work. |
| **ENV-2** | **CI is RED on main**: `python-tests` job fails its 100 %-branch-coverage gate on `src/pipeline/safety` at HEAD (full suite passes; the gate is what fails). | Decision recorded in the plan: restore the missing branch coverage (recommended — otherwise every push shows red and masks real failures). New parity test runs in CI (precedent: `test_relay_safety_boot_brownout.py` already parses `main.cpp`). |
| **SEC-2** | `run_pipeline.py:1498-1501`: parallel safety-monitor task connects without credentials — dead against the auth broker (backend SAFETY_WARNING/CUTOFF broadcasts never fire). Edge Core-0 safety unaffected. | Fix with credentials. |
| **SEC-3** | Minor hardening batch: WS `/ws` Origin check (CORS doesn't cover WebSockets); `hmac.compare_digest` in `verify_api_key`; label charset pattern on both pydantic models; 5 `changeme_pipeline_password` default sites across 3 scripts; `simulate_esp32.py` defaults to nonexistent user `pipeline` (anonymous-broker-only). | Code fixes; procedural rule for the simulator (below). |
| **PROC-1** | **Procedural rule**: never run `backend/scripts/simulate_esp32.py` while the physical rig is live — it ACKs commands for ANY device id (clearing pipeline cooldowns with fake hardware ACKs) and injects simulated telemetry on the same topics. | Written into the runbook (this wave). |

## 3. Decide lists

### 3a. CHANGES — executed in Wave 1

| Plan item | What | Owner |
| :-- | :-- | :-- |
| Run 0.1 | `tests/test_hardware_alignment.py` parity test (parse main.cpp consts; twin/HIL bench-parity sites, config values, topic symmetry, status strings, command semantics, DEVICE_ID chain; telemetry HARD assertion (the xfail was never needed — wave 1 wired the subscription in the same change); `max_aggregate_wattage == RATED_WATTS`) | code agent |
| Run 0.2 | Re-anchor 4 sites to firmware 250: twin default `esp32_firmware_sim.py:56`; HIL `hil_hardware_test.py:163-166` tautology → parse RATED_WATTS; stress scenario-4 `oc_node` `real_world_physical_stress.py:190`; e2e stage 8 `test_firmware_and_ai_e2e.py:318-335` real assert | code agent |
| Run 0.3 | Twin behavioral parity: status strings, WARNING branch, 256-byte cap, exact-case commands, lockout expiry, lock-set via core-1 path (+ the 8 assertion sites), PF init 1.0, **ROC dt-rescale to [0.134, 0.16] s** (C4); re-anchor `test_security_penetration.py:399-400` casing tests | code agent |
| Run 0.4 | Broker collision: Gate 0 pre-step + esp32 user + calibrate_ct credentials + PROC-1 rule (**runbook edits, done**); `start_broker.py` loopback-only bind + loud banner (done); **launcher fail-loud + `mosquitto-host.conf` (done, wave-3 close-out)**; **untrack passwd + `passwd.example` (done, wave-3 close-out — re-staged after a stash-based isolation check lost the first staging)** | docs + code agents |
| Run 0.5 | Stale-ref truth-up → symbol anchors in doc files (**done this wave**); code/config refs deferred to wave 2 | docs (done) |
| Run 0.6 | Config cleanup: dead qos keys, duplicate laptop key, dead `weights_path`/`anchors_path` (delete + point at real artifact — resolves X2), default-profile classes vs registry, `node_fridge` default | code agent |
| Run 0.7 | Firmware hardening: core-0 overcurrent latch flag, ±inf guard, PZEM-loss watchdog (SENSOR_FAULT + fail-safe open) | **DEFERRED-BENCH** (2026-09-10 close-out: main.cpp is UNCHANGED — this is the safety-rated, human-reviewed firmware change; it requires the WS-A review path + bench re-verification of Stages 2/7, not an agent patch. The twin's ±inf guard is now STRICTER than firmware (isfinite vs isnan) — parity note in HARDWARE_ALIGNMENT_CONTRACT) |
| Run 0.8 | RL hardware opt-out (RL-1): config pass-through, `tier0: true`, `rl.cooldown_seconds: 300`, NEVER_SHED regression | code agent |
| Run 0.9 | Security batch (SEC-1/2/3): ACL + `esp32` user + grant matrix + secrets.h update; WS origin (+ 127.0.0.1 default origins, wave-3); `compare_digest`; charset; **default sweeps (done, wave-3 close-out: all 5 sites now env-only, no hardcoded fallback)**; safety-monitor credentials; `VITE_API_KEY` public-by-design documented | code agents |
| Run 0.10 | Concurrency (CONC-1/2/3): 503 on label publish failure (+ test), atomic registry save, DB close cancel, enroll `--out` default | code agent |
| Run 1.1 | Detector-path e2e through `NILMTransientDetector.push()` (quiet σ≈2 W); fleet-window e2e kept as gate-plumbing coverage only, renamed to say so | code agent |
| Run 1.2 | Phantom channel fix (C5): `baseline_threshold_watts` → ~15 W config-read + `track()` in the no-transient branch | code agent |
| Run 1.3 | Unplug semantics (C9): document + pinning test | code agent |
| Run 1.4 | Success-criteria rewrite (C6/C8) in plan docs + tests: band-correctness, zero-confident-wrong incl. light-laptop + in-band-unknown | docs (X6 applied this wave) + code agent |
| Run 1.5 | `EMS_API_KEY` launcher default in `demo_full_system.py` `env_extra` | code agent |
| Run 1.6 | Frontend one-liners: `confidence: data.confidence` in `App.jsx` DEVICE_STATUS; stop passing `pendingUnknowns` | code agent |
| WS-A (code half) | When the ceiling rises, `config.hardware.yaml` `system_safety` rises in the **same commit** (parity-test-enforced) | code agent |
| WS-B | 3-class physical alignment: config appliances, `capture_bench_windows.py` VALID_CLASSES, registry wiring, heuristic fallback rules, label split-brain, test migration, demo profile | code agent |
| Run 4 (frontend) | Demo-integrity pass: hide Schedule/Settings, real breaker status + cutoff line, illustrative-labeled EnergyChart, INR rate; telemetry subscription + parity-test flip | code agent (later wave) |

### 3b. BENCH-DAY (requires the physical rig — no agent may do these)

| Item | What |
| :-- | :-- |
| WS-A ceiling | Read the projector nameplate on flash day: ≤ 320 W → 400 W rating; > 320 W → 500 W. Safety-rated single change, human-reviewed, bench re-verified. |
| Run 2 | State-scoped capture (phone ≤55 W state; laptop ~110–130 W, never enroll >172 W; projector any) → `prototype_registry_bench.pt` → `registry_path`. Gate: band-correct on re-presentation, unknowns (out-of-band kettle/vacuum AND in-band 45 W fan) rejected or label-looped, zero confident-wrong. |
| Run 3 | Overlap validation on achievable pairs — (laptop\|projector), (phone\|projector), (phone\|laptop), (laptop\|phone); (projector\|laptop) EXPECTED to trip arc-fault; (projector\|phone) marginal — both excluded from the demo script, sequenced **projector-first**. Gate: ≥90 % correct added-load naming, zero confident-wrong, sequential idle-socket path untouched. Fail → sequential ships, documented. |
| Run 4 rehearsal | Projector-first sequencing; light-laptop + in-band-unknown cases; LED bulb via phantom channel; trip demos per corrected math (400 W: 3 loads + lamp = 574 W; 500 W: laptop-under-load + lamp ≈ 654 W). |
| Runbook Gates 5–12 | With the corrected trip recipes; arc-fault behavior recorded at Gate 8/9 exactly as measured (projector-onto-laptop trip is EXPECTED, not a failure). |
| Physical milestones | Open-at-boot, cutoff with WiFi/broker down, lockout honored, ladder monotonic at the new rating — recorded with evidence. |

### 3c. HUMAN/SUDO (operator actions)

| Item | What |
| :-- | :-- |
| ENV-1 pre-flight | `sudo apt-get install -y mosquitto-clients`; `sudo usermod -aG dialout $USER` + re-login; recreate PlatformIO venv at persistent `~/.pio-venv`; `.venv/bin/pip install -r requirements.txt` (slowapi). |
| Gate 0 broker | `mosquitto_passwd` users (`ems_pipeline`, `esp32`), ACL deployment on the host broker, docker-compose down / single-owner verification. |
| Secrets | Real `MQTT_PASSWORD` / `EMS_API_KEY` at deployment (note: `changeme_pipeline_password` default also lives in scripts — swept in Run 0.9); real `secrets.h` values at Gate 1b. |
| CI decision | Restore branch coverage (recommended) or consciously accept red — recorded as ENV-2. |
| Accepted-by-design | `VITE_API_KEY` is public to the LAN by design (gate against accidents, not adversaries) — note stands in deployment docs. |

### 3d. NO-CHANGE (verified safe — do not "fix")

- **No dashboard route can toggle the relay** (correct for the rig); **no backend OFF publisher is live** (`run_pipeline.py:550` branch is dead code — backend is alert-only by architecture); **no auto-ON exists anywhere** (relay boots OFF, only a human ON energizes); no LOCKOUT_NACK spam; command retries bounded; heartbeat loss fail-passive server-side; `dist/` unused in the non-docker demo; CORS fine same-machine (LAN viewing = runbook items: `npm run dev -- --host` + `CORS_ORIGINS`).
- Firmware: exact topic match, bounded `snprintf` topic buffers (worst 53/64), payload cap 256, exact-case ON/OFF/WARNING, lockout honored. PZEM UART CRC-validated by lib + NaN skip in firmware.
- API: pydantic segment validators have no NaN/Inf bypass (empirically probed); export-csv has no path params; WS client messages broadcast-only; SQL parameterized. Frontend: no XSS sinks (React escaping).
- Concurrency: enrollment vs classification serialized structurally (single MQTT task + synchronous `handle_label_submitted`); no thread-vs-loop race in the API; per-device dict single-writer; firmware callback re-entrancy safe.
- **Do NOT tune** `EDGE_ROC_THRESHOLD`/suppression to make the (projector\|laptop) pair pass (C3). **Do NOT lower** the 20 W transient threshold (C5). **Do NOT retrain** as the primary path (§1).
- Unplug non-events: accept + document (C9).

## 4. Deferred to wave 2 (code/config files — not owned by the docs agent)

Stale `main.cpp:<line>` refs in files owned by code agents this wave (line counts
measured 2026-09-10; the plan's Run 0.5 flags the specific stale subset):
- `src/hardware/esp32_firmware_sim.py` comments — 23 ref lines (plan flags :60, :116, :138, :172; several cite `main.cpp:85` for `RELAY_ACTIVE_LOW`, now :96).
- `tests/test_relay_safety_boot_brownout.py` — 19 ref lines (plan flags :82, :101, :126, :152).
- `scripts/capture_bench_windows.py:16` (`main.cpp:441-446`).
- `scripts/real_world_physical_stress.py:146` (`main.cpp:207-217`; 3 ref lines total).
- `config/config.hardware.yaml:106` comment (`run_pipeline.py:165` → :197).
- Historical session logs `DEBUG_SESSION_2026-08-25.md` / `SESSION_2026-09-08.md` deliberately untouched (evidence, not truth).

## 5. Final-gate checklist (plan Verification item 8)

1. `pytest tests/test_hardware_alignment.py` green through the WS-A ceiling change.
2. RL NEVER_SHED regression: `node_bench_agg` (tier0) SHED-blocked at the publish site under `config.hardware.yaml`.
3. Phantom e2e: 9 W steady load with no transient appears in the PhantomTracker panel.
4. Detector-path e2e green; fleet-window e2e renamed as gate-plumbing coverage.
5. Full suite ≥549 green; re-anchored modules targeted first; CI green (coverage gate restored).
6. Run-2 gate on physical windows; Run-3 gate on achievable pairs, projector-first.
7. Bench: runbook Gates 5–12 with corrected trip recipes; arc-fault recorded as measured; pre-flight done before Gate 0.
8. This ledger re-checked end-to-end: every claim verified live or marked unverified.


---

## Wave-3 close-out (2026-09-10, cross-review + fix pass)

Adversarial cross-review (two reviewer specialists, empirical traces) confirmed
**no safety assertion was weakened anywhere** and closed the following:

- **Delta-layer robustness (cross-review RISks, fixed + traced):** variance-gated
  resolution (a slow SMPS soft-start no longer resolves mid-ramp with a partial
  delta — 12 W/s ramp now resolves to the correct full delta) and baseline
  handoff (a second plug-in ≥ the stability window after a resolve is measured
  against the NEW level — the 6 s-spacing trace now names the added load).
  Unplug/step-down deltas do NOT hand off (the re-fire must keep re-arming or
  the stale-window verdict returns — the C9 bug resurrected; pinned by tests).
- **SERVER_TIMEOUT parity:** the twin now arms its heartbeat at the ONLINE
  announcement (the connect equivalent) and republishes every 30 s while
  starved — matching main.cpp exactly. Previously the twin armed on first
  command and fired once (HIL evidence on that channel would not have
  transferred).
- **Breaker status:** time-windowed to the 5-min lockout — a stale cutoff no
  longer claims TRIPPED forever after recovery (new expiry test).
- **Telemetry lifecycle:** finiteness guard (json NaN never reaches WS),
  eviction-indexed (no unbounded growth).
- **Test honesty:** the vacuous tiny-dt test now asserts; `assert True`
  replaced with real invariants; the +70 W fan test moved to +75 W (the old
  one passed only via the re-fire correction and sat in the bulb∩fan
  envelope-overlap ambiguity zone — documented).
- **Bench-parity hardcodes:** e2e stage 8 now parses RATED_WATTS from
  main.cpp (the last hardcoded site).
- **Docs truth:** debug_status/claude_debug CLAUDE.md counts and twin
  description, runbook telemetry note, contract G1, root CLAUDE.md "FIVE"
  leftover, .gitignore now allows committing GOD_TIER_PLAN_2026-09-10.md.
- **WS-B physical-scope alignment (closed in the final pass, post-review):**
  `capture_bench_windows.py` VALID_CLASSES gains `phone` (the capture->enroll
  chain can now produce the physical 3-class registry); `config.hardware.yaml`
  `appliances:` -> [phone, laptop, projector] (bulb deliberately absent —
  phantom-tracked); heuristic `phone` band rule added (degraded mode: safe
  low-confidence -> unknown until Run 2 fits a real phone centroid — the
  centroid path and the no-USB-PD-data note are documented in the rule).
- **Deferred (human/bench):** Run 0.7 firmware hardening (above); WS-B.5 v1
  label-endpoint MQTT publish + EMS_CONFIG (off the demo path — the dashboard
  /api/submit-label loop works end-to-end); the WS-A ceiling decision
  (projector nameplate day); Run 2 physical capture; sudo host pre-flight
  (mosquitto-clients, dialout group, persistent pio venv).
