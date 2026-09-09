# God-Tier Plan — Physical Recognition: phone / laptop / projector (+ LED bulb as phantom load)

> **Date:** 2026-09-10 · **Status:** PLAN ONLY — no code changed
> **Scope lock (user instruction, supersedes the 2026-09-08 5-class lock):**
> physical classification classes are **phone, laptop, projector** — fan is
> dropped, and the bulb is **LED** (user decision, 2026-09-10), which sits
> below the 20 W detection threshold and is therefore **phantom-tracked, never
> classified**. The `bulb` class stays in the simulator demo profile only.
> **Overlap NILM is a core goal** (user decision): build it first, validate on
> real data, fall back to sequential if it fails the real-world gate.
> Built on the full readiness audit of 2026-09-10 (549/549 tests verified live;
> demo profile proven; hardware profile blocked by 6 items — all catalogued below).

> **⚠ AMENDED 2026-09-10 (X1–X7, per the twelve-specialist verification):** the approved
> execution plan ("sparkle" Runs 0–4) **supersedes §3's run sequencing**. **WS-C and WS-D are
> superseded by Runs 2–3** (state-scoped capture per C7; projector-first sequencing per C3;
> achievable-pairs overlap gate). WS-A/WS-B/WS-E remain live workstreams, scheduled per the new
> Runs. Evidence base: [`VERIFICATION_LEDGER_2026-09-10.md`](./VERIFICATION_LEDGER_2026-09-10.md);
> hardware-binding matrix: [`HARDWARE_ALIGNMENT_CONTRACT.md`](./HARDWARE_ALIGNMENT_CONTRACT.md).

---

## 0. The reframe: "completely change" is NOT what the evidence says

The audit proved the recognition core **works — for idle-socket plug-ins and gate plumbing
only** (amended X5 per verification C1/C2): ProtoNet + registry + envelope gate + label loop
recognized all 5 simulator classes 5/5 at confidence 1.0 on held-out *steady* windows fed
directly to the classifier, with 549 passing tests behind it. On the physical path the
classifier sees **97–99 % PRE-event data** — a loaded socket names the OLD load at confidence
1.0 until Run 3's delta-window fix lands. A rewrite would still destroy a verified system to
rebuild the same one; the defect is the window, not the model.

What actually stands between here and "projector + phone + laptop classify on real hardware
(LED bulb phantom-tracked)" is:

| # | Gap | Size |
| :-: | :-- | :-- |
| 1 | Hardware profile still 2-class, no registry | small (config) |
| 2 | Capture tool rejects one name (`phone`) of the new physical set | one line + names |
| 3 | No physical enrollment has ever run | **the real work** (bench) |
| 4 | 250 W rig cannot carry a ~300 W projector | **spec + firmware decision** |
| 5 | Overlap (simultaneous loads) misclassifies (M-2) — and per C1 the *sequential* path itself is only proven for idle-socket plug-ins | **core goal, gated, sequential fallback** |
| 6 | Robustness/hygiene debt (PZEM watchdog, label split-brain, telemetry, git) | bounded |

So: **no architecture change, no new ML models, no framework rewrites.**
Scope alignment + one safety-rated ceiling decision + physical data + hardening.

---

## 1. Two physical realities that shape everything

### 1a. The projector decides the relay ceiling (DECISION D-CEIL)

Current: `RATED_WATTS = 250` → trip 312.5 W. A 300 W projector alone sits in
the WARNING band; projector + anything **trips the relay**. If the projector is
physically in scope, the ceiling must rise. Coordination ladder re-check at the
two candidate ratings (same BOM — PZEM 10 A, SRD relay 10 A, 5 A fuse, 6 A
socket, 1.0 mm² wire — nothing to re-buy):

| | **RATED 400 W** (projector ≤ 320 W) | **RATED 500 W** (projector > 320 W) |
| :-- | :-- | :-- |
| WARNING (110%) | 440 W | 550 W |
| **CRITICAL trip (125%)** | **500 W** (2.17 A @ 230 V; 2.78 A @ 180 V) | **625 W** (2.72 A; 3.47 A) |
| Fuse margin | 5 A fuse = 1.8–2.3× above trip ✅ | 1.44–1.84× ✅ *(corrected X5/C10)* |
| projector+laptop (420 W) | clean, no warning | clean |
| +LED bulb (~474 W, all four) | WARNING band only — **no trip** | clean |
| Trip demonstrable? | +100 W lamp → ~574 W ✅ | laptop-under-load + lamp ≈ 654 W ✅ *(corrected X5/C10)* |

**Recommendation: 400 W** for a ≤ 320 W projector. With the LED bulb (9 W, not
60 W incandescent), **all four devices run simultaneously at ~474 W — under the
500 W trip**, so the full-stack overlap demo fits. This is a **safety-rated
spec amendment** (HARDWARE_FINAL_SPEC D8 + S1 + firmware constant
`main.cpp:74` + `config.hardware.yaml`), executed as a reviewed single change,
never a casual edit. Fan's removal makes 400 W comfortably sufficient.
**Decision rule on flash day: read the projector nameplate. ≤ 320 W → 400 W
rating. > 320 W → 500 W rating.** (A 350 W projector + 120 + 9 + 45 = 524 W
would trip a 400 W rating.)

### 1b. Bulb = LED → phantom-tracked, never classified (DECIDED 2026-09-10)

`TRANSIENT_THRESHOLD_W = 20.0` (`src/pipeline/aggregate_nilm.py:27`). An LED
bulb (5–12 W) never crosses it → **PhantomTracker standby accounting only** —
the same channel that already handles trickle phone chargers. This is physics,
not a tuning problem: lowering the threshold to ~9 W would sit inside 1 Hz
sensor noise and fire the transient detector on phantom fluctuations. The
accepted consequence:

- **Physical classification scope = 3 classes: phone, laptop, projector.**
- The LED bulb is a *dashboard story* (standby draw tracked and totalled),
  not a recognition story. `bulb` stays enrolled in the simulator demo profile.
- The 100 W incandescent lamp stays in the BOM regardless — it is the
  calibration reference and the trip ballast.

---

## 2. Workstreams

### WS-A — Ceiling amendment (safety-rated, single-threaded, human-reviewed)
1. Spec amendment: S1 scope → **3 classified classes (phone, laptop, projector) + the LED
   bulb as phantom-tracked load** (X1) + new load set; D8 ladder table at the
   chosen rating; B-9 note updated (trip still reachable).
2. `RATED_WATTS` in `main.cpp` (one constant), `max_aggregate_wattage`,
   `warning_pct/critical_pct`, `device_wattage_limits` in `config.hardware.yaml`.
3. **Fix the twin divergence found in the audit**: HIL scripts run
   `rated_watts=200` while firmware runs 250 — align twin/HIL to the new rating
   so HIL proves the shipped constant.
4. Regression: relay-safety + brownout + HIL suites; re-run Stage 2/7 gates on
   the bench (open-at-boot, cutoff with WiFi down).

### WS-B — 3-class physical alignment (the audit's 6 blockers, minus the ceiling)
1. `config/config.hardware.yaml`: `appliances: [phone, laptop, projector]`
   (`bulb` deliberately absent — LED is phantom-tracked, and the config comment
   must say so).
2. `scripts/capture_bench_windows.py`: `VALID_CLASSES` gains `phone`
   (physical set: phone, laptop, projector; `bulb`/`fan` retire to the
   simulator profile).
3. Registry wiring: after physical capture, `registry_path` →
   `prototype_registry_bench.pt`; the UK-DALE zero-envelope fallback (which
   produced the measured bulb→`laptop`@0.916 confident-wrong) is then dead.
4. Heuristic fallback: rules/centroids for `phone` (degraded mode must still
   name all 3 physical classes; `bulb` rules stay demo-only).
5. Label split-brain: `/api/v1/appliances/label-unrecognized` also publishes
   `home/ml/label`; set `EMS_CONFIG` wherever the API runs; one registry file
   per deployment.
6. Test migration: fan fixtures out of `test_ml_pipeline_recognition.py`,
   `test_e2e_five_class_recognition.py` (→ four-class), `test_api_extended.py`.
   **7. Demo profile**: `config.demo.yaml` drops fan; simulator drops `node_fan`
   (or keeps it as an OOD reject test — decide at execution).

### WS-C — Physical enrollment (the actual "make it real" work — human at the bench)
> *(Superseded by **Run 2** — state-scoped, single-state capture per class per verification C7;
> see the amendment banner. Kept for context.)*
Per class, one load at a time on an idle socket, per BRINGUP_RUNBOOK Gates 11–12:
`capture_bench_windows.py --class X --windows 8 --attest-physical …` →
`enroll_demo_devices.py --capture bench.npz --out prototype_registry_bench.pt`
→ verify each class on re-presentation + unknown-load rejection (kettle/vacuum).
~4 min per class capture; full afternoon with gates. **No agent can do this.**

### WS-D — Overlap NILM (CORE GOAL — build first, gate on real data, sequential fallback)
> *(Superseded by **Run 3** — correctly framed per C1/C3: the delta window is the fix for the
> loaded-socket defect, not a layer above a proven path; validation is on achievable ordered
> pairs with projector-first demo sequencing. See the amendment banner. Kept for context.)*

User decision 2026-09-10: overlap is the target; sequential is the fallback
**only if** overlap fails real-world validation. Current state: with one load
running, classification sees the pre-event aggregate and can be confidently
wrong (measured: 65+120 W → `tv` @ 1.000). `OverlapAwareNILMDetector` exists
but is dead code (M-2).

**Why this scope makes it tractable:** with fan dropped and the LED bulb below
threshold, the delta classes are just **{45, 120, 300} W** — well-separated,
no adjacent-band ambiguity. The build:

1. Wire the dormant `OverlapAwareNILMDetector` (needs `overlap_window_s > 5.0`
   to escape the base detector's 5 s cooldown) or equivalently classify the
   **post-step delta window** (steady-after − steady-before) instead of the
   pre-event aggregate.
2. Envelope-gate the delta (phone 37–55, laptop 98–141, projector 252–347 W
   padded bands from the bench enrollment).
3. Known hard cases to test explicitly: variable laptop draw (30–200 W by
   load state — the envelope band absorbs it), SMPS soft-start smearing the
   step over several samples, two loads switched on together (out of scope —
   sequence the demo), LED bulb delta (+9 W → correctly routed to
   PhantomTracker, NOT a classification event — this is a feature).

**Real-world validation gate (built into Run 4, on captured bench sequences):**
overlap must name the added load correctly in ≥ 90% of sequential plug-in
events across all ordered pairs of {phone, laptop, projector}, with **zero
confident-wrong** in the rehearsal script, and must never destabilize the
proven sequential path (it runs as a layer above it, feature-flagged). **Fail
the gate → ship sequential, document overlap as a known limitation.** No
timeline pressure on the fallback decision — it is made on evidence.

### WS-E — Robustness hardening (real-world survival)
1. **PZEM-loss watchdog** (firmware): N consecutive failed reads → publish
   SENSOR_FAULT, and after a longer window open the relay (fail-safe) — closes
   the audit's "blind but energized" gap. Safety-rated → same review path as WS-A.
2. Backend subscribes `home/sensor/+/telemetry` → V/I/PF on the dashboard.
3. Docker: `--config` per profile or `EMS_CONFIG` everywhere; mount
   `weights_demo/`; registry volume for persistence.
4. Broker hygiene: untrack `mosquitto/config/passwd`; LAN runbook (static IP or
   mDNS — hotspot DHCP churn breaks the ESP32 each session).
5. Host C test: delete or rewrite against real firmware logic (it currently
   tests its own mocks with stale expectations).
6. Repo truth-up: `REAL_WORLD_TESTING_PLAN.md` (30 A/CT era), runbook/capture
   old 4-class lists, stale `frontend/dist` rebuild, merge or drop the
   `worktree-ml-hardware-spec-fix` commit, remove `new 1/`, dead config keys
   (`open_set_threshold` — **its removal must also update the test pinning `== 0.65` at
   `tests/test_e2e_five_class_recognition.py:131`**, X3; phantom `weights_path`/`anchors_path`
   — delete the dead keys and point at the real artifact, one disposition, X2).

---

## 3. The agent fleet (deployed at execution, not now)

**Operational constraints found today:** subagent model overrides fail on this
token (Opus quota 402, Sonnet/Haiku 403) — the fleet runs on the default model
that works. Each Workflow run stays ≤ 15 agents (default guideline; raise via
/config if wanted). Total program ≈ 5 runs ≈ 40–60 agent-runs — "tens of
agents" as requested. **Safety-rated changes (WS-A, WS-E.1) are never
fan-out patched** — one agent, human review, bench re-verification.

The fleet reuses the repo's own skill personas (`.claude/skills/`):
`embedded-safety`, `ml-debugging`, `contract-integrity`, `verification`,
`test-triage`, `debugging`, `performance`, `codebase-navigation`.

| Fleet | Runs in | Job | Verify gate |
| :-- | :-- | :-- | :-- |
| **Fixing agents** (personalized) | Run 1–3 | One bounded defect each from the WS-B/E ledger; FAST FIX contract (reproduce → root cause → minimal patch → regression) | adversarial verify agent per fix, then suite |
| **Math agents** | Run 1 (parallel) | Ladder verification at new ceiling; phone/bulb band-overlap quantification (padded bands overlap 50–55 W); arc-fault loop-period model (Gate 8 predictor); 1 Hz × 128 s window sufficiency | independent second agent re-derives, must agree |
| **ML agents** | Run 4 (post-capture) | Enroll from bench.npz, validate envelopes, held-out confusion matrix per class, threshold calibration, unknown-reject audit | 3/3 classes ≥ 95% band-correct held-out, zero confident-wrong |
| **IoT agents** | Run 2 | MQTT topic/QoS/auth contract sweep; twin↔firmware parity pins; docker profile wiring; secrets/broker runbook automation | contract-integrity skill checklist |
| **Logging agents** | every run | Consolidate each run into a `claude_debug/` ledger (evidence, not claims); structured safety-events review | ledger reviewed by watchover |
| **Watchover agents** | every run end + final | Re-run full regression + e2e + HIL after each merge; final re-run of this audit's checklist as the completion gate; doc-vs-code drift report | 549+ green, all gates recorded |

**Run sequencing:** Run 1 = math + WS-B config/tooling fixes (no firmware) →
Run 2 = IoT + integration + hygiene → **[bench: gates 5–10 + WS-A on the rig]**
→ Run 3 = test migration + hardening → **[bench: WS-C capture]** → Run 4 = ML
enrollment + calibration → **[bench: recognition dry-run]** → Run 5 = watchover
final audit → demo rehearsal. Physical milestones interleave; agents never
touch mains.

---

## 4. Success criteria — "500%" defined measurably

1. Full suite green (≥ 549) after **every** workstream merge.
2. New 3-class e2e on **physical** windows (amended X6 per C6/C8 — "confidence ≥ 0.65" is
   dropped: it is vacuous for enrolled classes, single-survivor renormalization yields exactly
   1.0): phone / laptop / projector **band-correct** on ≥ 95 % of held-out physical windows,
   **zero confident-wrong** across the demo script — including a light-load laptop (37–55 W)
   case and an in-band unknown (e.g. 45 W fan) case; LED bulb correctly appears as phantom
   load, never a classification.
3. **Overlap gate (if it ships)**: ≥ 90% correct naming of the added load
   across ordered pairs, zero confident-wrong in rehearsal, sequential path
   untouched underneath. Gate failed → sequential ships, limitation documented.
4. Unknown loads (kettle, vacuum, monitor) → `UNRECOGNISED`, routed to the
   label loop, no envelope hijack.
5. Safety physically demonstrated: open-at-boot, cutoff with WiFi/broker down,
   lockout honored, ladder monotonic at the new rating — all recorded with
   evidence in the runbook format.
6. Every runbook gate PASS or explicitly NOT-RUN-with-reason. No gate marked
   passed without evidence.

## 5. Execution-day preliminaries
Update the scope lock in root `CLAUDE.md` (fan out; **3 physical classes — phone, laptop,
projector — with the LED bulb phantom-tracked**, X1), log the decision
in a session doc, then WS-A spec amendment first (everything else hangs off the
ceiling number).
