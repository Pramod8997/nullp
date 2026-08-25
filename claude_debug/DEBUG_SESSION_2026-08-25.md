# Debug Session Log — ML & Hardware Integration

> **Date:** 2026-08-25
> **Scope requested:** debug the ML part and the hardware-integration part
> **Operating rules used:** `claude_debug/claude/claude_opus_engineering_upgrade/.claude/`
> (skills: `debugging`, `embedded-safety`, `ml-debugging`, `test-triage`, `verification`)
> **Branch:** `main` — all changes are **uncommitted working-tree edits**
> **Result:** 6 defects found and fixed, 8 regression tests added, 1 stress scenario re-specified,
> 6 open items logged. Baseline moved **467 → 474 tests passing**.

---

## 1. Starting point — verified live, not trusted from docs

Per the upgrade playbook ("Do NOT blindly trust project status documents"), the claimed baseline
was re-run before any edit:

| Command | Claimed | Actually observed |
|---|---|---|
| `python -m pytest tests/ -q` | 467/467 | ✅ **467 passed** in 17.00s |
| `python scripts/hil_hardware_test.py` | 10/10 | ✅ 10/10 |
| `python scripts/real_world_physical_stress.py` | 7/7 | ✅ 7/7 |
| `python scripts/test_firmware_and_ai_e2e.py` | 8/8 | ✅ 8/8 |
| `python scripts/stress_test_hardware_sim.py` | 7/7 | ✅ 7/7 |

**The green baseline was real.** The defects below all existed *underneath* it — the suite was
green over six genuine bugs, two of which had tests named after them that asserted nothing.

---

## 2. Defects FIXED

### Hardware integration

#### H-1 — Twin gated overcurrent on inrush suppression; firmware does not
* **File:** `src/hardware/esp32_firmware_sim.py`, `core0_safety_step()`
* **Was:** `if power_w > critical_watts and not is_normal_inrush:`
* **Firmware truth:** `firmware/esp32_node/src/main.cpp:207-217` is **unconditional by design**,
  with an explicit multi-line comment citing `HARDWARE_FINAL_SPEC.md` D11′: inrush suppression
  exists *only* to stop a starting surge reading as an arc fault on the dP/dt channel.
* **Proof:** fresh node, `rated_watts=200` (ceiling 250 W), first sample `280 W` (140% overload)
  → twin left relay **CLOSED**. Real hardware opens on that first sample.
  The twin was **weaker than the shipped firmware** on a safety cutoff.
* **Fix:** gate removed. Overcurrent now trips on the first sample above 125% rated.

#### H-2 — NaN PZEM read poisoned safety state and reached the wire
* **File:** `src/hardware/esp32_firmware_sim.py`, `core0_safety_step()`
* **Firmware truth:** `main.cpp:167-171` — `if (isnan(...)) { vTaskDelay(100); continue; }`,
  i.e. the whole cycle is skipped, leaving `lastWatts`, the baseline ring and shared state alone.
* **Proof:** injecting one NaN gave `_last_watts=nan`, `_baseline_ring=[nan,0,0,0,0]`,
  `shared_power_watts=nan`, and Core 1 then published:
  * `home/sensor/<id>/power` → `'nan'`
  * `home/sensor/<id>/telemetry` → `{"v": 230.0, "i": 0.0, "w": NaN, "pf": 0.95}` — **bare `NaN`
    is not valid JSON**; a strict parser rejects it.
  A NaN in `_baseline_ring` also forces `is_normal_inrush` False for as long as it sits there,
  silently disabling inrush suppression.
* **Fix:** finite guard added mirroring the firmware — cycle skipped, no state touched.

#### H-3 — Relay polarity was entirely unmodelled (B-7 was undetectable)
* **File:** `src/hardware/esp32_firmware_sim.py`
* **Was:** `relay_active_low` was assigned in `__init__` and **never read anywhere**.
  `set_relay()` wrote only the logical state.
* **Proof:** constructing with `relay_active_low=True` vs `False` produced **zero** differing
  attributes and identical `gpio18_relay_state`. Polarity had no observable at all.
* **Compounding:** `tests/…::test_active_low_logic_correctness` asserted only
  `gpio18_relay_state`, so it passed under **either** polarity — and its comment described
  active-LOW semantics, contradicting the shipped `RELAY_ACTIVE_LOW = false`. Defect **B-7**
  (boot energises the load; every cutoff *closes* the relay) could not be caught by any test.
* **Also:** the twin's default was `True` — i.e. it modelled the exact inverted polarity the
  locked spec says must never be reintroduced.
* **Fix:**
  * added read-only `gpio18_level` property = electrical pin level, mirroring `setRelay()` at
    `main.cpp:137`. `gpio18_relay_state` keeps its logical meaning (81 call sites rely on it — deliberately unchanged).
  * default corrected `relay_active_low: True → False` to track the locked spec.
  * test rewritten to assert pin level under **both** polarities.

### ML / NILM

#### M-1 — One non-finite sample silently destroyed transient detection
* **File:** `src/pipeline/aggregate_nilm.py`, `NILMTransientDetector.push()`
* **Mechanism:** the value was buffered; `savgol_filter` spreads NaN/Inf across
  ±`sg_window//2` samples, and **every `np.abs(...) >= threshold` comparison against NaN is
  `False`** — so no detection fires.
* **Proof:** with a NaN adjacent to a real 2100 W step, the step was **never** detected across 30
  following samples. Corruption is **local (7 samples = sg_window)**, not permanent — detection
  recovers for a *later* separate step — but the coincident appliance event is **lost for good**,
  never classified, never billed. `±Inf` behaves identically.
* **Chains from H-2:** the twin published `'nan'`, so this was a live end-to-end
  hardware → ML failure path, not theoretical.
* **Fix:** non-finite samples rejected before buffering; buffer left unchanged.

#### M-3 — Overlap residual guard was a tautology
* **File:** `src/pipeline/aggregate_nilm.py`, `OverlapAwareNILMDetector.push()`
* **Mechanism:** the "only emit if the residual still has a meaningful transient" check ran on the
  **pre-clamp** residual, and `np.diff(x - c) == np.diff(x)` **exactly** — verified numerically.
  The guard filtered nothing; every registered baseline emitted a candidate.
* **Fix:** zero-clamp moved before the guard so it tests the array actually emitted.

#### M-4 — Placebo tests named after the real defect
* `test_nilm_nan_in_signal` and `test_nilm_inf_in_signal` both read `assert True`.
* All 6 `test_overlap_*` tests read `assert True`.
* Root cause: written against a **hallucinated `.process()` API** (forbidden per `CLAUDE.md` §4 —
  the real API is `push` / `get_current_segment` / `reset`), then `hasattr`-guarded into no-ops.
* **Fix:** all 8 rewritten against the real `push()` API with real assertions.

---

## 3. Stress scenario re-specified (deliberate, expected failure)

`scripts/real_world_physical_stress.py` → **scenario 4 "Inrush Current vs Arc-Fault
Discrimination"** drove a **1200 W** inrush into a **200 W-rated** node and asserted *no trip*.

That expectation held **only because of H-1**. `1200 W > 250 W` ceiling, so real hardware opens the
relay on the first sample. Fixing H-1 correctly broke this scenario — the test encoded a wrong
expectation, and per the `embedded-safety` skill the protection logic was **not** weakened to keep
it green.

Re-specified to isolate the channel actually under test:
* node rated **1200 W** (ceiling 1500 W) so the 12,000 W/s inrush stays *below* the overcurrent
  ceiling → the dP/dt channel is proven in isolation;
* arc fault `150 W → 1500 W` (13,500 W/s) with a warm 150 W baseline → suppression correctly no
  longer applies; asserts `shared_arc_fault`, not just relay state, so the **channel that fired**
  is verified;
* the unconditional overcurrent contract asserted separately (280 W on a 200 W line, cold baseline).

---

## 4. Verification performed

```bash
source .venv/bin/activate
python -m pytest tests/ -q                        # 474 passed  (was 467)
python scripts/real_world_physical_stress.py      # 7/7 PASS  (incl. re-specified scenario 4)
python scripts/hil_hardware_test.py               # 10/10 PASS
python scripts/test_firmware_and_ai_e2e.py        # 8/8 PASS
python scripts/stress_test_hardware_sim.py        # 7/7 PASS
graphify update .                                 # 2397 nodes, 4756 edges, 179 communities
```

### Before/after proof of every regression test
Source fixes were reverted (`git checkout -- src/…`), the new tests run against **unfixed** source,
then the fixes reapplied:

```
=== NEW TESTS AGAINST *UNFIXED* SOURCE ===
FAILED tests/test_ml_nilm_math_stress.py::test_nilm_nan_in_signal
FAILED tests/test_ml_nilm_math_stress.py::test_nilm_inf_in_signal
FAILED tests/test_relay_safety_boot_brownout.py::test_active_low_logic_correctness
FAILED tests/test_relay_safety_boot_brownout.py::test_default_polarity_matches_locked_firmware_constant
FAILED tests/test_relay_safety_boot_brownout.py::test_relay_de_energised_at_boot_under_both_polarities
FAILED tests/test_relay_safety_boot_brownout.py::test_overcurrent_is_unconditional_during_cold_baseline
FAILED tests/test_relay_safety_boot_brownout.py::test_nan_pzem_read_does_not_poison_safety_state
FAILED tests/test_relay_safety_boot_brownout.py::test_nan_never_published_to_mqtt
8 failed, 84 passed
=== after reapplying fixes ===
92 passed
```

All 8 fail before the fix and pass after. **Honest caveat:** two additions are *new coverage*,
not regressions, and pass on both sides of the fix —
* the 7 rewritten `test_overlap_*` tests (M-3 changed only guard *ordering*, which is not
  observable in the residual values themselves), and
* `test_inrush_suppression_still_protects_dpdt_channel`, which exists to catch
  **over**-correction of H-1 (i.e. that removing the overcurrent gate did not also remove
  suppression from the dP/dt channel).

### Tests added / rewritten
`tests/test_relay_safety_boot_brownout.py`
* `test_active_low_logic_correctness` — **rewritten**: asserts `gpio18_level` under both polarities
* `test_default_polarity_matches_locked_firmware_constant` — parses `main.cpp` and asserts the twin
  default tracks it (catches future drift automatically)
* `test_relay_de_energised_at_boot_under_both_polarities`
* `test_overcurrent_is_unconditional_during_cold_baseline`
* `test_inrush_suppression_still_protects_dpdt_channel`
* `test_nan_pzem_read_does_not_poison_safety_state`
* `test_nan_never_published_to_mqtt` — rejects bare `NaN` via `json.loads(parse_constant=…)`
* fixtures no longer pin `relay_active_low=True`

`tests/test_ml_nilm_math_stress.py`
* `test_nilm_nan_in_signal`, `test_nilm_inf_in_signal` — **rewritten** real
* all 6 `test_overlap_*` — **rewritten** real, plus new
  `test_overlap_window_shorter_than_cooldown_is_unreachable` pinning the M-2 constraint

---

## 5. Files changed (all uncommitted)

```
 claude_context/ARCHITECTURE_AND_APIS.md  |   6 +-   # relay_active_low default + gpio18_level
 claude_debug/ARCHITECTURE_AND_APIS.md    |   6 +-   # same
 claude_debug/HARDWARE_FINAL_SPEC.md      |   8 +-   # 4 stale "main.cpp:67" refs -> main.cpp:85
 claude_debug/debug_status.md             |  88 +-   # new §0 fix table, §5 open items, 467->474
 scripts/real_world_physical_stress.py    |  57 +-   # scenario 4 re-specified
 src/hardware/esp32_firmware_sim.py       |  46 +-   # H-1, H-2, H-3
 src/pipeline/aggregate_nilm.py           |  27 +-   # M-1, M-3
 tests/test_ml_nilm_math_stress.py        | 143 +-   # M-4 rewrites
 tests/test_relay_safety_boot_brownout.py | 135 +-   # H-3 + hardware regressions
 9 files changed, 455 insertions(+), 61 deletions(-)
```

Also fixed: `HARDWARE_FINAL_SPEC.md` cited `main.cpp:67` for `RELAY_ACTIVE_LOW` in 4 places
including the bring-up checklist, but the constant is at **`main.cpp:85`** — an operator following
the checklist would have inspected the wrong line.

---

## 6. OPEN ITEMS — not fixed, need a decision

Also recorded in `claude_debug/debug_status.md` §5.

1. **M-2 — the overlap/multi-label NILM feature is dead code.**
   `OverlapAwareNILMDetector` is **never instantiated in `src/` or `scripts/`** — only in
   `src/pipeline/__init__.py`'s export list and in tests. The pipeline
   (`scripts/run_pipeline.py:210,653`) uses plain `NILMTransientDetector`. Yet the architecture
   diagram in **both** `CLAUDE.md` files shows
   `NILM --> Overlap[OverlapAwareNILMDetector] --> ProtoNet` as a live stage.
   It is also **unreachable at its own defaults**: the base detector enforces a 5 s
   post-detection cooldown, so two detections can never fall inside the default
   `overlap_window_s = 3.0`. Verified: at defaults only `{'single'}` is ever emitted; at
   `overlap_window_s=8.0` you immediately get `{'multi_device','fridge','kettle','single'}`.
   **Decision:** wire it in with `overlap_window_s > 5.0`, or delete the class and fix the diagram.

2. **H-4 — lockout is latched on a different core than the firmware does it.**
   The twin sets `relay_locked` synchronously in Core 0 with the cutoff. The firmware sets
   `relayLocked`/`lockStartMs` in **Core 1** (`main.cpp:395-418`, and note `main.cpp:416` reads
   `if (powerWatts > criticalWatts && !relayLocked)`). Real hardware therefore has a window — up to
   one Core-1 loop period — where a cutoff has fired but the relay is **not yet locked**, so an
   `ON` command landing in that window would re-close onto an un-cleared fault. The twin cannot
   expose this race by construction. **This is a firmware change** (latch the lockout in Core 0
   under `sharedMux`), so it was flagged, not touched.

3. **15 placebo tests remain**, plus **39 calls to the hallucinated `.process()`** API,
   `hasattr`-guarded into no-ops:
   * `tests/test_chaos_engineering.py` — 14 bare `assert True`
   * `tests/test_ml_nilm_math_stress.py` — 12 remaining
   * `tests/test_hil_uart_corruption.py` — 1
   * `tests/test_watchdog.py::test_watchdog_zscore_magnitude` — **no assertions at all**
   * `tests/test_pipeline_stages.py` — 15 `.process()` calls
   Also `tests/test_ml_nilm_math_stress.py::test_nilm_buffer_memory_growth` asserts
   `len(getattr(detector,'buffer',[])) <= 10000` — the attribute is `_buffer`, so it asserts
   `len([]) <= 10000`.

4. **Import failures are masked into green runs.** `test_ml_nilm_math_stress.py`,
   `test_temperature_scaling.py`, `test_e2e.py` use
   `try: … except ImportError: X = MagicMock()`. A real import break passes as a MagicMock.

5. **`aiosqlite` teardown races the event loop.**
   `tests/test_pipeline_stages.py::test_stage9_rl_agent_produces_action` emits
   `PytestUnhandledThreadExceptionWarning: RuntimeError: Event loop is closed` from the aiosqlite
   worker thread — a connection outliving its loop. Harmless to the assertion, still an
   unawaited-teardown bug.

6. **Nothing here is physical validation.** Every result is simulation. The twin now matches
   `main.cpp` on the paths audited; no claim above substitutes for bench work per
   `REAL_WORLD_TESTING_PLAN.md`.

7. **Both `CLAUDE.md` playbooks are now stale in three places** (root and `claude_debug/`).
   Left unedited deliberately — these are the project's own governing playbooks, and two of the
   three corrections depend on the M-2 decision above:
   * `Current Health: 467/467` and "baseline is **467 passing tests**" → now **474**.
   * The mermaid diagram labels GPIO 18 as `Active-LOW Relay`, which **contradicts the locked
     `HARDWARE_FINAL_SPEC.md`** (`RELAY_ACTIVE_LOW = false`, active-HIGH). This is the same
     mislabelling that made defect B-7 look correct; a bring-up operator reading the diagram
     would wire the relay backwards.
   * `NILM --> Overlap --> ProtoNet` shows a stage that is not in the running pipeline (M-2).

---

## 7. Resuming in a new session

```bash
cd /home/pramodsb/Downloads/mjr && source .venv/bin/activate
git diff --stat                      # the 9 changed files above, still uncommitted
python -m pytest tests/ -q           # expect: 474 passed
```

Read `claude_debug/debug_status.md` §0 (what was fixed) and §5 (what is open).
Suggested next steps, highest value first:

1. Decide **M-2** — wire the overlap detector into `run_pipeline.py` or delete it and correct the
   architecture diagrams in both `CLAUDE.md` files.
2. Decide **H-4** — the Core-0 lockout latch in `firmware/esp32_node/src/main.cpp`. This is the
   only remaining item that is a *real hardware* safety gap rather than a simulation-fidelity gap.
3. Sweep the remaining 15 placebo tests and 39 `.process()` no-ops (item 3), and remove the
   `ImportError → MagicMock` masking (item 4).
4. Fix the aiosqlite teardown (item 5).
5. Refresh both `CLAUDE.md` playbooks (item 7) — the `Active-LOW Relay` diagram label is the
   dangerous one; fix it regardless of how M-2 is decided.
6. Commit. Nothing has been committed this session — `main` is clean of these edits.

**Method note for the next session:** the highest-yield technique here was diffing the Python
digital twin against `firmware/esp32_node/src/main.cpp` line by line. Three of the six defects
were twin-vs-firmware divergences invisible to the test suite, because the tests validate the
twin, and the twin was the thing that was wrong. `src/hardware/esp32_firmware_sim.py` claims in
its own docstring to be an "exact software model of main.cpp" — that claim is now true for the
paths audited (NaN rejection, overcurrent unconditionality, relay polarity), but the rest of the
file has not been line-diffed against the firmware.


**********************************************************************************************************************************************************************************************************
---

# CONTINUATION CHECKPOINT — 2026-08-25 API KEY SWITCH

## Purpose

This session is being handed off to a new Claude session because the previous session ran out of API credits/context.

The existing debugging work must be preserved.

Do NOT restart the investigation from scratch.

Do NOT repeat expensive diagnostics whose results are already documented in this file or in `ML_PIPELINE_FIX_2026-08-25.md`.

The actual repository is the source of truth for current code state.

---

## Current Repository State

The previous session reached:

- 473 tests passed
- 1 test failed
- Original baseline: 474 tests passed
- All current changes are uncommitted on `main`
- M-5, M-6, M-7 and M-8 fixes are implemented
- Label/enrollment loop has been verified end-to-end
- The remaining blocker is the centroid regression test

Failing test:

```text
tests/test_real_data_and_ml_fallback.py::TestHeuristicDiscrimination::test_centroid_path_used_when_fitted

● Compact summary
This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

Summary:

1. Primary Request and Intent

Original request (single user message after /clear, carried forward from the pre-compaction summary):

"continue with @claude_debug/DEBUG_SESSION_2026-08-25.md and fix ml pipline. It should work for phone, laptop, monitor if not recognized classify it as unrecognized device and i will label it."

Decomposed intent:

Resume the debug session in claude_debug/DEBUG_SESSION_2026-08-25.md (6 defects previously fixed, baseline 467→474 tests, all uncommitted on main).

Fix the ML pipeline so it classifies phone, laptop, monitor.

When a load is not recognized, report it as an unrecognized device rather than guessing — the user will supply the label, so the few-shot LABEL_REQUEST → enrollment → recognition loop must actually work.

Final request (mid-turn interrupt, this session):

"Stop everything now and log everything now i have to swich api keys"

This is a stop-and-checkpoint request. No further code changes should be made; this summary is the log.

2. Key Technical Concepts

ProtoNet few-shot metric learning; PrototypeRegistry; episodic training

OpenMax / Weibull EVT open-set recognition (Bendale & Boult CVPR 2016) — found dead and unfixable via embedding distance

Temperature-scaled softmax over negative squared prototype distances; confidence_gate(0.90)

Two-channel agreement as the recognition test (learned embedding channel confirmed by a deterministic absolute-watts channel) — the core architectural decision of this session

Physical power-envelope plausibility gating as a model-free reject channel

NILM; Savitzky-Golay filtering + derivative transient detection

UK-DALE / REDD real appliance datasets; leave-one-meter-out validation

Nearest-centroid in a 5-feature space (log_steady, log_peak, duty, log_overshoot, volatility)

Deployment profiles: config/config.yaml (household 3.5 kW), config/config.demo.yaml (consumer electronics 600 W, 7 appliances), config/config.hardware.yaml (physical rig 250 W, laptop + phone_charger)

Digital twin of ESP32 dual-core FreeRTOS firmware; MQTT transport; PZEM-004T

3. Files and Code Sections

scripts/run_pipeline.py — MODIFIED (the primary fix site)

The class is EMSOrchestrator, aliased FullPipeline = EMSOrchestrator at line 1519. Constructor signature: __init__(self, config: Optional[Dict] = None, stage_hook=None, rl_hook=None) — there is no config_path kwarg.

a) Imports + display constant:

from src.pipeline.heuristic_fallback import (
    HeuristicApplianceClassifier,
    plausible_classes,
    DEFAULT_RULES,
    ENVELOPE_SLACK,
    UNKNOWN as UNRECOGNISED,
)

# Wire/internal sentinel for "no known class fits this load". Kept as the string
# "unknown" deliberately: src/api/main.py, the dashboard's DeviceCards.jsx and 54
# test assertions all key on that literal, and renaming it would break the
# contract for no functional gain. UNRECOGNISED_DISPLAY is what a human should
# read — the operator is being told the system cannot name the device, not that
# the device is malfunctioning.
UNRECOGNISED_DISPLAY = "Unrecognised device"

b) __init__ additions (around former line 197-199):

# ── Heuristic Fallback (zero-torch, deterministic) ──
# `appliances:` scopes the classifier to what this deployment can
# actually present, so a class no socket on the rig can drive is never
# the answer. Previously the full rule set was always used, so the
# hardware profile (laptop + phone charger only) could report `hvac`.
self.heuristic_min_confidence = proto_cfg.get("heuristic_min_confidence", 0.55)
self.heuristic_clf = HeuristicApplianceClassifier(
    allowed_classes=self.config.get("appliances") or None
)
self._registry_clf_cache: Dict[tuple, HeuristicApplianceClassifier] = {}

# Raw power windows behind each device's recent unrecognised events.
# Enrollment needs watts; the DeltaStabilityAnalyzer only keeps
# embeddings, so the traces have to be retained separately.
self._unknown_windows: Dict[str, deque] = {}

# Bar a *confirmed* classification must clear to be reported as
# recognised. Deliberately separate from `confidence_threshold` (0.90),
# which gates a single softmax: this value gates the noisy-OR of two
# independent channels that have already agreed, so it is not the same
# quantity and must not borrow the same number. Set from measurement on
# real UK-DALE windows — see claude_debug/DEBUG_SESSION_2026-08-25.md.
self.recognition_threshold = proto_cfg.get("recognition_threshold", 0.45)

c) _classify_device — fully rewritten. Docstring records why (SupportSetManager is only populated from protonet.anchors_path, set only in config/config.yaml, so compute_prototypes() returned {} and every event on every profile returned ("unknown", 0.0, {}); the loaded 7-class registry was never read). Body:

# Use NILM-filtered segment when available (§2.1 fix)
if filtered_segment is not None:
    window_np = np.asarray(filtered_segment, dtype=np.float32)
else:
    if device_id not in self.power_windows:
        self.power_windows[device_id] = deque(maxlen=self.seq_len)
    self.power_windows[device_id].append(power_watts)
    window = self.power_windows[device_id]
    if len(window) < self.seq_len:
        return "pending", 0.0, {}
    window_np = np.array(list(window), dtype=np.float32)

if window_np.size < self.seq_len:
    window_np = np.pad(window_np, (0, self.seq_len - window_np.size))
window_np = window_np[:self.seq_len]

registry = getattr(self, "prototype_registry", None)

# ── Degraded mode: no encoder, or nothing enrolled to compare against ──
if self.encoder is None or registry is None or not registry.prototypes:
    if not self.support_manager.raw_windows:
        return self._classify_heuristic(window_np)
    try:
        return self.support_manager.classify(
            window_np, self.encoder, self.weibull, self.temp_scaler,
            self.confidence_threshold)
    except Exception as e:
        logger.error(f"Legacy ProtoNet classification error for {device_id}: {e}")
        return "error", 0.0, {}

try:
    _, _, dist_map = registry.classify(window_np)
    if not dist_map:
        return self._classify_heuristic(window_np)

    names = list(dist_map.keys())
    d2 = np.array([dist_map[n] for n in names], dtype=np.float64)
    if not np.all(np.isfinite(d2)):
        logger.warning(f"Non-finite prototype distance for {device_id} — treating as unrecognised")
        return UNRECOGNISED, 0.0, dist_map

    temperature = 1.0
    cal = getattr(self, "calibrated_scaler", None)
    if cal is not None:
        try:
            temperature = max(float(cal.temperature.item()), 0.05)
        except Exception:
            temperature = 1.0
    logits = -d2 / temperature
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum() + 1e-12

    order = np.argsort(-probs)
    best = names[order[0]]
    confidence = float(probs[order[0]])

    # ── Physical plausibility gate ──
    eligible = self._eligible_classes(window_np, names, registry)
    if not eligible:
        return UNRECOGNISED, 0.0, dist_map

    # ── The operator's own labels outrank the population prior ──
    # (long comment: without this the label loop silently does nothing —
    #  shipped `monitor` prototype sits almost on top of a newly enrolled
    #  35 W monitor, probability halves, result falls under threshold.
    #  Measured: enrolled-recall 1/3 -> 3/3.)
    enrolled_hits = {n for n in eligible if registry.power_envelope(n) is not None}
    if enrolled_hits:
        eligible = enrolled_hits

    # Always renormalise over the survivors, not only when the argmax
    # was vetoed: a confidence carried over from the full softmax would
    # still be diluted by classes the physical gate has just ruled out.
    keep = [i for i, n in enumerate(names) if n in eligible]
    sub = probs[keep] / (probs[keep].sum() + 1e-12)
    j = int(np.argmax(sub))
    best = names[keep[j]]
    confidence = float(sub[j])

    # ── Two-channel agreement is the recognition test ──
    # (long comment: measured fact — 65 W laptop called desktop_computer at
    #  p=0.72, 100 W charger called tv at p=0.86. Enrolled classes are
    #  confirmed by envelope containment; shipped classes need the
    #  deterministic centroid classifier to independently agree.
    #  Measured: acc|accepted 0.621 -> 0.851, coverage ~0.37.)
    if registry.power_envelope(best) is None:
        h_result = self._registry_heuristic(names).classify(window_np)
        if h_result.appliance == UNRECOGNISED or h_result.appliance != best:
            return UNRECOGNISED, 0.0, dist_map
        confidence = 1.0 - (1.0 - confidence) * (1.0 - float(h_result.confidence))

    if confidence < self.recognition_threshold:
        return UNRECOGNISED, confidence, dist_map

    return best, confidence, dist_map

except Exception as e:
    logger.error(f"ProtoNet classification error for {device_id}: {e}")
    return "error", 0.0, {}

d) New _registry_heuristic(self, names) -> HeuristicApplianceClassifier — cached per class-set (self._registry_clf_cache), scopes channel B to the registry's class list so the two channels can actually agree on the general household profile.

e) New _eligible_classes(self, window_np, names, registry) -> set:

feats = self.heuristic_clf.extract_features(window_np)
if not feats:
    return set(names)
steady = float(feats.get("steady_w", 0.0) or 0.0)
band_ok = plausible_classes(feats)
known_bands = {r.name for r in DEFAULT_RULES}

allowed = self.heuristic_clf.allowed_classes
enrolled = set(registry.envelopes) if registry is not None else set()

out = set()
for n in names:
    if allowed is not None and n not in allowed and n not in enrolled:
        continue
    env = registry.power_envelope(n) if registry is not None else None
    if env is not None:
        lo, hi = env
        # Pad for mains drift and PZEM resolution ONLY — the same
        # physical slack heuristic_fallback uses (±6% mains, P ∝ V², so
        # ~±12% power), plus 1 W for the meter's own quantisation.
        # It must NOT be widened to "cover the range a 5-sample
        # enrollment might have missed": at 25% an enrolled 45 W charger
        # spanned 33–57 W and captured a 35 W monitor.
        pad = max(ENVELOPE_SLACK * hi, 1.0)
        if lo - pad <= steady <= hi + pad:
            out.add(n)
    elif n in known_bands:
        if n in band_ok:
            out.add(n)
    else:
        out.add(n)      # gate may only veto where it has knowledge
return out

f) New _classify_heuristic(self, window_np) — returns the heuristic answer only if >= self.heuristic_min_confidence, else UNRECOGNISED.

g) LABEL_REQUEST broadcast (formerly ~line 890) — now carries raw watts:

stability, cluster_mean = self.delta_analyzer.push(embedding)

# Keep the RAW power windows that produced these embeddings.
buf = self._unknown_windows.setdefault(device_id, deque(maxlen=8))
buf.append(np.asarray(window_np[:128], dtype=np.float32).tolist())

if stability == 'stable':
    logger.info(f"❓ Stable unrecognised load on {device_id} ({power_watts:.1f}W) — requesting label")
    await self._broadcast_event({
        "type": "LABEL_REQUEST",
        "device_id": device_id,
        "power": round(power_watts, 2),
        "confidence": round(confidence, 3),
        # `segments` is what enrollment consumes: raw 128-sample POWER
        # windows in watts. This used to ship only `embedding`, and the
        # dashboard POSTed that straight back as `segments`. add_class()
        # then ran the CNN over a 128-D embedding as though it were a
        # power trace — silently, since the shapes match at (128,).
        "segments": [list(s) for s in buf],
        "embedding": cluster_mean.tolist() if cluster_mean is not None else [],
        "suggested_label": UNRECOGNISED_DISPLAY,
        "message": f"Unrecognised device on {device_id} at {power_watts:.0f} W. Please label it.",
    })

h) Low-confidence branch (formerly line ~974, the promotion bug) — heuristic re-run removed:

elif confidence < self.recognition_threshold:
    # No heuristic re-run here. `_classify_device` already consults the
    # deterministic classifier as its confirmation channel... The previous
    # code re-ran it and overwrote the class whenever the heuristic's
    # confidence merely beat ProtoNet's — which, since ProtoNet returned
    # 0.0 for every event, meant a 0.11-confidence guess became the
    # device's final answer and the operator was never asked for a label.
    logger.info(f"⚠️ Low confidence ({confidence:.3f}) for {class_name} on {device_id}. Skipping RL.")
    await self._broadcast_event({
        "type": "LOW_CONFIDENCE", "device_id": device_id,
        "classified_as": class_name, "confidence": round(confidence, 3),
        "threshold": self.recognition_threshold,
        "message": f"Classification uncertain ({confidence:.2f} < {self.recognition_threshold})",
    })

i) handle_label_submitted (line ~1305) — hardened:

segs = np.array(segments_list, dtype=np.float32)   # (K, 128)
if segs.ndim == 1:
    segs = segs.reshape(1, -1)
if segs.shape[-1] != 128:
    logger.error(f"Label segments wrong shape: {segs.shape}")
    return
if not np.all(np.isfinite(segs)):
    logger.error("Label segments contain NaN/Inf — refusing to enroll")
    return

# Reject embeddings passed in place of power traces. An embedding is
# 128-D too, so the shape check above cannot tell them apart... Watts are
# non-negative and a real appliance event reaches the 20 W on-threshold.
if float(segs.min()) < -1e-3 or float(segs.max()) < 20.0:
    logger.error(
        "Label segments do not look like power in watts "
        f"(min={segs.min():.3f}, max={segs.max():.3f}) — refusing to "
        "enroll '%s'. Expected raw 128-sample power windows, not "
        "embeddings.", class_name)
    return

self.prototype_registry.add_class(class_name, segs)

src/pipeline/heuristic_fallback.py — MODIFIED (this session, on top of prior task #3 work)

Prior session added: CENTROID_REJECT_RADIUS = 6.0, ENVELOPE_SLACK = 0.15, plausible_classes(), three centroids (desktop_computer, monitor, projector), reject_radius param. UNKNOWN = "unknown" at line 50.

This session, to fix the one failing test:

# in __init__, after centroid filtering:
# Centroids for classes that have no band rule. The physical gate is
# built from the rule set, so it has nothing to say about these — they
# must bypass it rather than be vetoed by it. Empty for the default
# centroid set, non-empty only when a caller supplies its own.
self._extra_centroids = {k for k in self.centroids
                         if k not in {r.name for r in self.rules}}

# in classify():
eligible = plausible_classes(f, self.rules)
if not eligible and not self._extra_centroids:
    return HeuristicResult(UNKNOWN, 0.0, features=f)

if self.centroids:
    result = self._classify_by_centroid(f, eligible)
    if result is not None:
        return result

if f.get("steady_w", 0.0) <= 0.0:
    return HeuristicResult(UNKNOWN, 0.0, features=f)

if not eligible:
    return HeuristicResult(UNKNOWN, 0.0, features=f)

return self._classify_by_rules(f, eligible)

# in _classify_by_centroid():
v = self.feature_vector(f)
rule_names = {r.name for r in self.rules}
names, dists = [], []
for name, c in self.centroids.items():
    if c.shape != v.shape:
        continue
    if eligible is not None and name in rule_names and name not in eligible:
        continue
    d = float(np.linalg.norm((v - c) / self.feature_scales))
    names.append(name); dists.append(d)

src/api/main.py — MODIFIED

Added import math (line 5)

LabelRequestEvent (line ~134) gained segments: List[List[float]] = [] and suggested_label: str = ""

label_entry in the MQTT LABEL_REQUEST handler (line ~288) now preserves "segments": evt.segments and "suggested_label": evt.suggested_label (so a reloaded dashboard reading /api/pending-labels can still submit)

LabelSubmission.segments description rewritten; validate_segments now also enforces:

if not all(math.isfinite(x) for x in seg):
    raise ValueError(f"Segment {i} contains NaN/Inf")
if min(seg) < -1e-3:
    raise ValueError(f"Segment {i} has negative values ({min(seg):.3f} W): expected "
                     "raw power in watts, not an embedding")
if max(seg) < 20.0:
    raise ValueError(f"Segment {i} never exceeds 20 W (max {max(seg):.3f}): expected "
                     "a real appliance power window, not an embedding")

frontend/src/components/DigitalTwin.jsx — MODIFIED

// Enrollment needs the raw 128-sample POWER windows, in watts, that the
// pipeline captured for this device — NOT event.embedding. Both are
// length-128 float arrays, so submitting the embedding passed every
// validation and then had the CNN run over it as though it were a power
// trace, producing a prototype built from nonsense.
const segments = Array.isArray(event.segments)
  ? event.segments.filter((s) => Array.isArray(s) && s.length === 128)
  : [];

if (segments.length === 0) {
  setError('No power-window data in event yet. Retrying next cycle.');
  setLoading(false);
  return;
}

src/models/protonet.py — MODIFIED (prior session, unchanged this session)

PrototypeRegistry gained ENVELOPE_KEY = "__power_envelopes__", self.envelopes, _steady_watts(support_segments, on_threshold_w=20.0), envelope record/merge in add_class, power_envelope(class_name), and backward-compatible save/load that pop the envelope key. classify() at line 597 returns (best, dists[best], dists). SupportSetManager.raw_windows is a Dict[str, List[np.ndarray]], empty by default.

Temp diagnostics — /home/pramodsb/.claude/jobs/944dafbc/tmp/

diag_ml.py, calib.py, separability.py, measure_conf.py, measure_gate.py, measure_agree.py, e2e.py, e2e_label.py

4. Errors and Fixes

AttributeError: module 'run_pipeline' has no attribute 'EMSPipeline' — the class is EMSOrchestrator; FullPipeline = EMSOrchestrator alias at line 1519. Fixed the harness.

TypeError: EMSOrchestrator.__init__() got an unexpected keyword argument 'config_path' — signature takes config: Optional[Dict]. Fixed by passing config=yaml.safe_load(open("config/config.demo.yaml")).

Enrolled-recall only 1/3 — new operator labels lost the softmax to near-identical shipped prototypes. Fixed with the enrolled-precedence rule (enrolled_hits) + always renormalising over survivors. → 3/3.

Enrolled envelope swallowed a neighbouring class — monitor 35W → my_charger conf=1.000, because the 25% pad made a 45 W charger span 33–57 W. Fixed by tightening pad to max(ENVELOPE_SLACK * hi, 1.0). → monitor 35W → monitor conf=0.604.

1 test regression (STILL FAILING): tests/test_real_data_and_ml_fallback.py::TestHeuristicDiscrimination::test_centroid_path_used_when_fitted — AssertionError: assert 'unknown' == 'tiny'. First fix attempt (_extra_centroids + rule-name exemption in _classify_by_centroid) did not resolve it. Root cause now understood from debugging: the test's _step_window(10, peak=12) has all samples at 10 W, below the 20 W on_threshold_w, so extract_features yields steady_w = 0.0, duty = 0.0. plausible_classes returns {phone_charger, router} via its steady <= 0 peak branch; _classify_by_centroid returns None (the feature vector's log10(max(0, 1e-6)) = -6 is enormously far from the tiny centroid's log10(10) = 1, so it exceeds CENTROID_REJECT_RADIUS = 6.0); classify() then hits the steady_w <= 0.0 → UNKNOWN return. This test predates my work (commit e94db22a) and tests/test_real_data_and_ml_fallback.py is unmodified in git.

No user feedback correcting my approach was received — the user sent only the original request and the stop/log interrupt.

5. Problem Solving

Everything below was established by measurement, not assumption.

The pipeline classified nothing. _classify_device called SupportSetManager.classify(), whose compute_prototypes() returns {} when raw_windows is empty. anchors_path is set only in config/config.yaml, never in the demo or hardware profiles, so every event on those profiles returned ("unknown", 0.0, {}). The trained 7-class registry was loaded, logged at startup, written to by handle_label_submitted, and never once read.

Confidence cannot be trusted, and no threshold fixes it. Accuracy is ~0.60 at every temperature tried (T=0.80138: 0.599; T=1.0: 0.599; T=0.25: 0.596; T=0.10: 0.611; T=0.05: 0.579) — lowering T only inflates confidence on wrong answers. At the fitted T, the existing confidence_threshold: 0.90 keeps only 16.7% of real known windows. Worse, the model is confidently wrong on exactly the user's target devices: laptop 65 W → desktop_computer @ 0.717, phone 45 W → monitor @ 0.690, phone 100 W → tv @ 0.862.

The physical gate alone barely helps (accuracy 0.631 ungated → 0.621 gated) because the errors are between classes with overlapping wattage bands.

Two-channel agreement is what works. Requiring the learned channel (registry prototype distance — scale-blind) to be confirmed by an independent deterministic channel (heuristic centroid/bands — uses absolute watts) raises accuracy-among-accepted from 0.621 → 0.851 and halves confidently-wrong on the target set. The cost is coverage (~0.37 of windows accepted), paid deliberately: an unanswered window costs one label, a wrong one corrupts that appliance's whole energy history.

Enrolled classes need different treatment than shipped ones. An operator label carries an envelope measured on their device; a shipped class carries a wide literature band shared with neighbours. So envelope containment is the confirmation for enrolled classes, and enrolled classes take precedence over shipped ones when both are eligible — otherwise the label loop silently does nothing.

The label loop was doubly broken and is now verified end-to-end. LABEL_REQUEST shipped only a 128-D embedding; the dashboard POSTed it as segments; add_class ran the CNN over it as though it were watts. All three layers now carry/validate raw power windows, and both the Pydantic validator and handle_label_submitted refuse embeddings-as-watts.

Final measured state (real code, config.demo.yaml):

Real UK-DALE windows: accepted 128/321 (0.399), acc|accepted 0.836

Novel out-of-family loads (kettle, oven, hairdryer, microwave, heater, EV, washing, fridge): 8/8 → unrecognised

Target set: monitor 35 W ✅, monitor 55 W ✅, laptop 120/200 W → ask, phone 5/18/100 W → ask, 2 still confidently wrong (laptop 65 W → desktop_computer, phone 45 W → monitor)

Label loop: enrolled-recall 3/3 at conf 1.000/1.000/1.000; kettle and heater still rejected after enrollment; embedding enrollment refused at both layers

Test suite: 473 passed, 1 failed (baseline was 474 passed)

Known data limitation (from the prior session, unchanged): the real data cannot support a modern phone-charger class — UK-DALE phone_charger has 39 windows, mean 5.2 W, only 2 usable above the 20 W threshold, from a single meter; router has zero. projector/phone_charger/router each come from one meter. Retraining will not fix "phone"; the user's own few-shot labels are the correct path, which is why the label loop was the priority.

6. All User Messages

/clear (local command, explicitly caveated as not to be responded to)

"continue with @claude_debug/DEBUG_SESSION_2026-08-25.md and fix ml pipline. It should work for phone, laptop, monitor if not recognized classify it as unrecognized device and i will label it."

"Stop everything now and log everything now i have to swich api keys"

No security-relevant constraints, credential-handling rules, or forbidden operations were stated by the user in this conversation.

7. Pending Tasks

#5 (pending) Add regression tests for the ML pipeline fixes — tests that fail before / pass after each fix, keeping the suite at ≥474 passing.

Resolve the 1 failing test test_centroid_path_used_when_fitted (root cause diagnosed, fix not yet applied). Options: (a) allow _classify_by_centroid to run for rule-less centroids even when steady_w <= 0 by moving the steady_w guard after the centroid path, (b) treat the test as a placebo that asserts sub-threshold behaviour the gate is designed to refuse and update it with a comment. Decide before claiming the baseline is restored.

Not yet done: graphify update . (required by CLAUDE.md §1.5 after modifying code files).

Not yet done: update claude_debug/DEBUG_SESSION_2026-08-25.md and claude_debug/ARCHITECTURE_AND_APIS.md to record M-5/M-6/M-7/M-8, the new recognition_threshold/heuristic_min_confidence config keys, UNRECOGNISED_DISPLAY, plausible_classes, PrototypeRegistry.power_envelope, and the new segments field on LABEL_REQUEST.

Optional / flagged for the user, not undertaken: add recognition_threshold: 0.45 explicitly to the config profiles; consider retraining with monitor/laptop-specific data (will not fix phone — data gap).

Carried over from the prior session, still open: M-2 (OverlapAwareNILMDetector unreachable at defaults, 5 s cooldown > 3.0 s overlap_window_s), H-4 (firmware Core-1 lockout latch race), 15 placebo tests + 39 hallucinated .process() calls, ImportError → MagicMock masking, aiosqlite teardown race.

8. Current Work

I was debugging the single failing test, tests/test_real_data_and_ml_fallback.py::TestHeuristicDiscrimination::test_centroid_path_used_when_fitted, which my heuristic-gate changes regressed (assert 'unknown' == 'tiny'). The suite stood at 473 passed, 1 failed against a 474 baseline.

I had run a diagnostic showing the mechanism:

features: {'peak_w': 12.0, 'steady_w': 0.0, 'duty': 0.0, 'overshoot': 1.0, 'volatility': 0.0, 'mean_w': 7.516}
extra_centroids: {'tiny', 'huge'}
eligible: ['phone_charger', 'router']
centroid result: None
classify: HeuristicResult(appliance='unknown', confidence=0.0, ...)

The test builds _step_window(10, peak=12) — a 10 W window, below the classifier's 20 W on_threshold_w — so steady_w = 0.0 and classify() returns UNKNOWN before the band rules, while _classify_by_centroid returns None because the degenerate log10(1e-6) = -6 feature puts it past CENTROID_REJECT_RADIUS = 6.0.

My last action was a grep confirming the test's provenance: it comes from commit e94db22a (the original 467-test suite) and tests/test_real_data_and_ml_fallback.py is not among the modified files (git status --short tests/ shows only test_ml_nilm_math_stress.py and test_relay_safety_boot_brownout.py). I was about to inspect FEATURE_SCALES to decide between fixing the classifier and amending the test. All other work (tasks #1, #2, #3, #4) was complete and verified end-to-end.

9. Optional Next Step

Per the stop request, no further work should proceed without the user's go-ahead. When resuming, the immediate next step is to resolve the one failing test and restore the ≥474 baseline, then complete task #5.

Verbatim, the state I left off at — the failing assertion:

FAILED tests/test_real_data_and_ml_fallback.py::TestHeuristicDiscrimination::test_centroid_path_used_when_fitted - AssertionError: assert 'unknown' == 'tiny'
1 failed, 473 passed, 4 warnings in 26.48s

and my own framing of it just before the interrupt:

"One failure — my eligibility gate vetoes centroid classes that have no rule, so the gate is vetoing where it has no knowledge."

Files changed and uncommitted on main (nothing committed this session): scripts/run_pipeline.py, src/pipeline/heuristic_fallback.py, src/api/main.py, frontend/src/components/DigitalTwin.jsx, src/models/protonet.py, plus the pre-existing modifications listed in the opening git status.

If you need specific details from before compaction (like exact code snippets, error messages, or content you generated), read the full transcript at: /home/pramodsb/.claude/projects/-home-pramodsb-Downloads-mjr/f0b7c07d-b903-4922-a2db-4d0b6a6595ea.jsonl
Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with "I'll continue" or similar. Pick up the last task as if the break never happened.

                                                           03:24 PM claude-opus-5

Stopping now. Writing the handoff log.

● Write(/home/pramodsb/Downloads/mjr/claude_debug/ML_PIPELINE_FIX_2026-08-25.md)
Wrote 379 lines to /home/pramodsb/Downloads/mjr/claude_debug/ML_PIPELINE_FIX_2026-08-25.md

ML Pipeline Fix — Session Log (2026-08-25)

Status at stop: work paused mid-task at the user's request (API key sw
itch).
Test state: 473 passed, 1 failed (baseline was 474 passed).
Everything is uncommitted on main. Nothing has been pushed.
Continues from [DEBUG_SESSION_2026-08-25.md](./DEBUG_SESSION_2026-08-25.
md).

0. The ask

"continue with @claude_debug/DEBUG_SESSION_2026-08-25.md and fix ml pipli
ne. It should
work for phone, laptop, monitor if not recognized classify it as unrecogni
zed device
and i will label it."

1. STOP THE PRESSES: the one failing test

FAILED tests/test_real_data_and_ml_fallback.py::TestHeuristicDiscrimination:
:test_centroid_path_used_when_fitted
AssertionError: assert 'unknown' == 'tiny'

This is the only regression and it is NOT yet resolved. Resume here.

What the test does (tests/test_real_data_and_ml_fallback.py:158-166)

centroids = {
    'tiny': (np.log10(10), np.log10(12), 0.75, 0.05, 0.05),    # a 10 W load
    'huge': (np.log10(3000), np.log10(3200), 0.75, 0.05, 0.05),
}
clf = HeuristicApplianceClassifier(centroids=centroids)
assert clf.classify(_step_window(10, peak=12)).appliance == 'tiny'    # <--
FAILS
assert clf.classify(_step_window(3000, peak=3200)).appliance == 'huge'  # pa
sses

Root cause — diagnosed, not guessed

Measured directly:

features:          {'peak_w': 12.0, 'steady_w': 0.0, 'duty': 0.0, 'overshoot
': 1.0, ...}
extra_centroids:   {'tiny', 'huge'}
eligible:          ['phone_charger', 'router']
centroid result:   None
classify:          unknown

_step_window(10, peak=12) is a 10 W load. extract_features only coun
ts samples
above on_threshold_w = 20.0, so steady_w = 0.0. classify() then hits t
he
if f.get("steady_w", 0.0) <= 0.0: return UNKNOWN guard before the band
rules, and
_classify_by_centroid returns None because... (see below).

The steady_w = 0.0 early-return is *pre-existing behaviour I did not add
*. What changed
is that the centroid path used to be reached and answer tiny first. Now it
returns None.

Why _classify_by_centroid returns None here

Not the eligibility filter — I already exempted no-rule centroids from it
(name in rule_names and name not in eligible), and tiny/huge are in
_extra_centroids so they are exempt. It must therefore be the reject rad
ius:
CENTROID_REJECT_RADIUS = 6.0 in scaled-feature units, with steady_w = 0.0
 →
log10(1e-6) = -6.0 in the feature vector, which throws the distance far ou
tside the
radius. This was the next thing I was checking when work stopped — I was
reading
FEATURE_SCALES to confirm the arithmetic.

Three candidate fixes (NOT yet decided — needs a judgement call)

**Compute the centroid feature vector from mean_w when steady_w == 0.
** A sub-20 W
window has real power, just none above the on-threshold. Most faithful, t
ouches
feature_vector, some risk to other callers.

Let _classify_by_centroid run before the steady_w <= 0 guard for
_extra_centroids only. Narrow, but leaves the log10(1e-6) = -6 distor
tion in place.

Accept the new behaviour and update the test to use a >20 W load (e.g
.
_step_window(40, peak=48) with matching centroids). Defensible — a 10 W
load is
below the classifier's stated operating floor, and CLAUDE.md §1.7 routes
3–10 W
trickle to PhantomTracker, not the classifier. But this is changing a
test to
match code, which needs the user's explicit sign-off — exactly the "pla
cebo test"
pattern the prior session was cleaning up.

My inclination is (3) with the user's agreement, because the test's prem
ise (that the
deterministic classifier should name a 10 W load) contradicts the documented
20 W floor.
Do not do this silently.

2. What was measured (the evidence base — do not re-derive this, it is ex

pensive)

2.1 The pipeline classified NOTHING before this session

_classify_device called SupportSetManager.classify(). That object is onl
y populated from
protonet.anchors_path, set in config/config.yaml alone — never in th
e demo or
hardware profiles. So compute_prototypes() returned {} and the method re
turned
("unknown", 0.0, {}) for every event on every profile. The trained 7-c
lass
PrototypeRegistry was loaded, logged at startup, written to by handle_lab
el_submitted,
and never once read. (Defect M-5.)

2.2 The learned embedding is power-scale-blind — open-set rejection cann

ot use it

probe

result

kettle 2000 W, d2 to nearest known prototype

5.89

in-distribution known windows, p99

5.10

washing machine 500 W

lands 1.02 from the 5 W router prototype



Novel-load distances sit inside the known range, so no distance threshol
d can work.
Absolute watts (measured directly by the PZEM) are the only reliable novelty
signal.
This is why the fix is a physical power-envelope gate, not a Weibull/OpenMax
tune.
(Defect M-6.)

Also: _weibull_by_name is empty in both shipped artifacts because
scripts/train_demo_models.py:171-179 uses the indexed openmax.fit(idx, ..
.) API, which
only populates _weibull[idx]. So compute_open_set_prob returns 0.0 unc
onditionally
(src/models/protonet.py:399-400). Dead code. Not yet fixed — see §5.

2.3 Confidence alone is useless: the model is CONFIDENTLY WRONG on the t

arget devices

Measured on the shipped 7-class demo artifact:

load

reported as

confidence

laptop 65 W

desktop_computer

0.72

phone charger 100 W

tv

0.86

phone charger 45 W

monitor

0.69

Raising the gate does not remove these — it removes the correct low-con
fidence
answers along with them. Threshold sweep (gated, real UK-DALE windows):

threshold

coverage

accuracy-among-accepted

0.00

1.000

0.621

0.45

0.733

0.684

0.55

0.566

0.780

0.90

0.160

0.922

Accuracy is ~0.60 at every temperature (T ∈ {0.80, 1.0, 0.25, 0.1, 0.05}
), so lowering
T only inflates confidence on wrong answers. The temperature is not the pr
oblem.

2.4 The two-channel agreement rule is what actually works

Require the learned channel to be confirmed by an independent, non-learned m
easurement:

A — registry prototype distance (learned shape, scale-blind)

B — deterministic heuristic centroid/band (uses absolute watts)

metric

ungated

agreement rule

accuracy among accepted (real UK-DALE)

0.621

0.851

novel out-of-family loads rejected

6/8

8/8

confidently-wrong on phone/laptop/monitor

4/9

2/9

Coverage falls to ~0.37–0.40 of windows accepted. This is a deliberate tra
de: an
unanswered window costs one label; a wrong one corrupts that appliance's who
le energy
history.

2.5 The real data cannot support a modern phone-charger class — retraini

ng will NOT fix "phone"

class

raw windows

usable (>20 W)

meters

note

phone_charger

39

2

1

mean 5.2 W, max 33 W

router

15

0

1

nothing above 20 W at all

projector

27

26

1

single meter → cannot generalise

UK-DALE's phone_charger is a ~5 W 2012-era trickle charger. A modern 45–10
0 W USB-PD
charger is a different appliance. laptop vs monitor is weakly separable
at best:
0.501 leave-one-meter-out on 3 classes (chance 0.333), with overlapping
bands
(laptop p10–p90 = 21–92 W, monitor 26–77 W).

Conclusion: the user's own instruction is the correct architecture. Make
recognise → reject → label → recognise actually work, and let their few-shot
labels
specialise the system to their real devices. Retraining was deliberately n
ot attempted;
it is a data-collection problem, not a training problem.

2.6 End-to-end verification of the delivered behaviour

=== does it classify at all? (real UK-DALE windows) ===
  desktop_computer   accepted=35/60  acc|accepted=0.914
  monitor            accepted=43/60  acc|accepted=0.953
  projector          accepted=26/60  acc|accepted=1.000
  tv                 accepted=17/60  acc|accepted=0.471
  laptop             accepted= 7/60  acc|accepted=0.000
  phone_charger      accepted= 0/39  (correctly abstains — see 2.5)
  router             accepted= 0/15  (correctly abstains — see 2.5)
  OVERALL            accepted=128/321 (0.399)  acc|accepted=0.836

=== novel out-of-family loads -> unrecognised: 8/8 ===
  kettle 2000W, oven 2500W, hairdryer 1200W, microwave 900W,
  heater 800W, ev 7000W, washing 500W, fridge 120W   ALL -> unknown

=== THE LABEL LOOP (the user's actual requirement) ===
  enrolled my_phone   envelope=(44.69, 45.51)
  enrolled my_laptop  envelope=(64.39, 65.67)
  enrolled my_monitor envelope=(34.86, 35.36)
  re-test:  my_phone -> my_phone (1.000)   my_laptop -> my_laptop (1.000)
            my_monitor -> my_monitor (1.000)      enrolled-recall: 3/3
  kettle 2000W -> unknown   heater 800W -> unknown   (no masquerading)

=== full label loop through the REAL code path ===
  1. unrecognised 45 W charger -> LABEL_REQUEST with 6 raw power windows  ✅
  2. LabelRequestEvent carries segments through the API                  ✅
  3. LabelSubmission ACCEPTS watts, REJECTS an embedding                 ✅
  4. handle_label_submitted -> registry -> recognised next event         ✅
  5. enrolling an embedding refused by the handler too                   ✅
  6. monitor 35W -> monitor (not captured by the 45 W enrolled class)    ✅

Still confidently wrong (2/9), unresolved: laptop 65W -> desktop_comput
er,
phone 45W -> monitor. Both are shipped-class errors rooted in §2.5's data
gap. The
label loop is the intended remedy: the user enrolls their own laptop and cha
rger, and
enrolled classes then take precedence (§3.2).

3. Files changed (all uncommitted)

3.1 src/pipeline/heuristic_fallback.py — the reject channel

New constants CENTROID_REJECT_RADIUS = 6.0, ENVELOPE_SLACK = 0.15
(±6% mains → ~±12% power since P ∝ V²).

New module function plausible_classes(features, rules=None, slack=ENVEL
OPE_SLACK) -> set
— the working power-envelope gate. steady_w is the discriminator; peak
is tested
one-sided (a load cannot peak below its own floor).

Added centroids desktop_computer, monitor, projector (fitted via
fit_heuristic_centroids.py --cache-tag _demo --dry-run).
Deliberately did NOT add phone_charger (fits to 5.0 W) or router (
6.0 W) —
unreachable below the 20 W on-threshold. Documented with a 🔴 comment tyin
g to
CLAUDE.md §1.7 (PhantomTracker owns 3–10 W trickle).

classify() now gates first; the centroid path may decline (returns Opti
onal).

_classify_by_centroid / _classify_by_rules take an eligible set.

__init__ gained reject_radius.

Last edits, made while fixing the failing test: _extra_centroids set
(centroids
with no band rule bypass the gate, since the gate has no knowledge of them
); the
not eligible early-return now also requires not self._extra_centroids;
a second
not eligible guard moved below the steady_w <= 0 check.
⚠️ These last edits are the ones that did not fully land — see §1.

3.2 scripts/run_pipeline.py — the core rewire

Imports plausible_classes, DEFAULT_RULES, ENVELOPE_SLACK, UNKNOWN a
s UNRECOGNISED.

New UNRECOGNISED_DISPLAY = "Unrecognised device". "unknown" is kept
as the
internal/wire sentinel on purpose — src/api/main.py, DeviceCards.jsx
and 54 test
assertions key on that literal. Renaming it would break the contract for n
o gain.

HeuristicApplianceClassifier(allowed_classes=self.config.get("appliances"
)) — the
hardware profile (laptop + phone_charger only) could previously report hv
ac.

New self.recognition_threshold (default 0.45, from the §2.3 sweep),
deliberately
separate from confidence_threshold (0.90): the latter gates a single
softmax, the
former gates the noisy-OR of two channels that have already agreed. Differ
ent quantities;
must not share a number.

_classify_device fully rewritten: registry → temperature-scaled soft
max over
-d2/T (T floored at 0.05, mirroring TemperatureScaler) → physical gate
→
renormalise over survivors → two-channel agreement → recognition_threshol
d.
Returns UNRECOGNISED when not recognised, which fires the existing LABEL
_REQUEST flow.

New _eligible_classes() — envelope source in preference order: registry
(enrolled),
then DEFAULT_RULES, then unconstrained. The gate may only veto wher
e it actually
has knowledge. Also drops registry classes outside appliances: (the shi
pped demo
artifact carries all 7 regardless of profile), exempting enrolled classes.

Enrolled classes take precedence over shipped ones when a window sits
inside an
enrolled envelope. Without this the label loop silently does nothing: the
shipped
monitor prototype sits almost on top of a newly enrolled 35 W monitor, t
he probability
halves between them, and the result falls under the threshold.
Measured: enrolled-recall 1/3 → 3/3.

Enrolled envelope pad is max(ENVELOPE_SLACK * hi, 1.0) — physical slack
only.
It was max(0.25*hi, 5.0), which made a 45 W charger span 33–57 W and s
wallow a 35 W
monitor; caught by test 6 and fixed.

New _registry_heuristic(names) — channel B scoped to the registry's clas
s set, cached
per class-set (construction filters rules; rebuilding per event would allo
cate on the
MQTT ingest path). Without the scoping the two channels could never agree
on the general
household profile.

New _classify_heuristic() — degraded mode, gated by heuristic_min_confi
dence (0.55).

Removed the low-confidence promotion at old line 820. It accepted any
heuristic
answer whose confidence merely beat ProtoNet's — and since ProtoNet return
ed 0.0 for
every event, a 0.11-confidence guess became the device's final classific
ation and
the operator was never asked for a label. The elif now keys on
recognition_threshold and does not re-run the heuristic (channel B has a
lready run).

LABEL_REQUEST now carries segments (raw 128-sample power windows i
n watts) plus
suggested_label. New self._unknown_windows: Dict[str, deque] (maxlen 8
) retains the
raw traces, because DeltaStabilityAnalyzer only keeps embeddings.

handle_label_submitted now rejects NaN/Inf and rejects embeddings pass
ed as watts
(min < -1e-3 or max < 20 W). An embedding is 128-D too, so the shape c
heck could
never tell them apart.

3.3 src/models/protonet.py — envelopes for enrolled classes

ENVELOPE_KEY = "__power_envelopes__", self.envelopes: Dict[str, Tuple[f
loat,float]].

New _steady_watts(support_segments, on_threshold_w=20.0) staticmethod.

add_class records/merges the observed envelope (so re-labelling wide
ns it).

New power_envelope(class_name).

Backward-compatible save/load (envelope key popped out of the prototyp
e dict).

PrototypeRegistry docstring documents the power-scale blindness.

3.4 src/api/main.py — label payload contract

import math added.

LabelRequestEvent gained segments: List[List[float]] and suggested_la
bel.

label_entry persists segments + suggested_label so a reloaded dashbo
ard can submit
a label from /api/pending-labels without the original WebSocket frame.

LabelSubmission.validate_segments now rejects non-finite values, negativ
es
(< -1e-3), and windows that never exceed 20 W — i.e. rejects embedding
s. The
length-128 check alone let the dashboard submit embeddings for months.

3.5 frontend/src/components/DigitalTwin.jsx

handleSubmit now submits event.segments (raw watts), not event.em
bedding.
Error message updated to "No power-window data in event yet."

4. Defect ledger

id

defect

status

M-5

Inference never ran — dead SupportSetManager, unused registry

✅

fixed

M-6

Open-set rejection dead (_weibull_by_name empty) and undistanc

eable

✅ worked around via physical gate; OpenMax itself still dead (§5)

M-7

Heuristic could not emit consumer classes, never said UNKNOWN

✅ f

ixed

M-8

Label loop doubly broken (embedding-as-window; registry never read)

✅ fixed, verified end-to-end

—

Low-confidence promotion at run_pipeline.py:820

✅ removed

—

allowed_classes never passed from config

✅ fixed

—

test_centroid_path_used_when_fitted regression

❌ OPEN — §1

5. Not done / next steps, in priority order

Resolve the failing test (§1). Blocking — the suite is at 473/474.

graphify update . — required by CLAUDE.md §1.5 after code edits. **
Not yet run.**

OpenMax is still dead code. train_demo_models.py must call the by-n
ame fit API
(or persist the class-name order) so _weibull_by_name is populated. Cur
rently
compute_open_set_prob returns 0.0 unconditionally. The physical gate ma
kes this
non-blocking, but it is misleading dead code in the critical path.

Add the regression tests for each fix (task #5, never started): infer
ence actually
returns a class; novel loads reject; enrolled class recognised on the nex
t event;
embedding-as-segments refused at both the validator and the handler; enro
lled
precedence; the 45 W-envelope-vs-35 W-monitor case.

Update claude_debug/ARCHITECTURE_AND_APIS.md with the new API surfa
ce:
plausible_classes, PrototypeRegistry.power_envelope, _eligible_class
es,
_registry_heuristic, _classify_heuristic, recognition_threshold,
UNRECOGNISED_DISPLAY, and the LABEL_REQUEST.segments contract. CLAUDE
.md §4's
anti-hallucination table also needs the new methods.

Consider recognition_threshold in the config files — it currently o
nly has a code
default (0.45). The demo/hardware profiles should state it explicitly.

Optional, flagged for the user's decision: retraining. Per §2.5 this
will not fix
phone/laptop/monitor — the UK-DALE data does not contain a modern USB-PD
charger and
laptop/monitor are barely separable. The honest path is enrolling the use
r's own devices
through the now-working label loop. Recommend NOT retraining.

6. Reproduction commands

source .venv/bin/activate

# the failing test, alone
python -m pytest tests/test_real_data_and_ml_fallback.py::TestHeuristicDiscr
imination::test_centroid_path_used_when_fitted -v

# full suite (expect 473 passed, 1 failed until §1 is resolved)
python -m pytest tests/ -q

Diagnostic scripts live in /home/pramodsb/.claude/jobs/944dafbc/tmp/ — t
hese are in a
job temp dir and will be deleted when the job is removed. Copy them out if
they are
worth keeping:

script

what it measures

diag_ml.py

initial "does anything classify at all" probe

calib.py

in-distribution vs novel d2 distributions (§2.2)

separability.py

leave-one-meter-out separability + data provenance (§2

.5)

measure_conf.py

confidence distribution across temperatures (§2.3)

measure_gate.py

gate effect on accuracy, threshold sweep (§2.3)

measure_agree.py

two-channel agreement rule (§2.4)

e2e.py

end-to-end through the real _classify_device (§2.6)

e2e_label.py

full label loop incl. API validation (§2.6)

Note: FullPipeline is an alias for EMSOrchestrator, whose __init__ tak
es
config=<dict>, not config_path=. The diagnostic scripts load the YAM
L themselves.
Interrupted · What should Claude do instead?

❯ /compact
⎿  Compacted (ctrl+o to see full summary)
Read claude_debug/ML_PIPELINE_FIX_2026-08-25.md (380 lines)
Referenced file src/pipeline/heuristic_fallback.py
Referenced file scripts/run_pipeline.py
Read ../../.claude/jobs/944dafbc/tmp/e2e_label.py (88 lines)
Referenced file src/api/main.py
