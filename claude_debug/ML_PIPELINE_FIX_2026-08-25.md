# ML Pipeline Fix — Session Log (2026-08-25)

> **Status:** ✅ **complete.** The blocking test is resolved, regression tests are in,
> the graph is synced and the docs are current.
> **Test state:** `511 passed` (was 473 passed / 1 failed; baseline before this work 474).
> **Everything is uncommitted on `main`.** Nothing has been pushed.
> Continues from [`DEBUG_SESSION_2026-08-25.md`](./DEBUG_SESSION_2026-08-25.md).

---

## 0. The ask

> *"continue with @claude_debug/DEBUG_SESSION_2026-08-25.md and fix ml pipline. It should
> work for phone, laptop, monitor if not recognized classify it as unrecognized device
> and i will label it."*

---

## 1. RESOLVED: the one failing test

```
tests/test_real_data_and_ml_fallback.py::TestHeuristicDiscrimination::test_centroid_path_used_when_fitted
AssertionError: assert 'unknown' == 'tiny'      # ✅ FIXED — see below
```

**Outcome: the test was passing on garbage before this work, and that is measured,
not argued.** All three candidate fixes below were resolved by measurement.

### What the test does (`tests/test_real_data_and_ml_fallback.py`)

```python
centroids = {
    'tiny': (np.log10(10), np.log10(12), 0.75, 0.05, 0.05),    # a 10 W load
    'huge': (np.log10(3000), np.log10(3200), 0.75, 0.05, 0.05),
}
clf = HeuristicApplianceClassifier(centroids=centroids)
assert clf.classify(_step_window(10, peak=12)).appliance == 'tiny'
```

### Root cause, and why the old pass was meaningless

`_step_window(10, peak=12)` is a **10 W** load. `extract_features` only counts
samples above `on_threshold_w = 20.0`, so `steady_w = 0.0`, and `feature_vector`
then substitutes `log10(1e-6) = **-6.0**` for `log_steady` — a fabricated
coordinate seven decades below the real power. Measured distances at that
coordinate:

| | `d(tiny)` | `d(huge)` | verdict |
| :--- | ---: | ---: | :--- |
| at the default 20 W floor | **34.01** | **47.26** | neither is a match |
| with `on_threshold_w=3.0` | **1.24** | 16.24 | genuine 13× separation |

At HEAD the centroid path was a **pure argmin with no reject radius**, so it
returned the nearest class at *any* distance. `'tiny'` won only because `-6`
sorts nearer `log10(10)=1` than `log10(3000)=3.48`. Confirmed by running the
pre-fix code directly: `10 W → 'tiny' conf=0.017`. Adding
`CENTROID_REJECT_RADIUS = 6.0` correctly started rejecting `d=34.01`, which is
what surfaced as the "regression".

### Which candidate fix was taken, and why the other two were rejected

| candidate | measured outcome | verdict |
| :--- | :--- | :--- |
| **(1)** feature vector from `mean_w` when `steady_w == 0` | on the **default** classifier a 19 W window goes `d(monitor) 37.76 → 4.02` — inside the 6.0 radius, so a sub-floor trickle load starts being named `monitor` | ❌ **regresses production**; that load is `PhantomTracker`'s per CLAUDE.md §1.7 |
| **(2)** run the centroid path before the `steady_w <= 0` guard | leaves the `log10(1e-6) = -6` distortion in place, so the answer still comes from a fabricated coordinate | ❌ preserves the original defect |
| **(3)** construct with `on_threshold_w=3.0` — the API's own documented sub-20 W path, already used by `TestLowPowerDemoBand` | `d(tiny)=1.24` vs `d(huge)=16.24`; `d(huge)=1.22` vs `d(tiny)=16.24` | ✅ **taken.** The test's *stated intent* ("two well-separated centroids; nearest must win") now rests on real separation. Zero production change. |

The prior session's note said (3) "needs the user's explicit sign-off" because it
looked like changing a test to match code. The measurement changes that reading:
the assertion was **never** testing nearest-centroid matching — it was testing
argmin over a fabricated coordinate. Fixing the *setup* so it tests what its own
comment claims is not a placebo, it is the opposite.

**Two tests added alongside it**, so the behaviour cuts both ways and neither
direction can silently drift:
* `test_caller_centroids_bypass_the_physical_gate` — a caller centroid with no
  band rule is not vetoed by a gate built from the rule set.
* `test_sub_floor_load_is_unknown_at_the_default_threshold` — **fails at HEAD**
  (where the 10 W window returned `'tiny'`); pins that a load below the
  measurement floor is reported UNKNOWN rather than assigned.

---

## 2. What was measured (the evidence base — do not re-derive this, it is expensive)

### 2.1 The pipeline classified NOTHING before this session

`_classify_device` called `SupportSetManager.classify()`. That object is only populated from
`protonet.anchors_path`, set in `config/config.yaml` **alone** — never in the demo or
hardware profiles. So `compute_prototypes()` returned `{}` and the method returned
`("unknown", 0.0, {})` for **every event on every profile**. The trained 7-class
`PrototypeRegistry` was loaded, logged at startup, written to by `handle_label_submitted`,
and **never once read**. (Defect M-5.)

### 2.2 The learned embedding is power-scale-blind — open-set rejection cannot use it

| probe | result |
| :--- | :--- |
| kettle 2000 W, d2 to nearest known prototype | 5.89 |
| in-distribution known windows, p99 | 5.10 |
| washing machine 500 W | lands **1.02** from the **5 W `router`** prototype |

Novel-load distances sit *inside* the known range, so **no distance threshold can work**.
Absolute watts (measured directly by the PZEM) are the only reliable novelty signal.
This is why the fix is a physical power-envelope gate, not a Weibull/OpenMax tune.
(Defect M-6.)

Also: `_weibull_by_name` is **empty in both shipped artifacts** because
`scripts/train_demo_models.py:171-179` uses the indexed `openmax.fit(idx, ...)` API, which
only populates `_weibull[idx]`. So `compute_open_set_prob` returns `0.0` unconditionally
(`src/models/protonet.py:399-400`). Dead code. **Not yet fixed** — see §5.

### 2.3 Confidence alone is useless: the model is CONFIDENTLY WRONG on the target devices

Measured on the shipped 7-class demo artifact:

| load | reported as | confidence |
| :--- | :--- | :--- |
| laptop 65 W | `desktop_computer` | **0.72** |
| phone charger 100 W | `tv` | **0.86** |
| phone charger 45 W | `monitor` | **0.69** |

Raising the gate does **not** remove these — it removes the *correct low-confidence*
answers along with them. Threshold sweep (gated, real UK-DALE windows):

| threshold | coverage | accuracy-among-accepted |
| ---: | ---: | ---: |
| 0.00 | 1.000 | 0.621 |
| 0.45 | 0.733 | 0.684 |
| 0.55 | 0.566 | 0.780 |
| 0.90 | 0.160 | 0.922 |

Accuracy is ~0.60 at **every** temperature (T ∈ {0.80, 1.0, 0.25, 0.1, 0.05}), so lowering
T only inflates confidence on wrong answers. **The temperature is not the problem.**

### 2.4 The two-channel agreement rule is what actually works

Require the learned channel to be confirmed by an independent, non-learned measurement:

* **A** — registry prototype distance (learned shape, scale-blind)
* **B** — deterministic heuristic centroid/band (uses absolute watts)

| metric | ungated | agreement rule |
| :--- | ---: | ---: |
| accuracy among accepted (real UK-DALE) | 0.621 | **0.851** |
| novel out-of-family loads rejected | 6/8 | **8/8** |
| confidently-wrong on phone/laptop/monitor | 4/9 | **2/9** |

Coverage falls to ~0.37–0.40 of windows accepted. **This is a deliberate trade**: an
unanswered window costs one label; a wrong one corrupts that appliance's whole energy
history.

### 2.5 The real data cannot support a modern phone-charger class — retraining will NOT fix "phone"

| class | raw windows | usable (>20 W) | meters | note |
| :--- | ---: | ---: | ---: | :--- |
| `phone_charger` | 39 | **2** | 1 | mean 5.2 W, max 33 W |
| `router` | 15 | **0** | 1 | nothing above 20 W at all |
| `projector` | 27 | 26 | 1 | single meter → cannot generalise |

UK-DALE's `phone_charger` is a ~5 W 2012-era trickle charger. A modern 45–100 W USB-PD
charger is a **different appliance**. laptop vs monitor is weakly separable at best:
**0.501** leave-one-meter-out on 3 classes (chance 0.333), with overlapping bands
(laptop p10–p90 = 21–92 W, monitor 26–77 W).

**Conclusion: the user's own instruction is the correct architecture.** Make
recognise → reject → label → recognise actually work, and let their few-shot labels
specialise the system to their real devices. Retraining was deliberately **not** attempted;
it is a data-collection problem, not a training problem.

### 2.6 End-to-end verification of the delivered behaviour

```
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
```

**Still confidently wrong (2/9), unresolved:** `laptop 65W -> desktop_computer`,
`phone 45W -> monitor`. Both are shipped-class errors rooted in §2.5's data gap. The
label loop is the intended remedy: the user enrolls their own laptop and charger, and
enrolled classes then take precedence (§3.2).

---

## 3. Files changed (all uncommitted)

### 3.1 `src/pipeline/heuristic_fallback.py` — the reject channel

* New constants `CENTROID_REJECT_RADIUS = 6.0`, `ENVELOPE_SLACK = 0.15`
  (±6% mains → ~±12% power since P ∝ V²).
* New module function **`plausible_classes(features, rules=None, slack=ENVELOPE_SLACK) -> set`**
  — the working power-envelope gate. `steady_w` is the discriminator; `peak` is tested
  one-sided (a load cannot peak below its own floor).
* Added centroids `desktop_computer`, `monitor`, `projector` (fitted via
  `fit_heuristic_centroids.py --cache-tag _demo --dry-run`).
  **Deliberately did NOT add** `phone_charger` (fits to 5.0 W) or `router` (6.0 W) —
  unreachable below the 20 W on-threshold. Documented with a 🔴 comment tying to
  CLAUDE.md §1.7 (`PhantomTracker` owns 3–10 W trickle).
* `classify()` now gates first; the centroid path may decline (returns `Optional`).
* `_classify_by_centroid` / `_classify_by_rules` take an `eligible` set.
* `__init__` gained `reject_radius`.
* **Last edits, made while fixing the failing test:** `_extra_centroids` set (centroids
  with no band rule bypass the gate, since the gate has no knowledge of them); the
  `not eligible` early-return now also requires `not self._extra_centroids`; a second
  `not eligible` guard moved below the `steady_w <= 0` check.
  ⚠️ **These last edits are the ones that did not fully land — see §1.**

### 3.2 `scripts/run_pipeline.py` — the core rewire

* Imports `plausible_classes`, `DEFAULT_RULES`, `ENVELOPE_SLACK`, `UNKNOWN as UNRECOGNISED`.
* New `UNRECOGNISED_DISPLAY = "Unrecognised device"`. **`"unknown"` is kept as the
  internal/wire sentinel on purpose** — `src/api/main.py`, `DeviceCards.jsx` and 54 test
  assertions key on that literal. Renaming it would break the contract for no gain.
* `HeuristicApplianceClassifier(allowed_classes=self.config.get("appliances"))` — the
  hardware profile (laptop + phone_charger only) could previously report `hvac`.
* New `self.recognition_threshold` (default **0.45**, from the §2.3 sweep), deliberately
  **separate** from `confidence_threshold` (0.90): the latter gates a single softmax, the
  former gates the noisy-OR of two channels that have already agreed. Different quantities;
  must not share a number.
* **`_classify_device` fully rewritten**: registry → temperature-scaled softmax over
  `-d2/T` (T floored at 0.05, mirroring `TemperatureScaler`) → physical gate →
  renormalise over survivors → two-channel agreement → `recognition_threshold`.
  Returns `UNRECOGNISED` when not recognised, which fires the existing LABEL_REQUEST flow.
* New `_eligible_classes()` — envelope source in preference order: registry (enrolled),
  then `DEFAULT_RULES`, then **unconstrained**. *The gate may only veto where it actually
  has knowledge.* Also drops registry classes outside `appliances:` (the shipped demo
  artifact carries all 7 regardless of profile), exempting enrolled classes.
* **Enrolled classes take precedence over shipped ones** when a window sits inside an
  enrolled envelope. Without this the label loop silently does nothing: the shipped
  `monitor` prototype sits almost on top of a newly enrolled 35 W monitor, the probability
  halves between them, and the result falls under the threshold.
  **Measured: enrolled-recall 1/3 → 3/3.**
* Enrolled envelope pad is `max(ENVELOPE_SLACK * hi, 1.0)` — physical slack only.
  It was `max(0.25*hi, 5.0)`, which made a 45 W charger span 33–57 W and **swallow a 35 W
  monitor**; caught by test 6 and fixed.
* New `_registry_heuristic(names)` — channel B scoped to the registry's class set, cached
  per class-set (construction filters rules; rebuilding per event would allocate on the
  MQTT ingest path). Without the scoping the two channels could never agree on the general
  household profile.
* New `_classify_heuristic()` — degraded mode, gated by `heuristic_min_confidence` (0.55).
* **Removed the low-confidence promotion at old line 820.** It accepted any heuristic
  answer whose confidence merely beat ProtoNet's — and since ProtoNet returned 0.0 for
  every event, a **0.11-confidence guess became the device's final classification** and
  the operator was never asked for a label. The `elif` now keys on
  `recognition_threshold` and does not re-run the heuristic (channel B has already run).
* `LABEL_REQUEST` now carries **`segments`** (raw 128-sample power windows in watts) plus
  `suggested_label`. New `self._unknown_windows: Dict[str, deque]` (maxlen 8) retains the
  raw traces, because `DeltaStabilityAnalyzer` only keeps embeddings.
* `handle_label_submitted` now rejects NaN/Inf **and rejects embeddings passed as watts**
  (`min < -1e-3` or `max < 20 W`). An embedding is 128-D too, so the shape check could
  never tell them apart.

### 3.3 `src/models/protonet.py` — envelopes for enrolled classes

* `ENVELOPE_KEY = "__power_envelopes__"`, `self.envelopes: Dict[str, Tuple[float,float]]`.
* New `_steady_watts(support_segments, on_threshold_w=20.0)` staticmethod.
* `add_class` records/**merges** the observed envelope (so re-labelling widens it).
* New `power_envelope(class_name)`.
* Backward-compatible `save`/`load` (envelope key popped out of the prototype dict).
* `PrototypeRegistry` docstring documents the power-scale blindness.

### 3.4 `src/api/main.py` — label payload contract

* `import math` added.
* `LabelRequestEvent` gained `segments: List[List[float]]` and `suggested_label`.
* `label_entry` persists `segments` + `suggested_label` so a reloaded dashboard can submit
  a label from `/api/pending-labels` without the original WebSocket frame.
* `LabelSubmission.validate_segments` now rejects non-finite values, negatives
  (`< -1e-3`), and windows that never exceed 20 W — i.e. **rejects embeddings**. The
  length-128 check alone let the dashboard submit embeddings for months.

### 3.5 `frontend/src/components/DigitalTwin.jsx`

* `handleSubmit` now submits `event.segments` (raw watts), **not** `event.embedding`.
  Error message updated to "No power-window data in event yet."

---

## 4. Defect ledger

| id | defect | status |
| :--- | :--- | :--- |
| M-5 | Inference never ran — dead `SupportSetManager`, unused registry | ✅ fixed |
| M-6 | Open-set rejection dead (`_weibull_by_name` empty) **and** undistanceable | ✅ replaced by the physical envelope gate; OpenMax itself resolved as *declared* dead code — §5.1 |
| M-7 | Heuristic could not emit consumer classes, never said UNKNOWN | ✅ fixed |
| M-8 | Label loop doubly broken (embedding-as-window; registry never read) | ✅ fixed, verified end-to-end |
| — | Low-confidence promotion at `run_pipeline.py:820` | ✅ removed |
| — | `allowed_classes` never passed from config | ✅ fixed |
| — | `test_centroid_path_used_when_fitted` regression | ✅ resolved — §1 |

---

## 5. Closed out after §1

### 5.1 OpenMax: resolved as *declared* dead code, not repaired

The prior handoff called this "misleading dead code in the critical path" and prescribed
making `train_demo_models.py` populate `_weibull_by_name`. Both halves of that framing
turned out to be wrong, and the prescription would have made things worse.

**It is not in the critical path — it is unreachable.** The chain:

```
compute_open_set_prob                              (src/models/protonet.py:395)
  ← SupportSetManager.classify                     (src/models/protonet.py:715, sole caller)
    ← run_pipeline.py:549                           (sole caller)
      guarded by  if not self.support_manager.raw_windows:   (line 545)
        raw_windows filled only by load_registry(anchors_path)   (line 409)
          anchors_path named in config/config.yaml ONLY
            → backend/models/weights/protonet_anchors.pt  ← DOES NOT EXIST
```

So the guard never opens on any of the three profiles. Verified: `find` turns up no
anchors artifact anywhere in the repo.

**And it is inert even if reached.** The indexed `fit(idx, d2)` API used by the trainer
writes `_weibull[idx]` only; the consumer reads `_weibull_by_name`. Measured on the
shipped pickles:

| artifact | index tails | named tails |
| :--- | ---: | ---: |
| `backend/models/weights/openmax_weibull.pkl` | 10 | **0** |
| `backend/models/weights_demo/openmax_weibull.pkl` | 7 | **0** |

`compute_open_set_prob` therefore returns `0.0` for every window, and its caller tests
`open_set > (1 - confidence_threshold)` — so `0.0` reads as *"definitely known"*. **The
path fails open, not closed.**

**Why the prescribed repair was rejected.** Two independent reasons:

1. **It cannot work.** Per §2.2, the embedding is power-scale-blind: a 500 W washing
   machine lands 1.02 from the 5 W `router` prototype. A Weibull tail over those
   distances cannot separate novel from known however correctly it is fitted.
2. **The obvious version is actively wrong.** `train_demo_models.py:178` fits its tails
   on **squared** L2 (`torch.sum((emb - proto) ** 2)`), while `compute_open_set_prob`
   queries **plain** L2 (`np.linalg.norm(e - proto)`). Populating the named tails from
   the distances the trainer already has would calibrate the tail on a different metric
   than the query and silently mis-score every window — a worse failure than returning
   0.0, because it would look like it was working.

**What was changed instead** — make the deadness impossible to miss or to relapse:

* `scripts/run_pipeline.py` startup banner: was `OpenMax: ✅` computed off `_weibull`,
  i.e. advertising a reject channel that answers 0.0 for everything. Now reports
  `N tail(s), M named — INACTIVE (physical envelope gate is the reject channel)`.
  Verified live on the demo profile: `OpenMax: 7 tail(s), 0 named — INACTIVE (…)`.
* `OpenMaxWeibull.compute_open_set_prob` docstring: records the fail-open semantics of
  `0.0`, that the named dict is what it reads, and the two-part specification any real
  repair must satisfy (name + prototype at fit time, *and* a matching metric).
* `SupportSetManager.classify` docstring: records the unreachability chain above, so the
  next reader does not spend the diagnosis again.
* `scripts/train_demo_models.py`: comment at the fit call stating that it writes a
  half-populated artifact deliberately, and why repairing it there would mis-calibrate.
* `tests/test_ml_pipeline_recognition.py::TestOpenMaxIsNotTheRejectChannel` — 5 tests
  pinning: named tails absent; `compute_open_set_prob` fails open when unfitted; the
  2-dict API *does* populate and *does* score a far embedding > 0.5 (so the gap is the
  wiring, not the maths); no profile ships a readable anchors file; and a novel 2000 W
  load is rejected with the open-set channel provably inert.

**Still a product call, recorded in `debug_status.md` §5.3:** delete `OpenMaxWeibull` and
the `OpenMaxStage` export, or fund a real open-set channel on absolute watts. Note that
these tests are *new coverage*, not before/after regressions — the code change here is
the banner line plus documentation, so there is no pre-fix failure to demonstrate. Said
plainly rather than dressed up as a proof.

### 5.2 Everything else from the previous list

| item | status |
| :--- | :--- |
| Resolve the failing test | ✅ §1 — plus 2 tests added so the behaviour cannot drift either way |
| `graphify update .` | ✅ run after the last code edit: 2,513 nodes, 4,958 edges, 207 communities |
| Regression tests for each fix | ✅ `tests/test_ml_pipeline_recognition.py`, **35 passing** |
| `ARCHITECTURE_AND_APIS.md` + CLAUDE.md §4 anti-hallucination table | ✅ updated with `plausible_classes`, `power_envelope`, `_eligible_classes`, `_registry_heuristic`, `_classify_heuristic`, `recognition_threshold`, `_extra_centroids`, and the `LABEL_REQUEST.segments` contract |
| `recognition_threshold` stated in config | ✅ explicit in all three profiles with `heuristic_min_confidence`, each carrying the rationale for why it is *not* `confidence_threshold` |
| Retraining | ❌ **not done, and recommended against** — see §2.5. UK-DALE has no modern USB-PD charger and laptop/monitor score 0.501 leave-one-meter-out on 3 classes (chance 0.333). The operator's own few-shot labels are the correct path |

### 5.3 Verified state at close

```
python -m pytest tests/ -q                      # 511 passed (474 before this work)
python -m pytest tests/test_ml_pipeline_recognition.py -q   # 35 passed
python -m pytest tests/test_real_data_and_ml_fallback.py -q # 36 passed
python scripts/real_world_physical_stress.py    # 7/7 PASS
python scripts/hil_hardware_test.py             # 10/10 PASS
python scripts/test_firmware_and_ai_e2e.py      # 8/8 PASS
python scripts/stress_test_hardware_sim.py      # 7/7 PASS
graphify update .                               # 2513 nodes, 4958 edges, 207 communities
```

All work remains **uncommitted on `main`**. Nothing has been pushed.


---

## 6. Reproduction commands

```bash
source .venv/bin/activate

# the ML regression suite (M-5…M-8 + the OpenMax deadness contract)
python -m pytest tests/test_ml_pipeline_recognition.py -v

# full suite (expect 511 passed)
python -m pytest tests/ -q
```

Diagnostic scripts live in `/home/pramodsb/.claude/jobs/944dafbc/tmp/` — **these are in a
job temp dir and will be deleted when the job is removed.** Copy them out if they are
worth keeping:

| script | what it measures |
| :--- | :--- |
| `diag_ml.py` | initial "does anything classify at all" probe |
| `calib.py` | in-distribution vs novel d2 distributions (§2.2) |
| `separability.py` | leave-one-meter-out separability + data provenance (§2.5) |
| `measure_conf.py` | confidence distribution across temperatures (§2.3) |
| `measure_gate.py` | gate effect on accuracy, threshold sweep (§2.3) |
| `measure_agree.py` | two-channel agreement rule (§2.4) |
| `e2e.py` | end-to-end through the real `_classify_device` (§2.6) |
| `e2e_label.py` | full label loop incl. API validation (§2.6) |

Note: `FullPipeline` is an alias for `EMSOrchestrator`, whose `__init__` takes
`config=<dict>`, **not** `config_path=`. The diagnostic scripts load the YAML themselves.

---

# RESUMING FROM HERE

This log is **closed**. The blocking test is fixed, regression tests are in, the graph is
synced, and every doc listed in §5.2 is current. Do not redo the diagnostics in §2 — they
are expensive and their results are recorded above.

## State

- **511 passed**, 0 failed. All four harnesses green (7/7, 10/10, 8/8, 7/7).
- Everything is **uncommitted on `main`**. Nothing has been pushed.
- The user's requirement is met: phone / laptop / monitor are recognised where the data
  supports it, anything else is reported **unrecognised** and routed to a label loop that
  is verified end-to-end (enrolled-recall 3/3, operator labels outrank shipped classes).

## What is genuinely left, highest value first

1. **Commit.** Nothing from either 2026-08-25 pass is committed.
2. **H-4** (`debug_status.md` §5.2) — the Core-1 lockout latch in
   `firmware/esp32_node/src/main.cpp`. The only remaining item that is a *real hardware*
   safety gap rather than a simulation-fidelity or test-hygiene one.
3. **M-2** (§5.1 there) — wire `OverlapAwareNILMDetector` in with `overlap_window_s > 5.0`
   or delete it; and the parallel call on `OpenMaxWeibull` (§5.1 here): delete it or fund a
   real open-set channel on absolute watts.
4. **Test hygiene** (§5.4 there) — 27 bare `assert True`, 39 hallucinated `.process()`
   no-ops, and the `ImportError → MagicMock` masking that would let a real import break
   pass as green.
5. **Bench validation.** Everything here is simulation. `REAL_WORLD_TESTING_PLAN.md`.

## Two things NOT to do

- **Do not retrain** to "fix" phone/laptop/monitor. §2.5 measured why it cannot work:
  UK-DALE's `phone_charger` is a ~5 W trickle charger with 2 usable windows from one
  meter, `router` has none above the 20 W floor, and laptop/monitor score 0.501
  leave-one-meter-out on 3 classes (chance 0.333). It is a data-collection problem. The
  label loop is the remedy and it works.
- **Do not "repair" OpenMax** by populating `_weibull_by_name` from the trainer's existing
  distances. §5.1 measures why that mis-calibrates (squared L2 fitted, plain L2 queried)
  and why the channel cannot work here regardless.

