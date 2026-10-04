# ML Findings — Agent 4 Audit

**Audit date:** 2026-10-04  
**Scope:** enrollment/registry provenance, phone/laptop/projector mapping, power
envelopes, held-out validation, unknown rejection, partial checkpoints,
delta/overlap behavior, and active appliance state.  
**Physical validation:** **NOT RUN**  
**Training/retraining:** not run.

## Verdict

The simulator recognition and label-loop paths are operational on the shipped
demo artifacts. The physical ML gate is **BLOCKED**: no attested physical
captures or physical registry are present, the hardware profile has no explicit
physical `registry_path`, and the current loader accepts partial ProtoNet
checkpoints with `strict=False`. Unknown rejection is proven for out-of-envelope
simulator loads, but not for physical loads or unknowns inside an enrolled power
envelope. Aggregate overlap classification has a working simulator delta path,
but persistent active-appliance state is still one classification per aggregate
node rather than an active set.

## Verification run

All commands used the matching project runtime:
`PYTHONPATH=venv/lib/python3.10/site-packages:. /usr/bin/python3.10`.

| Command | Result | Evidence boundary |
|---|---:|---|
| `-m pytest tests/test_ml_pipeline_recognition.py -q --tb=short` | **52 passed** | Shipped/demo artifacts; direct classifier and label-loop paths |
| `-m pytest tests/test_detector_path_e2e.py tests/test_overlap_delta.py -q --tb=short` | **20 passed** | Software ingest/detector/delta path; simulator-only |
| `-m pytest tests/test_real_data_and_ml_fallback.py -q --tb=short` | **36 passed** | Fallback and cached UK-DALE/REDD contracts; no physical data |
| `-m pytest tests/test_e2e_five_class_recognition.py -q --tb=short` | **20 passed** | Demo steady-window gate plumbing; simulator-only |

No source, test, or configuration file was changed by this audit.

## Verified software findings

### 1. Physical profile does not load a physical enrollment registry

`config/config.hardware.yaml` declares the correct current physical class list:

```yaml
appliances: [phone, laptop, projector]
```

It has no `protonet.registry_path` (`config/config.hardware.yaml:48-77`).
`EMSOrchestrator._resolve_registry_path()` therefore falls back to a file next
to `weights_path` (`scripts/run_pipeline.py:364-389`). The live trace resolved
the hardware profile to:

```text
backend/models/weights_demo/prototype_registry.pt
```

That registry contains seven UK-DALE/demo classes and **zero power envelopes**:

```text
desktop_computer, laptop, monitor, phone_charger, projector, router, tv
envelopes: {}
```

It has no `phone` class; the historical name is `phone_charger`. Direct traces
with the hardware profile returned `unknown` for 45 W, 120 W, 300 W, and 800 W
windows. The profile comments accurately describe this as not-yet-enrolled,
but it remains a release blocker for physical recognition.

### 2. The shipped enrolled registry is simulator/demo data, not physical data

The checked-in/available demo artifact
`backend/models/weights_demo/prototype_registry_enrolled.pt` contains ten
classes and six envelopes. The observed envelope values are:

| Class | Observed envelope (W) |
|---|---:|
| `phone` | 44.19–47.88 |
| `laptop` | 116.73–122.34 |
| `projector` | 297.39–301.87 |
| `bulb` | 59.48–60.37 |
| `fan` | 74.22–75.56 |
| `desktop_computer` | 245.43–253.28 |

The remaining registry classes (`monitor`, `phone_charger`, `router`, `tv`) have
no envelope. The artifact payload contains prototype entries and the reserved
`__power_envelopes__` entry, but no capture source, operator attestation,
hardware identity, timestamp, or physical-measurement provenance.

`scripts/enroll_demo_devices.py` explicitly uses the simulator distribution by
default (`sim_windows`, seeds beginning at 100) and prints
`simulated — NOT physical validation` (`scripts/enroll_demo_devices.py:33-45,
203-207`). This makes the demo artifact valid simulator evidence only.

### 3. Capture provenance is not carried into the registry artifact

`scripts/capture_bench_windows.py` can write an `.npz` provenance block with
MQTT topic, broker, device ID, timestamps, rate, operator/host, and an optional
`--attest-physical` note. Without that option it marks the capture
`UNATTESTED` (`scripts/capture_bench_windows.py:182-189,234-252`).

`scripts/enroll_demo_devices.py --capture` reads and prints this provenance,
but `PrototypeRegistry.save()` writes only prototypes and
`__power_envelopes__` (`src/models/protonet.py:649-665`). The provenance is not
copied into the registry. Therefore a registry file alone cannot establish that
its classes came from a real PZEM capture.

The inspected `data/real/cache/*.npz` files contain class arrays and
`__groups__*` dataset grouping tags, not physical-rig attestation. No bench
capture artifact for phone/laptop/projector was present in the inspected repo
paths.

### 4. Class mapping is correct for the physical profile but inconsistent in demo tooling

The current physical contract is `phone`, `laptop`, `projector`; the LED bulb
is below the 20 W transient floor and is phantom-tracked, and fan is not a
physical target. `config/config.hardware.yaml` and
`scripts/capture_bench_windows.py` accept this physical `phone` name.

The demo artifact and tests still contain simulator-only extras. In particular:

* `config/config.demo.yaml` scopes the demo to `phone`, `laptop`, `bulb`, and
  `projector`.
* The enrolled demo registry also contains `fan`, `desktop_computer`, and the
  historical UK-DALE classes.
* `scripts/enroll_demo_devices.py` still defines
  `REQUIRED_CLASSES = ("phone", "laptop", "bulb", "projector", "fan")` and
  `TARGET_CLASSES` adds `desktop_computer` (`:57-80`).
* Enrolled classes are exempt from the profile `appliances:` filter in
  `_eligible_classes()` (`scripts/run_pipeline.py:841-851`), so a simulator
  `fan` can still be named even though it is absent from the demo scope.

This does not prove a physical class failure, but it means the demo registry
must not be used as evidence for the three-class physical gate.

### 5. Envelope computation and enrollment behavior

`PrototypeRegistry.add_class()` embeds the supplied raw `(K,128)` watt windows,
computes the median of samples above 20 W per segment, and stores the min/max
observed steady values; repeated enrollment merges/widens the envelope
(`src/models/protonet.py:568-618`). This is the correct watts-not-embedding
contract, and the label handler rejects non-finite, negative, sub-20 W, and
wrong-length segments (`scripts/run_pipeline.py:1644-1684`).

The enrollment script does not create a train/validation split or calculate a
held-out metric. Every supplied capture window is used for `add_class()` and
then saved. The default capture method also uses sliding windows with a 16 s
stride over 128 s windows, so adjacent windows are correlated
(`scripts/capture_bench_windows.py:15-20`).

The registry save/load path is otherwise verified: the focused ML tests pass
the registry path round trip and preserve envelopes without exposing the
reserved key as a class. The save implementation uses a same-directory temp
file plus `os.replace()`.

### 6. Held-out validation is simulator-only

The focused e2e tests use simulator Gaussian profiles matching
`backend/scripts/simulate_esp32.py`. The enrollment seeds are separated from
the test seeds in the test fixtures (test seeds `0..11`; enrollment begins at
100), and the held-out simulator windows classify correctly for the tested demo
classes. This is a valid software data-separation check for that generator.

It is not held-out physical validation. There is no physical train/hold-out
capture pair, no per-appliance physical confusion matrix, no meter/nameplate
identity, and no attested re-presentation trace for phone, laptop, or
projector.

### 7. Partial ProtoNet checkpoints do not fail closed

The loader remaps keys and calls:

```python
missing, unexpected = proto.load_state_dict(remapped, strict=False)
```

It logs missing/unexpected keys and retains the encoder
(`scripts/run_pipeline.py:397-439`). A read-only trace removed
`attention.attn.0.weight` from the shipped checkpoint and observed:

```text
missing_count 1, unexpected_count 0, continues True
```

Thus a partial state dict can reach inference with randomly initialized
parameters. The registry loader also accepts a payload with no envelopes and
only logs a warning in the orchestrator (`scripts/run_pipeline.py:441-483`);
there is no artifact schema, provenance, class-set, or envelope-integrity gate.
This directly blocks the M1 requirement that partial checkpoints and
non-enrolled physical deployments fail closed.

### 8. Unknown rejection is effective only for the proven envelope cases

The physical watts gate rejects simulator loads outside all known envelopes.
The focused tests passed for kettle/oven/hairdryer/microwave/heater/EV and
other out-of-family signatures, returning internal `unknown` with zero
confidence. The label-request path carries raw watts windows and the label loop
successfully enrolls a new simulator class.

The known limitation is also pinned by a passing test: a novel 120 W load inside
the padded enrolled laptop envelope is named `laptop`. Enrolled-class precedence
and survivor renormalization can produce approximately 1.0 confidence for a
single surviving envelope. That confidence is not calibrated probability of
identity. An in-band unknown rehearsal, such as a 45 W fan or other load inside
one of the target bands, has not been run physically.

OpenMax is not the active reject channel. The shipped Weibull artifact has
index-keyed tails but no named tails, and the focused tests verify that
`compute_open_set_prob()` is inert/fails open when called through the legacy
path. The live rejection mechanism is the absolute-watts envelope gate in
`_classify_device()` (`scripts/run_pipeline.py:713-796`).

### 9. Heuristic fallback is safe but not sufficient for physical enrollment

With `allowed_classes=["phone", "laptop", "projector"]`, the fallback trace
returned:

| Window | Fallback result |
|---:|---|
| 9 W | `unknown`, 0.000 — appropriate for phantom-load handling |
| 45 W | `laptop`, 0.210 — below the 0.55 fallback acceptance floor |
| 120 W | `laptop`, 0.355 — below the acceptance floor |
| 300 W | `projector`, 0.750 |

There is no fitted phone centroid in
`src/pipeline/heuristic_fallback.py`; modern USB-PD phone data is not
represented. Consequently the degraded path abstains on a typical 45 W phone
signature until a real phone capture is enrolled/fitted. This is safe behavior,
but it is not a completed three-class physical fallback gate.

### 10. Delta/overlap software path passes; the old overlap detector is not live

The running orchestrator uses `NILMTransientDetector.push()` plus the
feature-gated post-step delta path in `scripts/run_pipeline.py:1076-1258`.
`delta_overlap: true` is enabled in the demo and hardware profiles. The
separately exported `OverlapAwareNILMDetector` in
`src/pipeline/aggregate_nilm.py:188-318` is not instantiated by the running
orchestrator.

The focused software path passed 20 tests, including:

* idle-socket phone/laptop/projector plug-ins;
* five ordered aggregate pairs, where the added load is named through the
  delta;
* slow-ramp/stability and baseline handoff behavior;
* unplug behavior without a new classification;
* a band-gap delta routed to `unknown` and the label loop.

All of this is simulator/software evidence. It does not establish the required
physical ≥90% added-load gate or zero-confident-wrong overlap rehearsal.

### 11. Persistent active-appliance state is not maintained for one aggregate node

The live state is keyed by device ID:

```python
self.device_states: Dict[str, int]
self.device_classifications: Dict[str, str]
self.last_device_power: Dict[str, float]
```

On a one-node aggregate socket, the delta result replaces the node's prior
classification. A read-only trace of a 120 W laptop followed by a 45 W phone
ended with:

```text
device_states:          {'aggregate_trace': 1}
device_classifications: {'aggregate_trace': 'phone'}
last_power:             {'aggregate_trace': 164.86}
active_appliance_set:   absent
```

The running code has no `active_appliances` or `active_devices` set. It retains
event history and aggregate power, but not a persistent `{laptop, phone}` active
set with per-appliance contributions. The overlap tests assert that event
history contains the running and added labels; they do not prove persistent
active-set state. This is the software portion of the M3 blocker.

### 12. Label enrollment is not fully wired across separate API/pipeline processes

The legacy `/api/submit-label` endpoint publishes `home/ml/label`, which the
pipeline subscribes to. The newer
`/api/v1/appliances/label-unrecognized` endpoint directly uses its own lazy
`FullPipeline` and persists the registry, but does not publish `home/ml/label`
(`src/api/main.py:822-912`). In a split API/pipeline deployment, the running
pipeline will not receive that new label until it reloads the registry or the
label is sent through the MQTT path. The focused endpoint tests use the same
local process/artifact and therefore do not prove cross-process live update.

## Simulator-only evidence versus missing physical evidence

### Verified simulator/software evidence

* Demo registry loads with phone/laptop/projector envelopes.
* Held-out simulator seeds classify the enrolled demo signatures.
* Out-of-envelope simulator signatures reject and can enter the label loop.
* Raw watt windows, rather than embeddings, are validated and enrolled.
* Registry save/load preserves prototypes and envelopes.
* Detector-path delta classification handles the tested synthetic ordered pairs.
* Phantom routing remains separate for sub-20 W loads.

### Not evidenced

* Any real PZEM capture for phone, laptop, or projector.
* Physical enrollment into `prototype_registry_bench.pt` or another attested
  registry selected by `config.hardware.yaml`.
* Physical held-out validation, per-class recall, confusion matrix, or
  zero-confident-wrong result.
* Physical out-of-envelope and in-envelope unknown rehearsal.
* Physical overlap/delta ordered-pair result and ≥90% gate.
* Physical projector nameplate/ceiling decision and hardware-profile agreement.
* Physical relay, PZEM, mains, calibration, EMI, or instrument evidence.

## Exact gate blockers

| Gate | Status | Exact blocker |
|---|---|---|
| **M1 — physical enrollment/registry** | **BLOCKED** | No attested physical capture or physical registry; hardware profile has no `registry_path` and resolves to the seven-class, zero-envelope UK-DALE registry. |
| **M1 — checkpoint integrity** | **OPEN/BLOCKING** | Partial ProtoNet state dicts are accepted with `strict=False`; missing-key inference continues with initialized parameters. Registry payload/schema/provenance is not validated. |
| **M2 — unknown rejection** | **PARTIAL/BLOCKED for physical PASS** | Out-of-envelope simulator rejection passes; in-envelope unknowns can be named with ~1.0 survivor confidence. No physical unknown rehearsal exists. |
| **M3 — overlap/disaggregation** | **PARTIAL/BLOCKED for physical PASS** | Software delta path passes synthetic tests, but no physical ordered-pair gate exists and the live aggregate state stores one replaceable class rather than an active appliance set. |
| **Class/provenance contract** | **OPEN** | Demo enrollment still targets simulator-only bulb/fan/desktop extras and the registry has no provenance metadata; these artifacts cannot support the physical three-class claim. |
| **Live label propagation** | **OPEN** | `/api/v1/appliances/label-unrecognized` persists through a separate lazy pipeline but does not publish `home/ml/label` to the running pipeline process. |
| **Hardware dependency** | **BLOCKED** | Projector scope remains coupled to the unresolved authoritative hardware/250 W ceiling contradiction; physical projector enrollment cannot be promoted before that decision. |

## Final status

**ML software:** simulator/demo paths verified by targeted tests.  
**Physical ML recognition:** **NOT READY**.  
**PHYSICAL VALIDATION: NOT RUN.**
