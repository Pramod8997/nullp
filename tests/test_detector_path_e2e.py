"""
Detector-path end-to-end tests — plug-in/unplug events THROUGH push().

Companion to test_e2e_five_class_recognition.py, which feeds steady windows
directly to `_classify_device` and therefore validates gate plumbing only.
Everything here routes samples through the real ingest path the rig uses
(`process_raw_mqtt` → `_handle_mqtt_message` → `NILMTransientDetector.push()`
→ `_classify_device`), with the plain-float payload format the firmware
publishes on home/sensor/{id}/power (main.cpp: "plain float string").

Why this file exists (the 2026-09-10 verification audit, findings C1/C2/C9):

  * At a plug-in transient, push() returns the last 128 samples ENDING at the
    detection sample — 97-99% PRE-event data. On an IDLE socket that is
    harmless: steady_w (median of samples >20 W) is computed over the few
    post-event samples, so the new load is classified correctly. On a LOADED
    socket it names the OLD load (pinned below as C1).
  * Quiet realistic loads (sigma ~2 W, like real hardware) fire the detector
    ONCE at the step, so the skew is not masked the way the sigma>=15 W
    simulator fleet masks it.

Honest success criteria for this path (per the audit's C6 verdict):

  * Confidence ~1.0 on these windows is an ARTIFACT of the single-survivor
    renormalisation in `_classify_device` — it is NOT P(correct) and must not
    be read as one. The meaningful gates asserted here are (a) band-correct
    classification of the load that actually changed, and (b) zero
    confident-wrong events (no DEVICE_STATUS confidently naming a class the
    socket never drew).

Pinned known limitations (asserted as current truth, not as desired behaviour):

  * C1 loaded-socket plug-in: the OLD load is named at conf ~1.0.
  * C9 unplug: the detector fires on the negative step (|dP/dt| threshold)
    and a STALE classification of the removed load is re-broadcast — see
    TestUnplugSemantics for the measured behaviour and why it differs from
    the "no survivor -> no event" reasoning in the plan.

Hardware note: software-simulator path, not physical validation (CLAUDE.md).
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.run_pipeline import FullPipeline  # noqa: E402

DEMO_CONFIG = "config/config.demo.yaml"
DEMO_WEIGHTS = "backend/models/weights_demo"

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DEMO_WEIGHTS, "protonet.pt")),
    reason="demo ProtoNet artefact absent; run scripts/train_demo_models.py",
)

# Quiet realistic loads, as a calm bench socket sees them (sigma ~1.5-2 W —
# nothing like the sigma>=15 W simulator fleet, which re-fires the detector
# every 5 s and masks the pre-event window skew).
IDLE_W = 2.0        # quiet socket baseline
SIGMA_QUIET = 1.5
SIGMA_LOAD = 2.0
PRE_SAMPLES = 130   # ~130 s of quiet baseline -> a full 128-sample window
POST_SAMPLES = 12   # post-event ticks: the 5-tick CNN burst + margin

# Demo steady draws (W) for the three classified plug-in classes. The demo
# bulb is a 9 W trickle load that never crosses the 20 W transient threshold
# (phantom-channel territory), so it is not a plug-in class here.
PLUG_IN_CLASSES = {
    "phone": 45.0,
    "laptop": 120.0,
    "projector": 300.0,
}

# Deterministic per-scenario seeds (held fixed; assertions below are pinned
# against these exact windows, like the rest of the ML suite).
SEEDS = {"phone": 11, "laptop": 12, "projector": 13,
         "loaded": 21, "unplug": 31}


def _quiet(level: float, sigma: float, n: int, rng) -> np.ndarray:
    """Clamped-at-zero gaussian noise around `level` (PZEM never reads < 0)."""
    return np.maximum(0.0, rng.normal(level, sigma, n)).astype(np.float32)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """Demo-profile orchestrator on a writable copy of the demo weights.

    Same construction as test_e2e_five_class_recognition.py: registry_path
    resolves to the writable prototype_registry_enrolled.pt copy, so nothing
    here can mutate the checked-in artefact. `_broadcast_event` and the MQTT
    publish are stubbed to recorders — the classification path itself (push,
    _classify_device, gates, label loop) is fully real; only the network
    egress is inert. The DB is intentionally left unconnected (as in every
    offline test), and the CSV fallback is pointed at tmp_path so the
    'Database not running' fallback writes nowhere persistent.
    """
    weights_dir = tmp_path / "weights"
    shutil.copytree(DEMO_WEIGHTS, weights_dir)
    with open(DEMO_CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    cfg["protonet"]["weights_path"] = str(weights_dir / "protonet.pt")
    if "registry_path" in cfg["protonet"]:
        cfg["protonet"]["registry_path"] = str(
            weights_dir / os.path.basename(cfg["protonet"]["registry_path"])
        )
    cfg["database"]["fallback_csv"] = str(tmp_path / "fallback.csv")

    pipe = FullPipeline(config=cfg)

    events: list = []

    async def _record(event: dict) -> None:
        events.append(event)

    async def _noop_publish(topic, payload, **kwargs) -> None:
        pass

    monkeypatch.setattr(pipe, "_broadcast_event", _record)
    monkeypatch.setattr(pipe.mqtt, "publish_command", _noop_publish)
    pipe._test_events = events  # the only test-only attribute
    return pipe


def _feed(pipe, device_id: str, watts) -> list:
    """Push 1 Hz samples through the real MQTT ingest path (plain-float
    payload, the firmware's format). Returns one row per tick with the
    pipeline's post-tick classification state."""
    rows = []

    async def _run():
        topic = f"home/sensor/{device_id}/power"
        for i, w in enumerate(watts):
            await pipe.process_raw_mqtt(topic, str(float(w)))
            rows.append({
                "i": i,
                "w": float(w),
                "class": pipe.device_classifications.get(device_id),
                "conf": pipe.last_known_confidences.get(device_id),
                # >0 means a transient fired this tick or within the last 5
                "cnn": pipe.cnn_active_ticks.get(device_id, 0),
            })

    asyncio.run(_run())
    return rows


def _classified_statuses(pipe, device_id: str) -> list:
    """DEVICE_STATUS broadcasts that carry a real classification."""
    return [e for e in pipe._test_events
            if e.get("device_id") == device_id
            and e.get("type") == "DEVICE_STATUS"
            and e.get("classification") not in (None, "pending")]


# ══════════════════════════════════════════════════════════════════════════
# Idle-socket plug-ins: the sequential path that IS proven correct
# ══════════════════════════════════════════════════════════════════════════

class TestIdleSocketPlugin:
    """A quiet socket (2 W noise), then a device plugged in and running.

    The trigger window is 97-99% pre-event, but on an idle socket that is
    harmless: steady_w (median of samples >20 W, heuristic_fallback.
    extract_features) is decided by the handful of post-event samples, so
    the NEW load is band-correct.
    """

    @pytest.mark.parametrize("cls", sorted(PLUG_IN_CLASSES))
    def test_idle_socket_plugin_is_band_correct(self, pipeline, cls):
        device_id = f"node_{cls}"
        rng = np.random.default_rng(SEEDS[cls])
        watts = np.concatenate([
            _quiet(IDLE_W, SIGMA_QUIET, PRE_SAMPLES, rng),
            _quiet(PLUG_IN_CLASSES[cls], SIGMA_LOAD, POST_SAMPLES, rng),
        ])
        rows = _feed(pipeline, device_id, watts)

        # The detector actually fired — the transient path was exercised,
        # not the steady-window bypass.
        assert any(r["cnn"] > 0 for r in rows), \
            "no transient fired: this test is not on the detector path"

        # Before the plug-in, no classification exists at all (the CNN only
        # runs on/after a transient — the §2.1 fix).
        assert all(r["class"] in (None, "pending") for r in rows[:PRE_SAMPLES])

        # Correct within the event burst (trigger tick or the 5 post-transient
        # ticks — tolerant of which tick, not of the answer). The end state is
        # the last burst classification; carried over on the trailing
        # no-transient ticks.
        assert pipeline.device_classifications[device_id] == cls, \
            f"idle-socket {cls} plug-in classified as " \
            f"{pipeline.device_classifications[device_id]!r}"

        # Every classified DEVICE_STATUS in the burst names the right class
        # at a confident level — zero confident-wrong on this path.
        statuses = _classified_statuses(pipeline, device_id)
        assert statuses, "no DEVICE_STATUS with a classification was broadcast"
        assert all(s["classification"] == cls for s in statuses), \
            [s["classification"] for s in statuses]
        assert all(s["confidence"] >= pipeline.recognition_threshold
                   for s in statuses)

        # The device is ON by the end of the burst.
        assert pipeline.device_states[device_id] == 1


# ══════════════════════════════════════════════════════════════════════════
# Loaded-socket plug-in: C1 pinned as the pre-Run-3 baseline
# ══════════════════════════════════════════════════════════════════════════

class TestLoadedSocketPlugin:
    def test_loaded_socket_plugin_names_the_added_load_run3(self, pipeline):
        """Run 3 LANDED (2026-09-10) — the C1 known limitation is FIXED.

        A 60 W load is already running on the socket; a phone (45 W) is then
        plugged in (steady 105 W). Historically (pre-Run-3, audit finding C1)
        the trigger window push() returns is 97-99% PRE-event, so steady_w =
        the OLD 60 W level → "bulb" @ conf 1.0, and the phone was never
        named. The delta-window overlap layer (preprocessing.delta_overlap,
        default ON) now arms on non-idle baselines: steady-after minus
        steady-before = +45 W → "phone", classified through the same
        envelope gate via a synthetic constant-delta segment.

        This is the FLIP of the old C1 pin, per the instruction that lived
        in this docstring: the test previously pinned "bulb" with a note
        that a failure naming the added load meant Run 3 had landed.
        """
        device_id = "node_loaded"
        rng = np.random.default_rng(SEEDS["loaded"])
        watts = np.concatenate([
            _quiet(60.0, SIGMA_QUIET, PRE_SAMPLES, rng),
            _quiet(105.0, SIGMA_LOAD, POST_SAMPLES, rng),  # 60 + 45 W phone
        ])
        rows = _feed(pipeline, device_id, watts)

        # The plug-in transient fired (this is the detector path, not the
        # steady-window bypass).
        assert any(r["cnn"] > 0 for r in rows)

        # Run 3 behaviour: the ADDED load is named via the delta window.
        assert pipeline.device_classifications[device_id] == "phone", \
            f"Run 3 delta pin drifted: expected 'phone' (the +45 W delta), " \
            f"got {pipeline.device_classifications[device_id]!r}"

        statuses = _classified_statuses(pipeline, device_id)
        assert statuses, "no DEVICE_STATUS with a classification was broadcast"
        # The added phone is named, at full confidence (single-envelope
        # survivor artifact), on every burst event.
        assert all(s["classification"] == "phone" for s in statuses), \
            [s["classification"] for s in statuses]
        assert all(s["confidence"] >= pipeline.recognition_threshold
                   for s in statuses)
        assert "bulb" not in {s["classification"] for s in statuses}, \
            "loaded-socket plug-in regressed to naming the OLD load — the " \
            "delta window (Run 3) stopped working for this case"


# ══════════════════════════════════════════════════════════════════════════
# Unplug semantics: C9, as measured
# ══════════════════════════════════════════════════════════════════════════

class TestUnplugSemantics:
    def test_unplug_fires_detector_no_stale_classification_run3(self, pipeline):
        """C9 — unplugs, as the code actually behaves (measured, not assumed).

        A 120 W laptop is running; it is unplugged (socket drops to ~2 W).

        1. The detector DOES fire on the negative step: push() thresholds
           |dP/dt| (np.abs(recent) >= scaled_threshold,
           src/pipeline/aggregate_nilm.py), so plug-in and unplug are
           indistinguishable at the detector.

        2. Pre-Run-3 the trigger window (97-99% PRE-event) made steady_w =
           the pre-unplug ~120 W level, so the removed load was (re)classified
           as "laptop" at conf ~1.0 — a stale verdict broadcast over ~2 W
           readings. The Run 3 delta layer FIXED this: the unplug's delta is
           negative (~-118 W), below the 20 W on-threshold, so NO
           classification event is emitted and no stale label is broadcast.
           This is the FLIP of the old stale-label pin, per the instruction
           that lived in this docstring.

        The honest live observables of the unplug: the detector fires,
        device_states (ON -> OFF), the falling wattage, and the phantom
        channel picking up the residual ~2 W floor.
        """
        device_id = "node_unplug"
        rng = np.random.default_rng(SEEDS["unplug"])
        watts = np.concatenate([
            _quiet(120.0, SIGMA_QUIET, PRE_SAMPLES, rng),
            _quiet(IDLE_W, SIGMA_QUIET, POST_SAMPLES, rng),
        ])
        rows = _feed(pipeline, device_id, watts)

        # 1. The negative step fires the detector (|dP/dt| threshold).
        assert any(r["cnn"] > 0 for r in rows[PRE_SAMPLES:]), \
            "unplug did not fire the transient detector"

        # 2. Run 3 behaviour: NO classification is emitted for the unplug —
        #    the device never leaves its pre-event "pending" state (it was
        #    never classified while running), and no DEVICE_STATUS carries
        #    a classification of the removed load.
        assert pipeline.device_classifications.get(device_id) in (None, "pending"), \
            f"unplug classification drifted: expected no classification, " \
            f"got {pipeline.device_classifications.get(device_id)!r}"
        statuses = _classified_statuses(pipeline, device_id)
        assert not statuses, \
            f"stale classification broadcast after unplug: {statuses}"

        # 3. The honest observables: state OFF, no label-loop traffic.
        assert pipeline.device_states[device_id] == 0
        label_reqs = [e for e in pipeline._test_events
                      if e.get("device_id") == device_id
                      and e.get("type") == "LABEL_REQUEST"]
        assert not label_reqs, \
            "unplug unexpectedly routed to the label loop"

        # 4. The residual ~2 W socket noise floor (is_off=True, 0 < w <= 15 W
        #    baseline) is picked up by the phantom channel as a small EMA —
        #    the laptop itself is simply gone from every channel.
        ema = pipeline.phantom_tracker.phantom_loads.get(device_id)
        assert ema is not None and 0.0 < ema <= pipeline.phantom_tracker.baseline_threshold
