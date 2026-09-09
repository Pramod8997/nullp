"""
Run 3 (WS-D) — delta-window overlap classification, end to end.

Everything here routes synthetic 1 Hz sequences through the REAL ingest path
(`process_raw_mqtt` → `_handle_mqtt_message` → `NILMTransientDetector.push()`
→ `_classify_device`), with quiet sigma ~1.5-2 W loads as a calm bench socket
sees them. Same construction pattern as tests/test_detector_path_e2e.py.

The defect this pins the fix for (audit C1, measured 10/10):

  At a plug-in transient, push() returns the last 128 samples ENDING at the
  detection sample — 97-99% PRE-event data. On a socket already carrying a
  load >= 20 W, the window's steady_w (median of samples > 20 W) is the OLD
  load's level, so the OLD load is named at confidence ~1.0 (60 W running +
  phone plugs in → "bulb" @ 1.000 — the 60 W level lands in the enrolled
  demo bulb envelope).

The fix under test (`preprocessing: delta_overlap: true` in the demo +
hardware profiles): on a NON-IDLE baseline the pipeline classifies the DELTA
(steady-after − steady-before) once post-event power has been stable for
`delta_stability_samples` samples, through the same envelope-gated
`_classify_device` path with the delta as the window's effective steady
level. Idle-socket plug-ins (baseline < 20 W) keep the proven sequential
window path, untouched underneath.

Documented decisions asserted here as behaviour, not aspiration:

  * Unplug / negative delta → NO classification event: the last known state
    is carried (the aggregate wattage and device_states are the honest
    observables; the classification string goes stale by design).
  * Delta in a band gap (no envelope survivor) → UNRECOGNISED → the existing
    label loop (LABEL_REQUEST), same semantics as an out-of-band window on
    the idle path. The buffered signature is the DELTA level (what was
    added), not the pre-event window (what was already running).
  * A delta below the 20 W on-threshold (e.g. the detector's 5 s cooldown
    re-firing the same step, whose measured delta is noise) carries the
    previous verdict rather than manufacturing a spurious unknown.
  * Envelope-overlap deltas (two enrolled bands containing the same delta)
    are decided by the embedding channel exactly as the window path decides
    them — out of scope for the demo pairs, which are single-survivor.

Confidence ~1.0 on these windows is the single-survivor renormalisation
artifact (audit C6), not P(correct); the gates asserted are band-correct
naming of the load that actually changed and zero confident-wrong events.

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
HARDWARE_CONFIG = "config/config.hardware.yaml"
DEMO_WEIGHTS = "backend/models/weights_demo"

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DEMO_WEIGHTS, "protonet.pt")),
    reason="demo ProtoNet artefact absent; run scripts/train_demo_models.py",
)

# Quiet loads, as a calm bench socket sees them (sigma ~1.5-2 W).
IDLE_W = 2.0
SIGMA_QUIET = 1.5
SIGMA_LOAD = 2.0
PRE_SAMPLES = 130        # idle/baseline history before the first event
SETTLE_SAMPLES = 25      # first load running alone before the second plugs in
POST_SAMPLES = 15        # post-event ticks: fire, 5 s cooldown re-fire,
                         # delta resolution at fire+9, trailing carry ticks

# Demo steady draws (W) — the enrolled demo registry's bands:
#   phone [44.2, 47.9], laptop [116.7, 122.3], projector [297.4, 301.9]
# (padded ±15% by _eligible_classes).
LEVELS = {"phone": 45.0, "laptop": 120.0, "projector": 300.0}

# Added-load deltas for the achievable ordered pairs. {+45, +120, +300} are
# single-survivor against the enrolled envelopes (margins >= 8 W vs ~2 W rms
# noise). 160 W is a true band gap: above laptop's padded hi (~140.7), below
# desktop_computer's padded lo (~207.4) — no envelope contains it.
DEAD_ZONE_DELTA = 160.0

SEEDS = {"idle": 11, "pair": 21, "unplug": 31, "dead": 41, "off": 51}


def _quiet(level: float, sigma: float, n: int, rng) -> np.ndarray:
    """Clamped-at-zero gaussian noise around `level` (PZEM never reads < 0)."""
    return np.maximum(0.0, rng.normal(level, sigma, n)).astype(np.float32)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """Demo-profile orchestrator on a writable copy of the demo weights.

    The demo profile carries `preprocessing: delta_overlap: true` (the flag
    this file tests); registry_path resolves to the writable
    prototype_registry_enrolled.pt copy, so nothing here can mutate the
    checked-in artefact. `_broadcast_event` and the MQTT publish are stubbed
    to recorders — the classification path (push, delta layer, gates, label
    loop) is fully real; only the network egress is inert. The DB stays
    unconnected (as in every offline test) and the CSV fallback is pointed
    at tmp_path.
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
    assert cfg["preprocessing"]["delta_overlap"] is True, \
        "demo profile must ship the delta layer ON — the fixture tests the flag"

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
                "method": pipe._classify_methods.get(device_id),
                # >0 means a transient fired this tick or within the burst
                "cnn": pipe.cnn_active_ticks.get(device_id, 0),
            })

    asyncio.run(_run())
    return rows


def _classified_statuses(pipe, device_id: str) -> list:
    """DEVICE_STATUS broadcasts that carry a real classification
    (not pending; 'unknown'/pseudo-classes ARE included)."""
    return [e for e in pipe._test_events
            if e.get("device_id") == device_id
            and e.get("type") == "DEVICE_STATUS"
            and e.get("classification") not in (None, "pending")]


def _label_requests(pipe, device_id: str) -> list:
    return [e for e in pipe._test_events
            if e.get("device_id") == device_id
            and e.get("type") == "LABEL_REQUEST"]


def _sequence(rng, *levels: float) -> np.ndarray:
    """Idle/baseline then one quiet segment per level: e.g.
    _sequence(rng, 120, 165) = 130 s idle, 25 s @120, 15 s @165."""
    parts = [_quiet(IDLE_W, SIGMA_QUIET, PRE_SAMPLES, rng)]
    for lvl in levels[:-1]:
        parts.append(_quiet(lvl, SIGMA_LOAD, SETTLE_SAMPLES, rng))
    parts.append(_quiet(levels[-1], SIGMA_LOAD, POST_SAMPLES, rng))
    return np.concatenate(parts)


# ══════════════════════════════════════════════════════════════════════════
# (a) Idle-socket plug-ins: the sequential window path, unchanged
# ══════════════════════════════════════════════════════════════════════════

class TestIdleSocketUnchanged:

    @pytest.mark.parametrize("cls", sorted(LEVELS))
    def test_idle_socket_plugin_uses_the_window_path(self, pipeline, cls):
        """Pre-event steady < 20 W never arms the delta layer: the plug-in is
        classified by the proven sequential window path, reported as
        method="window"."""
        device_id = f"node_{cls}"
        rng = np.random.default_rng(SEEDS["idle"])
        rows = _feed(pipeline, device_id, _sequence(rng, LEVELS[cls]))

        assert any(r["cnn"] > 0 for r in rows), \
            "no transient fired: this test is not on the detector path"
        assert pipeline._delta_pending.get(device_id) is None, \
            "idle socket armed the delta layer — the idle path must never arm"
        assert pipeline.device_classifications[device_id] == cls
        assert pipeline._classify_methods[device_id] == "window"

        statuses = _classified_statuses(pipeline, device_id)
        assert statuses, "no DEVICE_STATUS with a classification was broadcast"
        assert all(s["classification"] == cls for s in statuses), \
            [s["classification"] for s in statuses]
        assert all(s["method"] == "window" for s in statuses)
        assert all(s["confidence"] >= pipeline.recognition_threshold
                   for s in statuses)


# ══════════════════════════════════════════════════════════════════════════
# (b) Achievable ordered pairs: the ADDED load is named via the delta
# ══════════════════════════════════════════════════════════════════════════

# (added_class, running_class): the socket runs `running` at its demo level,
# then `added` plugs in on top. The delta {+45, +120, +300} is what the
# classifier must see and name as `added`.
PAIRS = [
    ("laptop", "projector"),    # +120 W onto 300 W
    ("phone", "projector"),     # +45 W onto 300 W
    ("phone", "laptop"),        # +45 W onto 120 W
    ("laptop", "phone"),        # +120 W onto 45 W
    ("projector", "laptop"),    # +300 W onto 120 W
]


class TestOrderedPairs:

    @pytest.mark.parametrize("added,running", PAIRS)
    def test_added_load_is_named_via_the_delta(self, pipeline, added, running):
        """The core-goal overlap case: a load plugs into a socket that is
        already running another load. The running load is first classified
        through the window path (it plugged into an idle socket), then the
        added load's plug-in must be classified as the ADDED class via the
        delta (steady-after − steady-before) — not the old load, which the
        pre-Run-3 window path named at confidence 1.0 (audit C1)."""
        device_id = f"node_{added}_on_{running}"
        rng = np.random.default_rng(int(SEEDS["pair"] + LEVELS[running]))
        total = LEVELS[running] + LEVELS[added]
        rows = _feed(pipeline, device_id, _sequence(rng, LEVELS[running], total))

        # Both transients actually fired (this is the detector path).
        assert any(r["cnn"] > 0 for r in rows[:PRE_SAMPLES + SETTLE_SAMPLES]), \
            "first plug-in did not fire the detector"
        assert any(r["cnn"] > 0 for r in rows[PRE_SAMPLES + SETTLE_SAMPLES:]), \
            "second (overlap) plug-in did not fire the detector"

        # The running load was named first, via the window path.
        first_classified = next(r for r in rows if r["class"] not in (None, "pending"))
        assert first_classified["class"] == running
        assert first_classified["method"] == "window"

        # The ADDED load is named via the delta, at full single-survivor
        # confidence, and it sticks (carried on the trailing ticks).
        assert pipeline.device_classifications[device_id] == added, \
            (f"overlap plug-in ({added} onto {running}, "
             f"{LEVELS[running]} -> {total} W) classified as "
             f"{pipeline.device_classifications[device_id]!r}")
        assert pipeline._classify_methods[device_id] == "delta"
        assert pipeline.last_known_confidences[device_id] >= \
            pipeline.recognition_threshold

        # Zero confident-wrong: no DEVICE_STATUS ever names a class other
        # than the running load (carried verdicts while the delta waits for
        # post-event stability) or the added load (the delta verdicts). The
        # C1 signature — a THIRD class read off the pre-event level, e.g.
        # "bulb" from a 60 W window — is gone.
        statuses = _classified_statuses(pipeline, device_id)
        assert statuses
        seen = {s["classification"] for s in statuses}
        assert seen <= {running, added}, sorted(seen)
        delta_statuses = [s for s in statuses if s["method"] == "delta"]
        assert delta_statuses, "no delta-method classification was broadcast"
        assert all(s["classification"] == added for s in delta_statuses), \
            [s["classification"] for s in delta_statuses]
        # The verdict sticks: the last classified broadcast is the added load.
        assert statuses[-1]["classification"] == added

        # The measured delta is what the classifier saw.
        assert not _label_requests(pipeline, device_id)


# ══════════════════════════════════════════════════════════════════════════
# (c) Unplug: no classification event
# ══════════════════════════════════════════════════════════════════════════

class TestUnplug:

    def test_unplug_from_120w_emits_no_classification(self, pipeline):
        """A 120 W load is running (never classified — it was there before
        the feed started, so no transient ever named it); it is unplugged
        (socket drops to ~2 W). The detector DOES fire on the negative step
        (|dP/dt| threshold), but the delta layer measures a NEGATIVE delta
        and emits NO classification: no class is ever assigned, no
        DEVICE_STATUS carries a classification, nothing routes to the label
        loop. (Pre-Run-3 the window path re-broadcast the removed load's
        level as a confident stale classification — audit C9.)"""
        device_id = "node_unplug"
        rng = np.random.default_rng(SEEDS["unplug"])
        watts = np.concatenate([
            _quiet(120.0, SIGMA_QUIET, PRE_SAMPLES, rng),
            _quiet(IDLE_W, SIGMA_QUIET, POST_SAMPLES, rng),
        ])
        rows = _feed(pipeline, device_id, watts)

        assert any(r["cnn"] > 0 for r in rows[PRE_SAMPLES:]), \
            "unplug did not fire the transient detector"

        # No classification event: the device was never classified and the
        # unplug must not classify it.
        assert pipeline.device_classifications.get(device_id) in (None, "pending")
        assert not _classified_statuses(pipeline, device_id), \
            "unplug broadcast a classification"
        assert not _label_requests(pipeline, device_id), \
            "unplug unexpectedly routed to the label loop"

    def test_unplug_after_classification_carries_the_last_verdict(self, pipeline):
        """The sequential demo shape: phone plugs into an idle socket (named
        'phone'), then is unplugged. The unplug emits NO classification
        event — the last verdict is carried (stale by design: the honest
        observables are the wattage and device_states). No label traffic."""
        device_id = "node_unplug_after"
        rng = np.random.default_rng(SEEDS["unplug"] + 1)
        watts = np.concatenate([
            _quiet(IDLE_W, SIGMA_QUIET, PRE_SAMPLES, rng),
            _quiet(LEVELS["phone"], SIGMA_LOAD, SETTLE_SAMPLES, rng),
            _quiet(IDLE_W, SIGMA_QUIET, POST_SAMPLES, rng),
        ])
        rows = _feed(pipeline, device_id, watts)

        # Sanity: the phone WAS classified before the unplug.
        assert any(r["class"] == "phone" for r in rows), \
            "phone plug-in was never classified — test setup broken"

        # The unplug did not re-classify anything: the verdict stays the
        # pre-unplug one (carried), never re-derived from the pre-event
        # window, never routed to the label loop.
        assert pipeline.device_classifications[device_id] == "phone"
        assert not _label_requests(pipeline, device_id)

        # No post-unplug status confidently names a NEW class: every
        # classified status is the carried 'phone' verdict.
        statuses = _classified_statuses(pipeline, device_id)
        assert statuses
        post = [s for s in statuses if s["power"] <= 10.0]
        assert post, "no post-unplug DEVICE_STATUS to inspect"
        assert all(s["classification"] == "phone" for s in post), \
            [(s["power"], s["classification"]) for s in post]


# ══════════════════════════════════════════════════════════════════════════
# (d) Dead-zone delta: existing no-survivor semantics (label loop)
# ══════════════════════════════════════════════════════════════════════════

class TestDeadZoneDelta:

    def test_band_gap_delta_is_unrecognised_and_routes_to_the_label_loop(
            self, pipeline):
        """A +160 W load is added to a 60 W baseline (socket 60 -> 220 W).
        +160 W falls in no enrolled envelope (above laptop's padded
        ~140.7 W hi, below desktop_computer's padded ~207.4 W lo) and in no
        in-scope shipped band, so the delta classification has no survivor:
        UNRECOGNISED, and the EXISTING unknown flow takes over — repeated
        stable unknowns fire LABEL_REQUEST so the operator can name the
        added load. This is exactly the semantics an out-of-band window
        gets on the idle path; the delta layer only changes WHICH watts are
        gated (the added load's, not the socket total's).

        The buffered/enrollable signature is the DELTA level (~160 W flat),
        not the pre-event 60 W window and not the 220 W socket total — so a
        label enrolled from this event describes the load that was added."""
        device_id = "node_deadzone"
        rng = np.random.default_rng(SEEDS["dead"])
        total = 60.0 + DEAD_ZONE_DELTA
        watts = np.concatenate([
            _quiet(60.0, SIGMA_QUIET, PRE_SAMPLES, rng),
            _quiet(total, SIGMA_LOAD, POST_SAMPLES, rng),
        ])
        rows = _feed(pipeline, device_id, watts)

        assert any(r["cnn"] > 0 for r in rows), \
            "no transient fired: this test is not on the detector path"

        # No survivor -> the unknown family, produced by the delta method.
        assert pipeline._classify_methods[device_id] == "delta"
        final = pipeline.device_classifications[device_id]
        assert final == "unknown" or final.startswith("unknown_"), \
            f"band-gap delta classified as {final!r}, expected the unknown family"

        # Nothing was confidently named (zero confident-wrong).
        statuses = _classified_statuses(pipeline, device_id)
        assert statuses
        assert all(s["classification"] in ("unknown",)
                   or s["classification"].startswith("unknown_")
                   for s in statuses), \
            [s["classification"] for s in statuses]
        assert all(s["method"] == "delta" for s in statuses)

        # The existing label loop fires (the unknown flow, fed the delta
        # signature, reaches stability across the fire + cooldown re-fire).
        labels = _label_requests(pipeline, device_id)
        assert labels, "band-gap delta never fired LABEL_REQUEST"
        for evt in labels:
            assert evt["method"] == "delta"
            assert evt["segments"] and len(evt["segments"][0]) == 128
            seg_level = float(np.median(evt["segments"][0]))
            # The signature is the ADDED load's level (the delta), not the
            # 60 W pre-event window and not the 220 W socket total.
            assert DEAD_ZONE_DELTA * 0.8 <= seg_level <= DEAD_ZONE_DELTA * 1.2, \
                f"label signature level {seg_level:.1f} W is not the delta"

    def test_seventy_five_w_delta_lands_in_the_enrolled_fan_band(self, pipeline):
        """A delta solidly inside the enrolled `fan` band (~74.2-75.6 W,
        padded ~62.9-87.0 W) is honestly named `fan` at single-survivor
        confidence. The padded fan band OVERLAPS the padded bulb band
        (~50.4-69.5 W) in 62.9-69.5 W — a delta drawn there has TWO envelope
        survivors and resolves by prototype distance (the registry's known
        adjacent-band ambiguity). +75 W stays clear of the overlap even with
        σ=2 W sample noise, so this test pins the unambiguous case. (An
        earlier +70 W version of this test passed only by luck: its noisy
        delta landed in the overlap and the verdict was corrected by the
        detector's 5 s cooldown re-fire re-deriving the delta against the
        stale baseline — the baseline-handoff fix makes that re-derivation
        correctly see delta ≈ 0, so the first, ambiguous verdict now sticks.)"""
        device_id = "node_seventy"
        rng = np.random.default_rng(SEEDS["dead"] + 1)
        watts = np.concatenate([
            _quiet(60.0, SIGMA_QUIET, PRE_SAMPLES, rng),
            _quiet(135.0, SIGMA_LOAD, POST_SAMPLES, rng),  # 60 + 75
        ])
        _feed(pipeline, device_id, watts)

        assert pipeline.device_classifications[device_id] == "fan"
        assert pipeline._classify_methods[device_id] == "delta"
        assert pipeline.last_known_confidences[device_id] >= \
            pipeline.recognition_threshold


# ══════════════════════════════════════════════════════════════════════════
# (e) Flag OFF: the pre-Run-3 window behaviour (the fallback)
# ══════════════════════════════════════════════════════════════════════════

class TestFlagOff:

    def test_flag_off_restores_the_window_behaviour_on_loaded_sockets(
            self, tmp_path, monkeypatch):
        """`preprocessing: delta_overlap: false` is the fallback story: the
        C1 behaviour returns verbatim. A 60 W load runs, a phone plugs in
        (socket 60 -> 105 W); the window the transient returns is 97-99%
        pre-event, its steady_w is the 60 W OLD level, which lands in the
        enrolled demo bulb envelope — the OLD load ("bulb") is named at
        confidence ~1.0 and the phone is never named. This is the pin the
        Run 3 flip turns back on."""
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
        cfg["preprocessing"]["delta_overlap"] = False

        pipe = FullPipeline(config=cfg)
        events: list = []

        async def _record(event: dict) -> None:
            events.append(event)

        async def _noop_publish(topic, payload, **kwargs) -> None:
            pass

        monkeypatch.setattr(pipe, "_broadcast_event", _record)
        monkeypatch.setattr(pipe.mqtt, "publish_command", _noop_publish)
        pipe._test_events = events

        device_id = "node_flag_off"
        rng = np.random.default_rng(SEEDS["off"])
        watts = np.concatenate([
            _quiet(60.0, SIGMA_QUIET, PRE_SAMPLES, rng),
            _quiet(105.0, SIGMA_LOAD, POST_SAMPLES, rng),  # 60 + 45 W phone
        ])
        rows = _feed(pipe, device_id, watts)

        assert any(r["cnn"] > 0 for r in rows)
        assert pipe._delta_pending.get(device_id) is None, \
            "delta layer armed with the flag OFF"
        # The old, wrong verdict — pinned as the fallback.
        assert pipe.device_classifications[device_id] == "bulb"
        assert pipe._classify_methods[device_id] == "window"
        statuses = _classified_statuses(pipe, device_id)
        assert statuses
        assert all(s["classification"] == "bulb" for s in statuses)
        assert all(s["method"] == "window" for s in statuses)

    def test_demo_and_hardware_profiles_ship_the_flag_on(self):
        """The fallback is a config flip, so both live profiles must carry
        the flag explicitly ON."""
        for path in (DEMO_CONFIG, HARDWARE_CONFIG):
            with open(path) as fh:
                cfg = yaml.safe_load(fh)
            pre = cfg.get("preprocessing") or {}
            assert pre.get("delta_overlap") is True, \
                f"{path} must ship preprocessing.delta_overlap: true"

    def test_flag_defaults_off_when_absent(self, tmp_path, monkeypatch):
        """A profile without the key (e.g. the base household config.yaml
        contract) keeps the pre-Run-3 window behaviour: the delta layer is
        strictly opt-in per profile."""
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
        cfg["preprocessing"].pop("delta_overlap")

        pipe = FullPipeline(config=cfg)
        assert pipe._delta_overlap is False
