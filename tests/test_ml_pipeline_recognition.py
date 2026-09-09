"""
Regression tests for the ML recognition path (defects M-5 … M-8).

Each test here fails against the pre-fix code and passes after. The defects,
from `claude_debug/ML_PIPELINE_FIX_2026-08-25.md`:

  M-5  Inference never ran. `_classify_device` called `SupportSetManager`,
       which is only populated from `protonet.anchors_path` — set in
       config/config.yaml alone. On the demo and hardware profiles
       `compute_prototypes()` returned {} and EVERY event on EVERY device
       returned ("unknown", 0.0, {}). The trained 7-class PrototypeRegistry was
       loaded, logged at startup, written to by `handle_label_submitted`, and
       never once read.
  M-6  Open-set rejection could not work off embedding distance: novel-load
       distances land *inside* the known-class range. Replaced by a physical
       power-envelope gate on absolute watts.
  M-7  The heuristic channel could not emit the deployment's own classes and
       never abstained, so a low-confidence guess became the final answer.
  M-8  The label loop was doubly broken — LABEL_REQUEST shipped only a 128-D
       embedding, the dashboard POSTed it back as `segments`, and `add_class`
       ran the CNN over it as though it were watts. Both are length-128 float
       arrays, so nothing caught it.

These run against the real shipped demo artefact, not a mock: the defects were
all in how the real weights were (not) consulted, so a mocked registry would
reproduce none of them. Weights are copied to a tmpdir because
`handle_label_submitted` persists the registry, and enrolling a class must not
mutate the checked-in artefact.

Scope and honest criteria (2026-09-10): like the steady-window e2e, these
tests exercise the classification gates on windows handed straight to
`_classify_device` / the registry — they do not route plug-in events through
`NILMTransientDetector.push()`, so they say nothing about the transient path
(trigger-window skew, loaded-socket C1, unplug C9). That coverage lives in
tests/test_detector_path_e2e.py. A confidence of ~1.0 on an enrolled class is
the single-envelope-survivor renormalisation artifact, not P(correct); the
meaningful gates are band-correct classification of the drawn watts and zero
confidently-wrong answers.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest
import yaml

from scripts.run_pipeline import FullPipeline, UNRECOGNISED, UNRECOGNISED_DISPLAY

DEMO_CONFIG = "config/config.demo.yaml"
DEMO_WEIGHTS = "backend/models/weights_demo"
SEQ_LEN = 128

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DEMO_WEIGHTS, "protonet.pt")),
    reason="demo ProtoNet artefact absent; run scripts/train_demo_models.py",
)


def _configured_registry_path() -> str:
    """Whatever the demo profile actually points at, shipped default included."""
    with open(DEMO_CONFIG) as fh:
        proto = (yaml.safe_load(fh).get("protonet") or {})
    return proto.get(
        "registry_path",
        os.path.join(os.path.dirname(proto.get("weights_path", "")),
                     "prototype_registry.pt"),
    )

def _window(steady, peak=None, n=SEQ_LEN, edge_frac=0.25, jitter=0.0, seed=0):
    """An ON-transient power window in watts: idle, then a step to `steady`."""
    rng = np.random.default_rng(seed)
    w = np.zeros(n, dtype=np.float32)
    edge = int(edge_frac * n)
    w[edge:] = steady
    if peak is not None:
        w[edge] = peak
    if jitter:
        w[edge:] += rng.normal(0, jitter, n - edge).astype(np.float32)
    return np.maximum(w, 0.0)


@pytest.fixture
def pipeline(tmp_path):
    """Demo-profile orchestrator with a *writable copy* of the demo weights."""
    weights_dir = tmp_path / "weights"
    shutil.copytree(DEMO_WEIGHTS, weights_dir)
    with open(DEMO_CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    cfg["protonet"]["weights_path"] = str(weights_dir / "protonet.pt")
    # `registry_path` must be redirected too, or `handle_label_submitted` writes
    # its enrolments straight into the repo's own artifact and these tests stop
    # being isolated. copytree already brought every registry file along.
    if "registry_path" in cfg["protonet"]:
        cfg["protonet"]["registry_path"] = str(
            weights_dir / os.path.basename(cfg["protonet"]["registry_path"])
        )
    return FullPipeline(config=cfg)


def _enroll(pipeline, name, steady, peak=None, k=5):
    segs = [_window(steady, peak=peak, jitter=1.0, seed=s).tolist() for s in range(k)]
    pipeline.handle_label_submitted(name, segs)
    return segs


# ══════════════════════════════════════════════════════════════════════════
# M-5 — inference actually consults the registry
# ══════════════════════════════════════════════════════════════════════════

class TestInferenceReachesTheRegistry:

    def test_registry_is_loaded_and_support_manager_is_empty(self, pipeline):
        # The exact conditions of M-5: the registry holds the enrolled classes
        # (5 primary + desktop_computer from the demo fleet + the shipped
        # UK-DALE stand-ins) while SupportSetManager — what the old code
        # classified against — is empty. Both must hold, or this file is not
        # testing the defect.
        assert len(pipeline.prototype_registry.class_names()) == 10
        assert pipeline.support_manager.raw_windows == {}

    def test_distances_are_computed_for_every_registry_class(self, pipeline):
        # The old path returned ("unknown", 0.0, {}) — an EMPTY dist_map. A
        # populated map is proof the encoder ran and the prototypes were read.
        _, _, dists = pipeline._classify_device("dev", 300.0, _window(300, peak=330))
        assert set(dists) == set(pipeline.prototype_registry.class_names())
        assert all(np.isfinite(d) for d in dists.values())

    def test_an_in_band_load_is_actually_named(self, pipeline):
        # M-5's user-visible symptom: nothing was ever classified. A 300 W
        # desktop-class window must come back with a real class name.
        name, conf, _ = pipeline._classify_device("dev", 300.0, _window(300, peak=330))
        assert name not in (UNRECOGNISED, "pending", "error")
        assert name in pipeline.prototype_registry.class_names()
        assert conf >= pipeline.recognition_threshold

    def test_short_window_is_pending_not_misclassified(self, pipeline):
        # Buffering must be reported as such rather than answered from a
        # zero-padded window.
        name, conf, dists = pipeline._classify_device("fresh_dev", 120.0)
        assert (name, conf, dists) == ("pending", 0.0, {})


# ══════════════════════════════════════════════════════════════════════════
# M-6 — novel loads are rejected on physics, not on embedding distance
# ══════════════════════════════════════════════════════════════════════════

class TestOpenSetRejection:

    # Out-of-family for the demo profile's consumer-electronics class set.
    NOVEL = [("kettle", 2000), ("oven", 2500), ("hairdryer", 1200),
             ("microwave", 900), ("heater", 800), ("ev_charger", 7000)]

    @pytest.mark.parametrize("label,watts", NOVEL)
    def test_novel_load_is_unrecognised(self, pipeline, label, watts):
        name, conf, _ = pipeline._classify_device(
            "dev", float(watts), _window(watts, peak=watts * 1.05))
        assert name == UNRECOGNISED, f"{label} {watts} W masqueraded as {name}"
        assert conf == 0.0

    def test_embedding_distance_alone_cannot_separate_novel_from_known(self, pipeline):
        # This is why the reject decision is made on watts (M-6). If this
        # assertion ever inverts, a distance threshold has become viable and the
        # physical gate could be revisited — so it is pinned deliberately.
        _, _, novel = pipeline._classify_device("dev", 800.0, _window(800, peak=840))
        _, _, known = pipeline._classify_device("dev", 35.0, _window(35, peak=38))
        assert min(novel.values()) < max(known.values()), (
            "novel-load distance no longer sits inside the known-class range")

    def test_a_novel_load_still_has_a_nearest_class(self, pipeline):
        # The rejection is the gate's doing, not an absent prediction: the
        # learned channel does name a class, and it is wrong.
        _, _, dists = pipeline._classify_device("dev", 2000.0, _window(2000, peak=2100))
        assert dists, "expected distances even for a rejected window"
        assert min(dists, key=dists.get) in pipeline.prototype_registry.class_names()


# ══════════════════════════════════════════════════════════════════════════
# M-7 — the deployment's class list is honoured; the heuristic may abstain
# ══════════════════════════════════════════════════════════════════════════

class TestDeploymentScopingAndAbstention:

    def test_heuristic_scoped_to_configured_appliances(self, pipeline):
        # Previously the full rule set was always used, so the hardware profile
        # (laptop + phone_charger only) could report `hvac`.
        with open(DEMO_CONFIG) as fh:
            configured = set(yaml.safe_load(fh)["appliances"])
        assert pipeline.heuristic_clf.allowed_classes == configured
        assert {r.name for r in pipeline.heuristic_clf.rules} <= configured

    def test_household_class_cannot_be_reported_on_the_demo_profile(self, pipeline):
        for watts in (800, 1200, 2000, 2500):
            name, _, _ = pipeline._classify_device(
                "dev", float(watts), _window(watts, peak=watts * 1.05))
            assert name in (UNRECOGNISED, "pending"), (
                f"{watts} W reported {name!r}, outside the demo appliance list")

    def test_heuristic_fallback_abstains_below_its_own_floor(self, pipeline):
        # Degraded mode (no encoder / empty registry). A guess under
        # `heuristic_min_confidence` must be reported unrecognised, not promoted
        # to the device's final answer as the old low-confidence branch did.
        pipeline.encoder = None
        pipeline.prototype_registry = None
        name, conf, dists = pipeline._classify_device("dev", 65.0, _window(65, peak=72))
        assert dists == {}
        if name != UNRECOGNISED:
            assert conf >= pipeline.heuristic_min_confidence

    def test_recognition_threshold_is_not_the_confidence_gate(self, pipeline):
        # Two independent quantities: confidence_threshold (0.90) gates a single
        # softmax, recognition_threshold gates the noisy-OR of two channels that
        # have already agreed. Sharing a number would silently couple them.
        assert pipeline.recognition_threshold != pipeline.confidence_threshold
        assert 0.0 < pipeline.recognition_threshold < 1.0


# ══════════════════════════════════════════════════════════════════════════
# M-8 — the label loop
# ══════════════════════════════════════════════════════════════════════════

class TestLabelEnrollmentLoop:

    def test_unrecognised_load_becomes_recognised_after_labelling(self, pipeline):
        # The user's actual requirement: "if not recognized classify it as
        # unrecognized device and i will label it". A 95 W load sits in the gap
        # between the enrolled fan (floor 62.9 W) and laptop (floor ~99 W)
        # envelopes, so it is unrecognised, and must be recognised on the NEXT
        # event once labelled — proving inference reads the same registry
        # `handle_label_submitted` writes to.
        before, _, _ = pipeline._classify_device("dev", 95.0, _window(95, peak=100))
        assert before == UNRECOGNISED

        _enroll(pipeline, "my_laptop", 95, peak=100)

        after, conf, _ = pipeline._classify_device("dev", 95.0, _window(95, peak=100))
        assert after == "my_laptop"
        assert conf >= pipeline.recognition_threshold

    def test_enrollment_records_the_observed_power_envelope(self, pipeline):
        _enroll(pipeline, "my_laptop", 95, peak=100)
        env = pipeline.prototype_registry.power_envelope("my_laptop")
        assert env is not None
        lo, hi = env
        assert 90.0 <= lo <= hi <= 100.0

    def test_operator_label_outranks_a_shipped_class(self, pipeline):
        # Without enrolled-precedence the shipped `monitor` prototype sits
        # almost on top of a newly enrolled 35 W monitor, the probability halves
        # between them, and the device the operator just named still reports
        # unrecognised. Measured then: enrolled-recall 1/3.
        # NOTE: monitor is no longer an enrolled class in the 5-class registry,
        # so a 35 W window is now unrecognised rather than `monitor` — the
        # shipped-prototype precondition must hold only up to that.
        before = pipeline._classify_device("dev", 35.0, _window(35, peak=38))[0]
        assert before in ("monitor", UNRECOGNISED)

        _enroll(pipeline, "my_monitor", 35, peak=38)

        name, conf, _ = pipeline._classify_device("dev", 35.0, _window(35, peak=38))
        assert name == "my_monitor"
        assert conf >= pipeline.recognition_threshold

    def test_enrolled_envelope_does_not_swallow_a_neighbouring_load(self, pipeline):
        # The envelope pad is physical slack only. At 25% + 5 W an enrolled
        # 45 W charger spanned 33-57 W, captured a 35 W monitor, and — since
        # enrolled classes take precedence — discarded the correct answer.
        _enroll(pipeline, "my_charger", 45, peak=50)
        assert pipeline._classify_device(
            "dev", 45.0, _window(45, peak=50))[0] == "my_charger"
        assert pipeline._classify_device(
            "dev", 35.0, _window(35, peak=38))[0] != "my_charger"

    def test_enrolled_class_does_not_absorb_a_novel_high_power_load(self, pipeline):
        _enroll(pipeline, "my_laptop", 65, peak=72)
        for watts in (800, 2000):
            name, _, _ = pipeline._classify_device(
                "dev", float(watts), _window(watts, peak=watts * 1.05))
            assert name == UNRECOGNISED

    def test_relabelling_merges_the_envelope(self, pipeline):
        _enroll(pipeline, "my_laptop", 65, peak=72)
        lo1, hi1 = pipeline.prototype_registry.power_envelope("my_laptop")
        _enroll(pipeline, "my_laptop", 120, peak=132)
        lo2, hi2 = pipeline.prototype_registry.power_envelope("my_laptop")
        assert lo2 <= lo1 and hi2 > hi1
        assert hi2 >= 115.0

    def test_enrollment_survives_a_registry_save_load_round_trip(self, pipeline):
        # Envelopes ride in the same file as the prototypes under a reserved
        # key; a round trip must not lose them or leak the key as a class.
        _enroll(pipeline, "my_laptop", 65, peak=72)
        registry = pipeline.prototype_registry
        # Ask the pipeline where it saves rather than re-deriving it here: this
        # test hardcoded `weights_dir/prototype_registry.pt` and so silently
        # stopped following the save path once `registry_path` became
        # configurable — the same duplication that lost operator labels.
        path = pipeline._resolve_registry_path()
        assert os.path.exists(path)

        reloaded = type(registry)(registry.encoder)
        reloaded.load(path)
        assert "my_laptop" in reloaded.class_names()
        assert registry.ENVELOPE_KEY not in reloaded.class_names()
        assert reloaded.power_envelope("my_laptop") == pytest.approx(
            registry.power_envelope("my_laptop"))


class TestLabelPayloadIsPowerNotEmbedding:
    """
    The core of M-8. An embedding and a power window are both length-128 float
    arrays, so the shape check could never tell them apart — the dashboard
    submitted embeddings and `add_class` ran the CNN over them silently.
    """

    def test_handler_refuses_an_embedding(self, pipeline):
        before = set(pipeline.prototype_registry.class_names())
        embeddings = np.random.default_rng(0).normal(0, 1, (5, SEQ_LEN)).tolist()
        pipeline.handle_label_submitted("poisoned", embeddings)
        assert set(pipeline.prototype_registry.class_names()) == before

    def test_handler_refuses_nonfinite_segments(self, pipeline):
        before = set(pipeline.prototype_registry.class_names())
        bad = _window(65, peak=72).tolist()
        bad[10] = float("nan")
        pipeline.handle_label_submitted("nan_class", [bad])
        assert set(pipeline.prototype_registry.class_names()) == before

    def test_handler_refuses_a_sub_threshold_window(self, pipeline):
        # Never reaches the 20 W on-threshold, so no envelope could be measured
        # from it and the prototype would describe nothing.
        before = set(pipeline.prototype_registry.class_names())
        pipeline.handle_label_submitted("trickle", [_window(6, peak=8).tolist()])
        assert set(pipeline.prototype_registry.class_names()) == before

    def test_handler_refuses_the_wrong_length(self, pipeline):
        before = set(pipeline.prototype_registry.class_names())
        pipeline.handle_label_submitted("short", [[100.0] * 64])
        assert set(pipeline.prototype_registry.class_names()) == before

    def test_handler_accepts_real_watts(self, pipeline):
        pipeline.handle_label_submitted("my_laptop", [_window(65, peak=72).tolist()])
        assert "my_laptop" in pipeline.prototype_registry.class_names()

    def test_api_validator_rejects_an_embedding_and_accepts_watts(self):
        # Second layer: the same contract enforced at the HTTP boundary, so a
        # hand-rolled POST cannot bypass what the handler now refuses.
        pydantic = pytest.importorskip("pydantic")
        from src.api.main import LabelSubmission

        watts = _window(65, peak=72).tolist()
        ok = LabelSubmission(device_id="d", label="my_laptop", segments=[watts])
        assert len(ok.segments) == 1

        embedding = np.random.default_rng(0).normal(0, 1, SEQ_LEN).tolist()
        with pytest.raises(pydantic.ValidationError):
            LabelSubmission(device_id="d", label="x", segments=[embedding])

    def test_label_request_event_carries_power_windows(self):
        # The transport contract the dashboard reads. `segments` must survive
        # into the event model, or the dashboard has nothing but `embedding` to
        # submit — which is exactly how M-8 arose.
        from src.api.main import LabelRequestEvent

        watts = _window(45, peak=50).tolist()
        evt = LabelRequestEvent(
            type="LABEL_REQUEST", device_id="dev", power=45.0, confidence=0.0,
            segments=[watts], suggested_label=UNRECOGNISED_DISPLAY,
        )
        assert evt.segments[0] == pytest.approx(watts)
        assert evt.suggested_label == UNRECOGNISED_DISPLAY
        assert max(evt.segments[0]) >= 20.0


# ══════════════════════════════════════════════════════════════════════════
# M-6 residual — OpenMax is inert AND unreachable, and must stay declared so
#
# The prior session logged this as "misleading dead code in the critical path".
# It is not in the critical path at all; these tests pin why, so that nobody
# either (a) trusts the channel, or (b) "repairs" it by populating names from
# the squared-L2 distances the trainer happens to have, which would calibrate
# the tail on a different metric than the query.
# ══════════════════════════════════════════════════════════════════════════

class TestOpenMaxIsNotTheRejectChannel:

    def test_shipped_artifact_has_tails_but_no_named_tails(self, pipeline):
        # `train_demo_models.py` fits via the indexed API, which writes
        # `_weibull[idx]` only. The startup banner used to log "✅" off that
        # dict while the runtime read the empty one.
        assert pipeline.weibull._weibull, "index-keyed tails should be present"
        assert not getattr(pipeline.weibull, "_weibull_by_name", {}), (
            "named tails are what compute_open_set_prob reads; if these are now "
            "populated, re-read its docstring — the metric must match plain L2")

    def test_open_set_prob_fails_open_when_unfitted(self, pipeline):
        # 0.0 means "cannot judge", but SupportSetManager.classify reads it as
        # "definitely known" via `open_set > (1 - confidence_threshold)`. So the
        # legacy path fails OPEN. Pinned because it is a fail-open default.
        emb = np.random.default_rng(0).normal(0, 50, 128)
        p = pipeline.weibull.compute_open_set_prob(emb, ["laptop", "monitor"], [9.0, 9.0])
        assert p == 0.0
        assert not (p > (1.0 - pipeline.confidence_threshold))

    def test_the_maths_works_when_fitted_by_name_so_the_gap_is_the_wiring(self):
        # Proves the defect is the training call, not the Weibull implementation:
        # via the 2-dict API the same method scores a far embedding as novel.
        from src.models.protonet import OpenMaxWeibull

        rng = np.random.default_rng(1)
        known = rng.normal(0, 1, (40, 8))
        omw = OpenMaxWeibull(num_classes=1, tail_size=20)
        omw.fit({"known": known.mean(axis=0)}, {"known": known})
        assert omw._weibull_by_name, "2-dict API must populate the named tails"
        far = np.full(8, 50.0)
        assert omw.compute_open_set_prob(far, ["known"], [0.0]) > 0.5

    def test_legacy_anchor_path_is_unreachable_on_every_profile(self):
        # The guard that makes SupportSetManager.classify — and with it the only
        # compute_open_set_prob caller — dead: no shipped profile points at an
        # anchors file that exists.
        for profile in ("config/config.yaml", "config/config.demo.yaml",
                        "config/config.hardware.yaml"):
            with open(profile) as fh:
                cfg = yaml.safe_load(fh)
            anchors = (cfg.get("protonet") or {}).get("anchors_path", "")
            assert not (anchors and os.path.exists(anchors)), (
                f"{profile} now ships anchors at {anchors!r}; the legacy classify "
                "path becomes live and its open-set gate fails open")

    def test_physical_gate_not_openmax_is_what_rejects(self, pipeline):
        # The positive statement: rejection of a novel load happens with the
        # open-set channel provably inert.
        assert not getattr(pipeline.weibull, "_weibull_by_name", {})
        name, conf, _ = pipeline._classify_device("dev", 2000.0,
                                                  filtered_segment=_window(2000, peak=2100))
        assert name == UNRECOGNISED



# ══════════════════════════════════════════════════════════════════════════
# M-9 — the five required demo classes are actually recognised
#
# The recognition scope (config protonet.classes) is phone / laptop / bulb /
# projector / fan. On the shipped UK-DALE registry that was 0/96: measured on
# `data/real/cache/ukdale_windows_demo.npz`, the trained `laptop` prototype is
# a 21 W netbook (p50) and `phone_charger` has ZERO windows above the 20 W
# on-threshold, so the artifact cannot represent the demo fleet's 120 W laptop
# or 45 W USB-PD charger at any threshold. `bulb` and `fan` had no prototype
# at all. The remedy is enrolment (scripts/enroll_demo_devices.py), which
# gives each class a prototype AND the power envelope observed on it — the
# independent absolute-watts channel `_classify_device` demands before naming
# a device.
# ══════════════════════════════════════════════════════════════════════════

# rated/var straight from backend/scripts/simulate_esp32.py:DEMO_DEVICES —
# the generator the demo fleet actually publishes from.
FLEET_PROFILES = {
    "phone":            (45.0, 15.0),
    "laptop":           (120.0, 25.0),
    "bulb":             (60.0, 4.0),
    "projector":        (300.0, 20.0),
    "fan":              (75.0, 6.0),
    "desktop_computer": (250.0, 35.0),
}
REQUIRED_FIVE = ("phone", "laptop", "bulb", "projector", "fan")

# Enrolment uses seeds 100..109 (scripts/enroll_demo_devices.ENROLL_SEED_BASE),
# so 0..11 are genuinely held out and this is not a train-on-test result.
TEST_SEEDS = range(12)


def _fleet_window(rated, var, seed, n=SEQ_LEN, on_fraction=0.8):
    """One 1 Hz window as simulate_esp32.simulate_device would publish it."""
    rng = np.random.default_rng(seed)
    w = np.zeros(n, dtype=np.float32)
    off = int(n * (1.0 - on_fraction))
    w[off:] = np.maximum(0.0, rng.normal(rated, var, n - off))
    return w


class TestFiveRequiredClassesAreRecognised:
    # The enrolled artifact is generated, not committed (the whole demo weights
    # directory is gitignored), so a checkout that has not run the enrolment step
    # skips these rather than reporting a red suite for a missing setup step.
    pytestmark = pytest.mark.skipif(
        not os.path.exists(_configured_registry_path()),
        reason=f"enrolled registry absent at {_configured_registry_path()}; "
               f"run scripts/enroll_demo_devices.py",
    )

    def test_enrolled_registry_is_the_one_in_use(self, pipeline):
        # If the demo profile ever loses `registry_path`, every assertion below
        # would silently be testing the shipped artifact instead.
        envelopes = pipeline.prototype_registry.envelopes
        for cls in REQUIRED_FIVE:
            assert cls in envelopes, (
                f"{cls} has no power envelope — the demo profile is not pointing "
                f"at an enrolled registry; run scripts/enroll_demo_devices.py")

    @pytest.mark.parametrize("cls", REQUIRED_FIVE)
    def test_required_class_is_named_on_held_out_windows(self, pipeline, cls):
        rated, var = FLEET_PROFILES[cls]
        got = [pipeline._classify_device(
                   f"node_{cls}", rated,
                   filtered_segment=_fleet_window(rated, var, s))[0]
               for s in TEST_SEEDS]
        # Full recall is the delivered behaviour, not an aspiration: an enrolled
        # envelope is a tight physical constraint and these windows sit inside it.
        assert got.count(cls) == len(got), f"{cls}: got {got}"

    def test_confidence_clears_the_recognition_threshold(self, pipeline):
        for cls in REQUIRED_FIVE:
            rated, var = FLEET_PROFILES[cls]
            _, conf, _ = pipeline._classify_device(
                f"node_{cls}", rated,
                filtered_segment=_fleet_window(rated, var, 0))
            assert conf >= pipeline.recognition_threshold, f"{cls} conf={conf}"

    def test_the_fifth_fleet_node_did_not_regress(self, pipeline):
        # Enrolling only the required four made this WORSE than the baseline:
        # the enrolled projector envelope (297-302 W) pads out to 252-347 W, an
        # enrolled class outranks a shipped one, and a 250 W desktop window got
        # called `projector`. Measured over 24 windows: 12/24 shipped, 8/24 with
        # four enrolled, 21/24 with all five. Enrolling every fleet node is what
        # keeps it in its own lane.
        #
        # 250 W still sits just inside projector's padded floor (252 W minus
        # nothing to spare), so ~1 window in 8 goes to projector and the numbers
        # below are the measured behaviour, not a rounded aspiration. Closing
        # that last gap needs a narrower pad than the shared physical slack, a
        # change that reaches beyond this scope.
        rated, var = FLEET_PROFILES["desktop_computer"]
        got = [pipeline._classify_device(
                   "node_desktop", rated,
                   filtered_segment=_fleet_window(rated, var, s))[0]
               for s in TEST_SEEDS]
        assert got.count("desktop_computer") >= 10, got
        assert got.count("projector") <= 1, f"projector stealing desktop windows: {got}"

    @pytest.mark.parametrize("label,rated,var", [
        ("kettle", 2000.0, 50.0),
        ("oven", 2500.0, 80.0),
        ("heater", 800.0, 30.0),
        ("hairdryer", 1200.0, 40.0),
        ("microwave", 900.0, 60.0),
        ("washing_machine", 500.0, 40.0),
        ("ev_charger", 7000.0, 100.0),
    ])
    def test_novel_out_of_family_load_is_still_unrecognised(self, pipeline,
                                                            label, rated, var):
        # Enrolment must not buy coverage by weakening the reject channel.
        name, _, _ = pipeline._classify_device(
            f"node_{label}", rated, filtered_segment=_fleet_window(rated, var, 7))
        assert name == UNRECOGNISED, f"{label} {rated} W was named {name}"

    def test_a_novel_load_inside_an_enrolled_envelope_is_the_known_limit(self, pipeline):
        # The honest cost of the trade, pinned so nobody reads the suite above as
        # "novelty detection is solved". A 120 W load that is NOT the laptop sits
        # inside the enrolled laptop envelope (117-122 W, padded 99-140 W), and a
        # 1 Hz power window carries no scale-independent shape cue that separates
        # them — the embedding is power-scale-blind (ML_PIPELINE_FIX §2.2). So it
        # is named `laptop`. Distinguishing these needs another measurement
        # (power factor, per-socket metering), not another threshold.
        name, _, _ = pipeline._classify_device(
            "node_fridge", 120.0, filtered_segment=_fleet_window(120.0, 10.0, 7))
        assert name == "laptop"


class TestRegistryPathRoundTrips:

    def test_label_submission_saves_where_startup_reads(self, pipeline):
        # `handle_label_submitted` used to derive its save path from
        # `weights_path` while startup read `registry_path`. The moment those
        # differ, every operator label is written to a file the next boot never
        # opens and the label loop silently forgets everything.
        import torch

        path = pipeline._resolve_registry_path()
        _enroll(pipeline, "my_device", 78.0, peak=82.0)
        assert os.path.exists(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert "my_device" in saved
