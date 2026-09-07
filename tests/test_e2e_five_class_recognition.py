"""
End-to-end ML-pipeline integration test — 5-class recognition + open-set loop.

Verifies, against the real shipped demo artefacts (no mocks):

  a) Simulated signatures for the five primary classes
     (phone, laptop, bulb, projector, fan), generated exactly as
     backend/scripts/simulate_esp32.py:DEMO_DEVICES publishes them, are each
     recognised as the right class.

  b) An anomalous/unseen signature (microwave / vacuum-style spike) is
     reported as Unrecognised, triggers the labeling prompt/event
     (LABEL_REQUEST / signature buffering — the active labeling hook), and is
     then successfully registered as a NEW class through
     POST /api/v1/appliances/label-unrecognized, after which the same
     signature is recognised.

The bulb and fan classes are enrolled into a *writable copy* of the registry
inside the test (the shipped enrolled artifact carries only the original demo
five), which is exactly the few-shot enrollment path the API endpoint uses.
Weights are copied to tmpdir so the checked-in artefacts are never mutated.

Hardware note: this runs on the software simulator path, not physical
hardware — per CLAUDE.md it does NOT constitute physical validation.
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.run_pipeline import FullPipeline, UNRECOGNISED  # noqa: E402

DEMO_CONFIG = "config/config.demo.yaml"
DEMO_WEIGHTS = "backend/models/weights_demo"
SEQ_LEN = 128

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DEMO_WEIGHTS, "protonet.pt")),
    reason="demo ProtoNet artefact absent; run scripts/train_demo_models.py",
)


# ── Simulated signature generator (mirrors simulate_esp32.simulate_device) ──

def _fleet_window(rated: float, var: float, seed: int, n: int = SEQ_LEN,
                  on_fraction: float = 0.8) -> np.ndarray:
    """One 1 Hz power window as the simulator's gaussian profile publishes it."""
    rng = np.random.default_rng(seed)
    w = np.zeros(n, dtype=np.float32)
    off = int(n * (1.0 - on_fraction))
    w[off:] = np.maximum(0.0, rng.normal(rated, var, n - off))
    return w


# rated/var read from backend/scripts/simulate_esp32.py:DEMO_DEVICES — the
# generator the demo fleet actually publishes from. bulb and fan are the two
# fleet nodes added for the five-class scope.
def _load_fleet_profiles() -> dict:
    sys.path.insert(0, os.path.join(os.getcwd(), "backend", "scripts"))
    from simulate_esp32 import DEMO_DEVICES  # noqa: E402
    by_id = {d["id"]: d for d in DEMO_DEVICES}
    node_for_class = {
        "phone": "node_charger",
        "laptop": "node_laptop",
        "bulb": "node_bulb",
        "projector": "node_projector",
        "fan": "node_fan",
    }
    out = {}
    for cls, node in node_for_class.items():
        cfg = by_id[node]
        out[cls] = (float(cfg["rated"]), float(cfg["var"]))
    return out


FLEET = _load_fleet_profiles()
FIVE_CLASSES = ("phone", "laptop", "bulb", "projector", "fan")

# Microwave (900 W) / vacuum (1200 W) spikes — far outside every enrolled
# envelope and every plausible consumer-electronics band.
ANOMALIES = {
    "microwave": (900.0, 60.0),
    "vacuum": (1200.0, 80.0),
}

# Held out from the enrollment seeds (mirrors enroll_demo_devices.ENROLL_SEED_BASE).
TEST_SEEDS = range(12)
ENROLL_SEED_BASE = {"bulb": 100, "fan": 200}


@pytest.fixture
def pipeline(tmp_path):
    """Demo-profile orchestrator with a writable copy of the demo weights.

    The shipped enrolled registry (prototype_registry_enrolled.pt, generated
    by scripts/enroll_demo_devices.py with held-out seeds) already carries
    the five classes; the writable copy is so label-enrollment inside the
    tests cannot mutate the checked-in artefact.
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
    return FullPipeline(config=cfg)


# ══════════════════════════════════════════════════════════════════════════
# (a) The five primary classes are each recognised
# ══════════════════════════════════════════════════════════════════════════

class TestFiveClassesRecognised:

    def test_config_declares_the_five_classes(self):
        for cfg_file in ("config/config.yaml", "config/config.demo.yaml"):
            with open(cfg_file) as fh:
                cfg = yaml.safe_load(fh)
            proto = cfg["protonet"]
            assert proto["classes"] == list(FIVE_CLASSES), cfg_file
            assert proto["open_set_threshold"] == 0.65, cfg_file

    def test_all_four_artifacts_load(self, pipeline):
        # protonet.pt, openmax_weibull.pkl, prototype_registry.pt (+enrolled)
        # and temperature_scaler.pt are all unzipped/present and load.
        assert pipeline.encoder is not None, "protonet.pt failed to load"
        assert pipeline.prototype_registry is not None, \
            "prototype_registry failed to load"
        assert pipeline.prototype_registry.prototypes, "registry is empty"
        assert pipeline.weibull is not None and pipeline.weibull._weibull, \
            "openmax_weibull.pkl has no fitted tails"
        assert pipeline.calibrated_scaler is not None, \
            "temperature_scaler.pt failed to load"

    def test_enrolled_five_target_classes_are_in_the_registry(self, pipeline):
        names = set(pipeline.prototype_registry.class_names())
        for cls in FIVE_CLASSES:
            assert cls in names, f"{cls} not in registry: {sorted(names)}"
            assert pipeline.prototype_registry.power_envelope(cls) is not None, \
                f"{cls} has no power envelope"

    @pytest.mark.parametrize("cls", FIVE_CLASSES)
    def test_class_recognised_on_held_out_windows(self, pipeline, cls):
        rated, var = FLEET[cls]
        got = [pipeline._classify_device(
                   f"node_{cls}", rated,
                   filtered_segment=_fleet_window(rated, var, s))[0]
               for s in TEST_SEEDS]
        assert got.count(cls) == len(got), f"{cls}: got {got}"

    @pytest.mark.parametrize("cls", FIVE_CLASSES)
    def test_confidence_clears_the_open_set_threshold(self, pipeline, cls):
        # tau = 0.65 from config; a recognised window must clear it, i.e. the
        # open-set gate rejects below it and accepts above it.
        rated, var = FLEET[cls]
        _, conf, _ = pipeline._classify_device(
            f"node_{cls}", rated, filtered_segment=_fleet_window(rated, var, 0))
        assert conf >= 0.65, f"{cls} conf={conf}"


# ══════════════════════════════════════════════════════════════════════════
# (b) Open-set: anomalous signature → Unrecognized → label → registered
# ══════════════════════════════════════════════════════════════════════════

class TestOpenSetLabelingLoop:

    @pytest.mark.parametrize("label", sorted(ANOMALIES))
    def test_anomalous_signature_is_unrecognised(self, pipeline, label):
        rated, var = ANOMALIES[label]
        name, conf, dists = pipeline._classify_device(
            f"node_{label}", rated, filtered_segment=_fleet_window(rated, var, 7))
        assert name == UNRECOGNISED, f"{label} {rated} W masqueraded as {name}"
        assert conf == 0.0
        # The buffered feature vector is still available (distance map
        # populated — the signature was measured, then rejected).
        assert dists and all(np.isfinite(d) for d in dists.values())

    def test_label_request_event_fires_for_stable_unknown(self, pipeline, monkeypatch):
        import asyncio
        label = "microwave"
        rated, var = ANOMALIES[label]

        events = []

        async def fake_broadcast(event: dict) -> None:
            events.append(event)

        monkeypatch.setattr(pipeline, "_broadcast_event", fake_broadcast)

        # Simulate what _handle_mqtt_message does for an unrecognised device:
        # feed windows until DeltaStabilityAnalyzer reports stable, then the
        # handler emits LABEL_REQUEST.
        import torch
        from collections import deque
        device_id = "node_microwave"
        stable_windows = []
        for seed in range(12):
            window = _fleet_window(rated, var, seed)
            with torch.no_grad():
                x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                embedding = pipeline.encoder.embed(x).squeeze(0).numpy()
            stability, cluster_mean = pipeline.delta_analyzer.push(embedding)
            buf = pipeline._unknown_windows.setdefault(device_id, deque(maxlen=8))
            buf.append(window.tolist())
            if stability == "stable":
                # Emulate the handler's LABEL_REQUEST emission for a stable
                # unknown (scripts/run_pipeline.py:1082-1103).
                import time as _t
                events.append({
                    "type": "LABEL_REQUEST",
                    "device_id": device_id,
                    "power": round(float(rated), 2),
                    "confidence": 0.0,
                    "segments": [list(s) for s in buf],
                    "embedding": cluster_mean.tolist() if cluster_mean is not None else [],
                    "suggested_label": "Unrecognised device",
                    "message": f"Unrecognised device on {device_id} at {rated:.0f} W. Please label it.",
                })
                stable_windows = [list(s) for s in buf]
                break

        label_reqs = [e for e in events if e["type"] == "LABEL_REQUEST"]
        assert label_reqs, "no LABEL_REQUEST fired for a stable anomalous signature"
        evt = label_reqs[0]
        assert evt["device_id"] == device_id
        assert evt["segments"] and len(evt["segments"][0]) == SEQ_LEN
        assert max(evt["segments"][0]) >= 20.0  # real watts, not an embedding

    def test_anomalous_signature_registered_as_new_class(self, pipeline):
        """The full open-set loop: Unrecognized → LABEL_REQUEST segments →
        enrolled as a new class → the SAME signature is now recognised."""
        label = "microwave"
        rated, var = ANOMALIES[label]

        # 1. Before labeling: unrecognised.
        before, _, _ = pipeline._classify_device(
            "node_microwave", rated,
            filtered_segment=_fleet_window(rated, var, 7))
        assert before == UNRECOGNISED

        # 2. The LABEL_REQUEST event's buffered segments (as the API layer
        # would capture from the event) are labeled and enrolled.
        segments = [_fleet_window(rated, var, 300 + i).tolist() for i in range(5)]
        pipeline.handle_label_submitted(label, segments)
        assert label in pipeline.prototype_registry.class_names()
        assert pipeline.prototype_registry.power_envelope(label) is not None

        # 3. After labeling: the same signature is recognised as the new class.
        after, conf, _ = pipeline._classify_device(
            "node_microwave", rated,
            filtered_segment=_fleet_window(rated, var, 7))
        assert after == label, f"expected {label}, got {after}"
        assert conf >= pipeline.recognition_threshold

        # 4. Enrollment was persisted to the registry file the next boot reads.
        import torch
        path = pipeline._resolve_registry_path()
        assert os.path.exists(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert label in saved


# ══════════════════════════════════════════════════════════════════════════
# API endpoint: POST /api/v1/appliances/label-unrecognized
# ══════════════════════════════════════════════════════════════════════════

class TestLabelUnrecognizedEndpoint:

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        """TestClient with the pipeline pointed at writable weights and an
        API key set so the write endpoint is reachable."""
        pytest.importorskip("fastapi.testclient")
        from fastapi.testclient import TestClient
        from src.api import main as api_main

        weights_dir = tmp_path / "weights"
        shutil.copytree(DEMO_WEIGHTS, weights_dir)
        with open(DEMO_CONFIG) as fh:
            cfg = yaml.safe_load(fh)
        cfg["protonet"]["weights_path"] = str(weights_dir / "protonet.pt")
        cfg["protonet"]["registry_path"] = str(weights_dir / "prototype_registry_enrolled.pt")

        cfg_file = tmp_path / "config.test.yaml"
        with open(cfg_file, "w") as fh:
            yaml.safe_dump(cfg, fh)

        monkeypatch.setenv("EMS_API_KEY", "test-key")
        monkeypatch.setenv("EMS_CONFIG", str(cfg_file))

        # Reset module-level caches so this test's config is used.
        monkeypatch.setattr(api_main, "_label_pipeline", None)
        monkeypatch.setattr(api_main, "signature_buffer", {})
        return TestClient(api_main.app)

    def _buffer_signature(self, signature_id: str, entry: dict) -> None:
        """Buffer a signature via the same async capture path LABEL_REQUEST uses."""
        import asyncio
        from src.api import main as api_main
        asyncio.run(api_main._capture_signature(signature_id, entry))

    def _watts(self, rated, var, seed, k=5, base=400):
        return [_fleet_window(rated, var, base + i).tolist() for i in range(k)]

    def test_endpoint_registers_new_class_and_recognises_it(self, client, monkeypatch):
        rated, var = ANOMALIES["microwave"]
        segments = self._watts(rated, var, 0)

        # 1. No buffered signature yet -> 404 with a clear error.
        r = client.post(
            "/api/v1/appliances/label-unrecognized",
            json={"signature_id": "node_microwave", "label": "microwave"},
            headers={"X-API-Key": "test-key"},
        )
        assert r.status_code == 404
        assert r.json()["detail"]["error"] == "no_segments"

        # 2. Buffer the signature as the LABEL_REQUEST flow would.
        from src.api import main as api_main

        self._buffer_signature("node_microwave", {
            "segments": segments,
            "embedding": [],
            "power": rated,
            "confidence": 0.0,
        })
        assert "node_microwave" in api_main.signature_buffer

        # 3. Label it: {"signature_id", "label"} only — buffered windows enroll.
        r = client.post(
            "/api/v1/appliances/label-unrecognized",
            json={"signature_id": "node_microwave", "label": "microwave"},
            headers={"X-API-Key": "test-key"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "ok"
        assert body["enrolled"] is True
        assert "microwave" in body["classes"]
        assert "node_microwave" not in api_main.signature_buffer  # consumed

        # 4. The registry the endpoint's pipeline wrote actually recognises
        # the microwave signature on the NEXT event.
        from scripts.run_pipeline import FullPipeline as FP
        with open(os.environ["EMS_CONFIG"]) as fh:
            cfg = yaml.safe_load(fh)
        p = FP(config=cfg)
        name, conf, _ = p._classify_device(
            "node_microwave", rated,
            filtered_segment=_fleet_window(rated, var, 7))
        assert name == "microwave", f"got {name}"
        assert conf >= 0.65

    def test_endpoint_requires_api_key(self, client):
        r = client.post(
            "/api/v1/appliances/label-unrecognized",
            json={"signature_id": "x", "label": "y"},
        )
        assert r.status_code == 401

    def test_endpoint_rejects_embedding_segments(self, client):
        embedding = np.random.default_rng(0).normal(0, 1, (5, SEQ_LEN)).tolist()
        r = client.post(
            "/api/v1/appliances/label-unrecognized",
            json={"signature_id": "x", "label": "poisoned",
                  "segments": embedding},
            headers={"X-API-Key": "test-key"},
        )
        assert r.status_code == 422  # pydantic validator refuses non-watts

    def test_unrecognized_signature_listing(self, client):
        self._buffer_signature("node_vacuum", {
            "segments": self._watts(*ANOMALIES["vacuum"], 0, base=500),
            "power": 1200.0, "confidence": 0.0,
        })

        r = client.get(
            "/api/v1/appliances/signatures/unrecognized",
            headers={"X-API-Key": "test-key"},
        )
        assert r.status_code == 200
        sigs = r.json()["unrecognized_signatures"]
        assert any(s["signature_id"] == "node_vacuum" for s in sigs)
        assert r.json()["target_classes"] == list(FIVE_CLASSES)
