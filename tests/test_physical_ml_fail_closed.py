"""Deployment-boundary regressions for the physical ML profile."""

from copy import deepcopy

import numpy as np
import yaml

from scripts.run_pipeline import FullPipeline


def _hardware_config():
    with open("config/config.hardware.yaml") as fh:
        return yaml.safe_load(fh)


def test_hardware_profile_without_physical_registry_fails_closed():
    """A physical profile without an enrolled registry must abstain."""
    cfg = _hardware_config()
    pipeline = FullPipeline(config=cfg)

    assert pipeline.ml_inference_blocked is True
    assert pipeline.prototype_registry is None
    label, confidence, distances = pipeline._classify_device(
        "physical_gate", 120.0, np.full(128, 120.0, dtype=np.float32)
    )
    assert label == "unknown"
    assert confidence == 0.0
    assert distances == {}


def test_demo_profile_keeps_existing_degraded_classifier_behavior():
    """The physical fail-closed policy must not disable the demo profile."""
    with open("config/config.demo.yaml") as fh:
        demo_cfg = yaml.safe_load(fh)
    pipeline = FullPipeline(config=deepcopy(demo_cfg))
    assert pipeline.ml_inference_blocked is False
