"""
Run 0.8 regression — the hardware profile opts the physical bench node out of
RL shedding.

run_pipeline.py constructs the RL agent with the orchestrator's own config.
Before the fix it called TabularQLearningAgent() bare, which always loaded
config/config.yaml regardless of --config: no profile key could influence the
agent, so `devices.node_bench_agg.tier0: true` was inert. With an empty
Q-table and epsilon exploration returning SHED, the promotion gate could then
latch and open the physical relay uncommanded (~1 stray OFF per cooldown
window, indefinitely).
"""
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.run_pipeline import FullPipeline  # noqa: E402
from src.rl.agent import TabularQLearningAgent  # noqa: E402

HARDWARE_CONFIG = "config/config.hardware.yaml"


def _hardware_cfg(tmp_path):
    with open(HARDWARE_CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    # Never touch the real DB / CSV fallback from a test.
    cfg["database"]["path"] = str(tmp_path / "ems_state.db")
    cfg["database"]["fallback_csv"] = str(tmp_path / "fallback.csv")
    return cfg


class TestHardwareRLOptOut:

    def test_agent_receives_the_hardware_profile(self, tmp_path):
        # The regression this guards: run_pipeline.py used to construct
        # TabularQLearningAgent() without config, so it always loaded
        # config/config.yaml and no hardware-profile key could reach it.
        pipeline = FullPipeline(config=_hardware_cfg(tmp_path))
        assert "node_bench_agg" in pipeline.agent.NEVER_SHED
        assert pipeline.agent.cooldown == 300.0  # from the hardware rl: section

    def test_default_config_does_not_protect_the_bench_node(self):
        # Proves the assertion above is not vacuous: with the default
        # config/config.yaml the bench node is NOT in the blacklist.
        agent = TabularQLearningAgent()  # loads config/config.yaml
        assert "node_bench_agg" not in agent.NEVER_SHED

    @pytest.mark.asyncio
    async def test_publish_site_blocks_shed_for_tier0_bench_node(
            self, tmp_path, monkeypatch):
        """A confident non-tier0 class stays SHED-eligible while the tier0
        bench node is NEVER_SHED-blocked at the publish site (the
        device_is_critical defense-in-depth check in _handle_mqtt_message)."""
        pipeline = FullPipeline(config=_hardware_cfg(tmp_path))
        published = []

        async def record_publish(topic, payload, **kwargs):
            published.append((topic, payload))

        monkeypatch.setattr(pipeline.mqtt, "publish_command", record_publish)
        # Keep the Q-update from writing its CSV log into the CWD.
        monkeypatch.setattr(pipeline.agent, "update", lambda *a, **k: None)
        # Force the known+confident branch deterministically: classification
        # is not under test here, the publish gate is.
        monkeypatch.setattr(pipeline, "_classify_device",
                            lambda *a, **k: ("laptop", 0.99, {}))
        monkeypatch.setattr(pipeline.agent, "act", lambda *a, **k: "SHED")
        pipeline.promo_gate._promoted = True  # LIVE mode, not shadow

        async def feed(device_id):
            published.clear()
            # Force the CNN path without depending on detector warm-up.
            pipeline.cnn_active_ticks[device_id] = 5
            await pipeline._handle_mqtt_message(
                f"home/sensor/{device_id}/power", "65.0")

        # Confident non-tier0 class: the SHED command IS published.
        await feed("node_laptop")
        assert ("home/plug/node_laptop/command", "OFF") in published

        # Tier0 bench node: blocked at the publish site — no relay command.
        await feed("node_bench_agg")
        assert not any(t == "home/plug/node_bench_agg/command"
                       for t, _ in published)
