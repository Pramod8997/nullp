"""Hardware-alignment parity guard (Run 0.1).

Enforces claude_debug/HARDWARE_ALIGNMENT_CONTRACT.md: firmware/esp32_node/src/
main.cpp is the single source of truth for every hardware-binding value, and
every other site — the twin's defaults, config/config.hardware.yaml, the
pipeline's MQTT surface, the status strings, the command semantics, the
DEVICE_ID chain — must agree with it in the same commit. A failure here is a
defect (a site left behind by a constant change), not a tuning opportunity.

The assertions compare sites against the *parsed* firmware constant, never
against hardcoded wattages, so a deliberate one-commit move (e.g. the WS-A
RATED_WATTS ceiling change) stays green while a partial move goes red.

Parsing precedent: tests/test_relay_safety_boot_brownout.py:100-109.
All paths are repo-root-relative, matching the suite's invocation
(`python -m pytest tests/ ...` from the repo root).
"""

import pathlib
import re

import pytest
import yaml

from src.hardware.esp32_firmware_sim import ESP32FirmwareNode, VirtualPZEM004T

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MAIN_CPP = REPO_ROOT / "firmware" / "esp32_node" / "src" / "main.cpp"
TWIN_PY = REPO_ROOT / "src" / "hardware" / "esp32_firmware_sim.py"
CONFIG_HW = REPO_ROOT / "config" / "config.hardware.yaml"
PIPELINE_PY = REPO_ROOT / "scripts" / "run_pipeline.py"
SECRETS_H = REPO_ROOT / "firmware" / "esp32_node" / "include" / "secrets.h"
WIRING_DOC = REPO_ROOT / "claude_debug" / "WIRING_STEP_BY_STEP.md"

if not MAIN_CPP.exists():
    pytest.skip(
        "firmware/esp32_node/src/main.cpp is missing — hardware-alignment "
        "parity cannot be enforced (the file is tracked, so CI checkouts have it)",
        allow_module_level=True,
    )

_MAIN_CPP = MAIN_CPP.read_text()
_TWIN_PY = TWIN_PY.read_text()
_PIPELINE_PY = PIPELINE_PY.read_text()
_HW_CONFIG = yaml.safe_load(CONFIG_HW.read_text())


# ── Firmware constants (the source of truth) ─────────────────────────────


def _fw_float(name: str) -> float:
    m = re.search(rf"const\s+float\s+{name}\s*=\s*([0-9.eE+-]+)\s*;", _MAIN_CPP)
    assert m, f"const float {name} not found in {MAIN_CPP}"
    return float(m.group(1))


def _fw_int(decl: str, name: str) -> int:
    m = re.search(rf"const\s+{decl}\s+{name}\s*=\s*(\d+)\s*;", _MAIN_CPP)
    assert m, f"const {decl} {name} not found in {MAIN_CPP}"
    return int(m.group(1))


def _fw_bool(name: str) -> bool:
    m = re.search(rf"const\s+bool\s+{name}\s*=\s*(true|false)\s*;", _MAIN_CPP)
    assert m, f"const bool {name} not found in {MAIN_CPP}"
    return m.group(1) == "true"


FW_RATED_WATTS = _fw_float("RATED_WATTS")
FW_CRITICAL_PCT = _fw_float("CRITICAL_PCT")
FW_RELAY_ACTIVE_LOW = _fw_bool("RELAY_ACTIVE_LOW")
FW_RELAY_PIN = _fw_int("int", "RELAY_PIN")
FW_PZEM_RX_PIN = _fw_int("int", "PZEM_RX_PIN")
FW_PZEM_TX_PIN = _fw_int("int", "PZEM_TX_PIN")
FW_EDGE_ROC_THRESHOLD = _fw_float("EDGE_ROC_THRESHOLD")
FW_BASELINE_INRUSH_CEIL = _fw_float("BASELINE_INRUSH_CEIL")
FW_INRUSH_HEADROOM = _fw_float("INRUSH_HEADROOM")
FW_SAFETY_LOCKOUT_MS = _fw_int("unsigned long", "SAFETY_LOCKOUT_MS")


# ── 1. Pins & polarity ───────────────────────────────────────────────────


def test_firmware_pins_match_wiring_docs_and_twin():
    """RELAY/PZEM pins are physical wiring facts (contract §1.2): main.cpp,
    the bench wiring doc of record, and the twin's GPIO naming must agree.
    A pin change is a re-wiring event, never a code-only tweak."""
    # Whitespace-flattened and stripped of markdown blockquote/bold markers so
    # doc line-wrapping cannot mask a match.
    raw = WIRING_DOC.read_text()
    wiring = " ".join(raw.replace(">", " ").replace("*", " ").split())
    for pin in (FW_RELAY_PIN, FW_PZEM_RX_PIN, FW_PZEM_TX_PIN):
        assert f"GPIO {pin}" in wiring, (
            f"GPIO {pin} from main.cpp is absent from {WIRING_DOC.name}"
        )
    # The UART is crossed on purpose: ESP32 RX (PZEM_RX_PIN) <- PZEM TX,
    # ESP32 TX (PZEM_TX_PIN) -> PZEM RX (WIRING_STEP_BY_STEP.md §2 note).
    assert f"ESP32 RX (GPIO {FW_PZEM_RX_PIN})" in wiring
    assert f"ESP32 TX (GPIO {FW_PZEM_TX_PIN})" in wiring
    # The twin models the relay on the firmware's GPIO by attribute name.
    assert f"gpio{FW_RELAY_PIN}_relay_state" in _TWIN_PY, (
        "twin models the relay on a different GPIO than RELAY_PIN"
    )


def test_twin_defaults_match_firmware_constants():
    """Twin ctor defaults are bench-parity sites (contract §1.1/§1.3): they
    must track the firmware constants, not independent choices."""
    probe = ESP32FirmwareNode(device_id="parity_probe")
    assert probe.rated_watts == FW_RATED_WATTS, (
        f"twin default rated_watts drifted: twin={probe.rated_watts}, "
        f"firmware RATED_WATTS={FW_RATED_WATTS}"
    )
    # The locked spec pins active-HIGH (HARDWARE_FINAL_SPEC.md D4/D5); a flip
    # is the D5 purchase-contingency only, with bring-up Stage 2 re-run.
    assert FW_RELAY_ACTIVE_LOW is False, (
        "RELAY_ACTIVE_LOW flipped in main.cpp — re-run bring-up Stage 2 "
        "(HARDWARE_FINAL_SPEC.md D4/D5) before trusting any twin evidence"
    )
    assert probe.relay_active_low is FW_RELAY_ACTIVE_LOW, (
        "twin default polarity drifted from RELAY_ACTIVE_LOW"
    )
    assert probe.safety_lockout_seconds == FW_SAFETY_LOCKOUT_MS / 1000.0, (
        "twin safety_lockout_seconds drifted from SAFETY_LOCKOUT_MS"
    )


def test_twin_inrush_suppression_constants_match_firmware():
    """The twin's inrush-suppression constants must equal BASELINE_INRUSH_CEIL
    and INRUSH_HEADROOM (contract §1.3)."""
    m = re.search(r"baseline_avg\s*<\s*([0-9.]+)", _TWIN_PY)
    assert m, "twin inrush baseline ceiling not found in esp32_firmware_sim.py"
    assert float(m.group(1)) == FW_BASELINE_INRUSH_CEIL, (
        "twin inrush baseline ceiling drifted from BASELINE_INRUSH_CEIL"
    )
    m = re.search(r"baseline_avg\s*\+\s*([0-9.]+)", _TWIN_PY)
    assert m, "twin inrush headroom not found in esp32_firmware_sim.py"
    assert float(m.group(1)) == FW_INRUSH_HEADROOM, (
        "twin inrush headroom drifted from INRUSH_HEADROOM"
    )


def test_virtual_pzem_default_power_factor_is_one():
    """PF init 1.0 mirrors POWER_FACTOR / sharedPf at main.cpp:75/:122."""
    assert VirtualPZEM004T().power_factor == 1.0


# ── 2. Arc-fault ROC dt clamp (contract §1.5, ledger C4) ─────────────────


def _warmed_node(load_watts: float) -> ESP32FirmwareNode:
    """A node whose 5-sample baseline ring is full of load_watts. With
    load >= BASELINE_INRUSH_CEIL the inrush suppression is OFF, so the arc
    -fault channel sees the raw step (the hardware condition for C3/C4)."""
    node = ESP32FirmwareNode(device_id="roc_probe")
    node.set_relay(True)
    node.pzem.set_load(load_watts)
    for _ in range(5):
        node.core0_safety_step(sim_dt=0.1)
    return node


def test_twin_roc_dt_clamp_matches_pzem_cadence():
    """The firmware's effective arc-fault dt is 0.134-0.16 s (PZEM register
    cadence), so on hardware a 120 W laptop step reads ~750-896 W/s and never
    trips, while a projector-sized step does. The twin must model exactly
    this; with an unclamped dt of 0.1 s it computes 1200 W/s for the laptop
    step and trips where hardware does not (ledger C4)."""
    # +120 W step onto a 60 W baseline (suppression off: 60 >= 50 W ceiling):
    # clamped ROC = 120/0.134 ≈ 895.5 W/s < EDGE_ROC_THRESHOLD -> no trip.
    laptop = _warmed_node(60.0)
    laptop.pzem.set_load(180.0)
    laptop.core0_safety_step(sim_dt=0.1)
    assert laptop.shared_arc_fault is False, (
        "a 120 W step tripped arc-fault: the twin's ROC dt clamp regressed "
        "(hardware does not trip a 120 W laptop step)"
    )
    assert laptop.gpio18_relay_state is True

    # +250 W step (60 -> 310 W, still below the 312.5 W overcurrent line so
    # the arc-fault channel is isolated): clamped ROC = 250/0.134 ≈ 1865.7
    # W/s > threshold -> trips. Unclamped it would read 2500 W/s.
    big = _warmed_node(60.0)
    big.pzem.set_load(310.0)
    big.core0_safety_step(sim_dt=0.1)
    assert big.shared_arc_fault is True, "a 250 W step must trip arc-fault"
    assert big.gpio18_relay_state is False
    assert big.shared_arc_fault_roc > FW_EDGE_ROC_THRESHOLD
    # The clamp itself is pinned: at sim_dt=0.1 the effective dt binds at its
    # 0.134 s lower edge (contract §1.5), never at the nominal 0.1 s.
    assert big.shared_arc_fault_roc == pytest.approx(250.0 / 0.134)

    # Contract §1.5 canonical projector case: a +300 W step reads 1875-2239
    # W/s (at sim_dt=0.1 the clamp binds at 0.134 s -> 300/0.134 ≈ 2238.8).
    # Overcurrent co-fires here (360 W > 312.5 W) exactly as on hardware.
    projector = _warmed_node(60.0)
    projector.pzem.set_load(360.0)
    projector.core0_safety_step(sim_dt=0.1)
    assert projector.shared_arc_fault is True
    assert 1875.0 <= projector.shared_arc_fault_roc <= 2239.0, (
        "projector-step ROC left the contract §1.5 range (1875-2239 W/s)"
    )


# ── 3. Hardware config profile ───────────────────────────────────────────


def test_hardware_config_matches_firmware_constants():
    """config/config.hardware.yaml is the rig profile: its safety ladder and
    single device must track the firmware constants (contract §1.1), and the
    RL opt-out (tier0 NEVER_SHED + rl cooldown) must stay present (Run 0.8)."""
    safety = _HW_CONFIG["system_safety"]
    assert safety["max_aggregate_wattage"] == FW_RATED_WATTS, (
        "max_aggregate_wattage drifted from RATED_WATTS — the monitor would "
        "fire ALERT_CRITICAL every second inside the band the firmware allows"
    )
    limits = safety["device_wattage_limits"]
    assert limits["node_bench_agg"] == FW_RATED_WATTS
    assert limits["default"] == FW_RATED_WATTS
    assert safety["critical_pct"] == FW_CRITICAL_PCT

    devices = _HW_CONFIG["devices"]
    assert set(devices) == {"node_bench_agg"}, (
        "the hardware profile declares exactly one node (spec scope S1)"
    )
    assert devices["node_bench_agg"]["tier0"] is True, (
        "node_bench_agg must be tier0/NEVER_SHED: the RL agent may not open "
        "the physical relay uncommanded"
    )
    assert "cooldown_seconds" in _HW_CONFIG.get("rl", {}), (
        "the hardware profile needs its own rl.cooldown_seconds — without it "
        "the RL agent falls back to its 15 s default"
    )
    # Feeds the pipeline's power subscription (see topic-symmetry test).
    assert _HW_CONFIG["mqtt"]["topics"]["reads"] == "home/sensor/+/power"


# ── 4. MQTT topic symmetry ───────────────────────────────────────────────


def _firmware_topics():
    """The five topic format strings main.cpp's setup() builds via snprintf."""
    return re.findall(
        r'snprintf\(\s*topic\w+,\s*sizeof\(topic\w+\),\s*"([^"]+)"', _MAIN_CPP
    )


def test_mqtt_topic_symmetry_firmware_to_pipeline():
    """Every topic the firmware builds must be covered by the pipeline's
    subscription/publish surface (contract §1.4). run_pipeline.py is checked
    as text — importing it would drag in the whole backend."""
    topics = set(_firmware_topics())
    expected = {
        "home/sensor/%s/power",
        "home/sensor/%s/telemetry",
        "home/plug/%s/command",
        "home/sensor/%s/status",
        "home/plug/%s/ack",
    }
    assert topics == expected, f"firmware topic set changed: {topics ^ expected}"

    run_block = re.search(r"self\.mqtt\.run\(\[(.*?)\]\)", _PIPELINE_PY, re.S)
    assert run_block, "self.mqtt.run([...]) subscription list not found in run_pipeline.py"
    subscriptions = run_block.group(1)
    # Power arrives via config mqtt.topics.reads (asserted above to be
    # home/sensor/+/power); the remaining subscriptions are pinned as literals.
    assert "topics']['reads']" in subscriptions, (
        "the mqtt.run list no longer subscribes the config reads topic "
        "(home/sensor/+/power)"
    )
    for sub in (
        "home/sensor/+/telemetry",
        "home/plug/+/ack",
        "home/ml/label",
        "home/sensor/+/status",
    ):
        assert sub in subscriptions, (
            f"pipeline no longer subscribes {sub} — the firmware publishes it "
            "(home/sensor/+/telemetry carries the PZEM V/I/PF frames)"
        )
    # Command reach: the pipeline must be able to publish home/plug/<id>/command.
    assert "home/plug/{device_id}/command" in _PIPELINE_PY, (
        "pipeline command publish site (home/plug/<id>/command) is gone"
    )


# ── 5. Status-string parity ──────────────────────────────────────────────


def test_status_and_ack_strings_present_in_firmware_and_twin():
    """The twin must emit the exact status/ACK strings the firmware emits
    (Run 0.3) — otherwise HIL evidence on those channels does not transfer."""
    for s in (
        "OVERCURRENT:",
        "SERVER_TIMEOUT",
        "ONLINE",
        "OFFLINE",
        "LOCKOUT_NACK",
        "ON_CONFIRMED",
        "OFF_CONFIRMED",
    ):
        assert s in _MAIN_CPP, f"status string {s!r} missing from main.cpp"
        assert s in _TWIN_PY, f"status string {s!r} missing from the twin"


# ── 6. Command semantics ─────────────────────────────────────────────────


def test_command_matching_is_exact_and_case_sensitive():
    """The firmware compares payloads with == "ON"/"OFF"/"WARNING"
    (main.cpp callback): case-sensitive, no strip()/upper() normalization.
    The twin must keep the same contract."""
    for cmd in ("ON", "OFF", "WARNING"):
        assert f'== "{cmd}"' in _MAIN_CPP, (
            f'main.cpp lost the exact == "{cmd}" comparison'
        )
        assert f'command == "{cmd}"' in _TWIN_PY, (
            f'twin lost the exact command == "{cmd}" comparison'
        )
    assert ".upper()" not in _MAIN_CPP and ".strip(" not in _MAIN_CPP, (
        "main.cpp normalizes command payloads — the contract is exact match"
    )


@pytest.mark.asyncio
async def test_non_exact_command_does_not_energize():
    """Behavioral pin of the exact-match semantics: a lowercase or padded
    payload must be ignored — no relay change, no ACK."""
    published = []

    async def publish(topic, payload):
        published.append((topic, payload))

    node = ESP32FirmwareNode(device_id="parity_probe", mqtt_publish_fn=publish)
    await node.handle_mqtt_command("on")
    await node.handle_mqtt_command(" ON")
    assert node.gpio18_relay_state is False, "a non-exact payload energized the relay"
    assert published == [], "a non-exact payload produced an ACK"


# ── 7. DEVICE_ID chain ───────────────────────────────────────────────────


def test_device_id_chain_matches_config():
    """EMS_DEVICE_ID (secrets.h) must name the one device the hardware
    profile declares — a mismatch means the backend silently ignores every
    reading this node publishes (contract §1.4)."""
    assert "node_bench_agg" in _HW_CONFIG["devices"]
    if not SECRETS_H.exists():
        pytest.skip("secrets.h absent (gitignored, local-only file)")
    m = re.search(r'#define\s+EMS_DEVICE_ID\s+"([^"]+)"', SECRETS_H.read_text())
    assert m, "EMS_DEVICE_ID #define not found in secrets.h"
    assert m.group(1) == "node_bench_agg", (
        f"EMS_DEVICE_ID is {m.group(1)!r} but the hardware profile declares "
        "only node_bench_agg — every reading would be silently ignored"
    )


# ── 8. Behavioral mini-checks (in-process, fast) ─────────────────────────


def test_overcurrent_cutoff_is_immediate_and_latches_for_core1():
    """Above CRITICAL_PCT x RATED_WATTS the relay must open in core 0 on the
    first sample (unconditional); the anti-thrashing lockout itself is taken
    by the core-1 tick that consumes the latch (main.cpp:226-231 + :427-437)."""
    node = ESP32FirmwareNode(device_id="parity_probe")
    node.set_relay(True)
    node.pzem.set_load(FW_RATED_WATTS * FW_CRITICAL_PCT + 50.0)
    node.core0_safety_step()
    assert node.gpio18_relay_state is False, "overcurrent must open the relay in core 0"
    assert node.shared_overcurrent_latch is True
    assert node.relay_locked is False, "the lockout belongs to the core-1 tick"
    # Cold baseline: the same step is inrush-suppressed on the arc-fault
    # channel (BASELINE_INRUSH_CEIL/INRUSH_HEADROOM), so only overcurrent fired.
    assert node.shared_arc_fault is False


@pytest.mark.asyncio
async def test_overcurrent_status_published_after_core1_tick():
    """The core-1 tick consumes the overcurrent latch, takes the lockout and
    publishes the OVERCURRENT:<watts> status alert (main.cpp:425-437)."""
    published = []

    async def publish(topic, payload):
        published.append((topic, payload))

    node = ESP32FirmwareNode(device_id="parity_probe", mqtt_publish_fn=publish)
    node.set_relay(True)
    node.pzem.set_load(FW_RATED_WATTS * FW_CRITICAL_PCT + 50.0)
    node.core0_safety_step()
    await node.core1_telemetry_tick(force_publish=True)
    status = [payload for topic, payload in published if topic == node.topic_status]
    assert any(p.startswith("OVERCURRENT:") for p in status), (
        "core-1 tick did not publish the OVERCURRENT: status alert"
    )
    assert node.relay_locked is True, "core-1 tick did not consume the latch"


@pytest.mark.asyncio
async def test_oversized_command_payload_is_dropped():
    """Payloads over 256 bytes are dropped before parsing (main.cpp:258-262):
    no relay change, no ACK."""
    published = []

    async def publish(topic, payload):
        published.append((topic, payload))

    node = ESP32FirmwareNode(device_id="parity_probe", mqtt_publish_fn=publish)
    node.set_relay(True)
    await node.handle_mqtt_command("A" * 257)
    assert node.gpio18_relay_state is True, "an oversized payload changed relay state"
    assert node.relay_locked is False
    assert published == []
