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

from src.hardware.esp32_firmware_sim import (
    PZEM_FAIL_TRIP_COUNT as TWIN_PZEM_FAIL_TRIP_COUNT,
    ESP32FirmwareNode,
    VirtualPZEM004T,
)

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
FW_PZEM_FAIL_TRIP_COUNT = _fw_int("int", "PZEM_FAIL_TRIP_COUNT")


# ── Structural parsing of main.cpp (code only, never prose) ──────────────
#
# main.cpp documents its safety invariants at length, and several of those
# comments QUOTE THE VERY CODE THEY REPLACED — `isnan`, and
# `powerWatts > criticalWatts && !relayLocked`. A regression that deletes the
# code would therefore still satisfy a raw substring search, because the prose
# explaining the old hole is still in the file. Every structural assertion
# below runs against comment-stripped, string-literal-blanked source so that
# "the firmware still does X" cannot be satisfied by "the firmware still talks
# about X".


def _strip_cpp_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def _blank_cpp_literals(code: str) -> str:
    """Blank string/char literal contents in place (length preserved), so
    braces and operators inside payload formats cannot confuse the block
    parsing below."""
    return re.sub(
        r"\"(?:[^\"\\\n]|\\.)*\"|'(?:[^'\\\n]|\\.)*'",
        lambda m: m.group()[0] + " " * (len(m.group()) - 2) + m.group()[-1],
        code,
    )


_MAIN_CPP_CODE = _strip_cpp_comments(_MAIN_CPP)
_MAIN_CPP_STRUCT = _blank_cpp_literals(_MAIN_CPP_CODE)


def _fn_body(name: str) -> str:
    """Whitespace-flattened body of `void name(...)` in main.cpp.

    Which CORE a safety check lives on is the whole question for the core-0 ->
    core-1 latches, so assertions must be able to say "in SafetySamplingTask"
    or "in loop", not merely "somewhere in the file".
    """
    m = re.search(rf"\bvoid\s+{name}\s*\([^)]*\)\s*\{{", _MAIN_CPP_STRUCT)
    assert m, f"void {name}(...) not found in {MAIN_CPP}"
    start = m.end() - 1
    depth = 0
    for i in range(start, len(_MAIN_CPP_STRUCT)):
        if _MAIN_CPP_STRUCT[i] == "{":
            depth += 1
        elif _MAIN_CPP_STRUCT[i] == "}":
            depth -= 1
            if depth == 0:
                return " ".join(_MAIN_CPP_STRUCT[start:i + 1].split())
    raise AssertionError(f"unbalanced braces in {name}() in {MAIN_CPP}")


def _guarded_block(body: str, statement: str):
    """The unique `if (<cond>) { <block> }` in `body` whose block performs
    `statement`, returned as (cond, block).

    This is what lets a test assert WHAT a cutoff is gated on rather than that
    the identifiers merely appear: dropping one condition from a safety `if`
    leaves every substring in the file intact.
    """
    hits = [
        m
        for m in re.finditer(r"if \((?P<cond>[^{};]*)\) \{(?P<blk>[^{}]*)\}", body)
        if statement in m.group("blk")
    ]
    assert len(hits) == 1, (
        f"expected exactly one `if (...) {{ ... }}` performing {statement!r} in "
        f"{MAIN_CPP}, found {len(hits)} — the safety branch was moved, "
        "duplicated or deleted"
    )
    return hits[0].group("cond"), hits[0].group("blk")



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
    """PF init 1.0 mirrors the POWER_FACTOR constant / the sharedPf
    initialiser in main.cpp."""
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
        # PZEM loss-of-measurement watchdog. The operator's ONLY signal that a
        # node has gone blind is this string: the relay is already open and
        # locked out for 5 minutes, and PZEM_FAULT is what distinguishes
        # "sensor/UART dead" from "overload" in the status stream. If either
        # side drops it the fault reports as silence.
        "PZEM_FAULT",
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
    by the core-1 tick that consumes the latch (the overcurrent cutoff in
    SafetySamplingTask() + the sharedOvercurrentLatch block in loop())."""
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
    publishes the OVERCURRENT:<watts> status alert (the sharedOvercurrentLatch
    block in loop() in main.cpp)."""
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
    """Payloads over 256 bytes are dropped before parsing (the
    MAX_MQTT_PAYLOAD guard in callback() in main.cpp):
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


# ── 9. Core-0 safety structure ───────────────────────────────────────────
#
# These pin the SHAPE of three core-0 safety mechanisms in main.cpp, not just
# the presence of their identifiers. The firmware cannot be compiled or run
# here (platformio is not installed), so static structure is the only thing
# standing between a plausible-looking edit and a re-opened hazard on an
# energised socket. Each test names the hazard its regression reopens.


def test_core0_rejects_every_non_finite_pzem_read():
    """Core 0's PZEM guard must reject ALL non-finite values on ALL FOUR
    reads, not just NaN.

    The library signals a failed Modbus frame with NAN, but a corrupted frame
    decoded into a float can land on +/-Inf. An Inf reaching lastWatts and the
    baseline ring poisons the arc-fault channel PERMANENTLY: every later
    `powerWatts > lastWatts` comparison against Inf is false, so dP/dt can
    never trip again on that node until it reboots. isnan() is a strict subset
    of !isfinite(), so a revert to isnan() is a silent loss of protection with
    no visible symptom.
    """
    body = _fn_body("SafetySamplingTask")
    m = re.search(r"if \((?P<cond>!isfinite\(.*?)\) \{", body)
    assert m, (
        "core 0's PZEM validity guard is no longer an `if (!isfinite(...))` "
        f"in SafetySamplingTask() in {MAIN_CPP} — a non-finite read can now "
        "reach lastWatts / the baseline ring and permanently disable dP/dt"
    )
    cond = m.group("cond")
    guarded = set(re.findall(r"!isfinite\((\w+)\)", cond))
    assert guarded == {"powerWatts", "pzemVoltage", "pzemCurrent", "pzemPf"}, (
        "core 0 must guard all four PZEM reads against non-finite values; "
        f"guarded = {sorted(guarded)}. An unguarded channel is an unguarded "
        "path into shared safety state."
    )
    assert "&&" not in cond, (
        "the PZEM guard was weakened from OR to AND — it would now skip the "
        "cycle only when EVERY read is bad, letting a single Inf through"
    )
    assert "isnan(" not in _MAIN_CPP_CODE, (
        f"a bare isnan() guard is back in {MAIN_CPP}. isnan() passes +/-Inf "
        "straight into the safety state it is supposed to protect — use "
        "!isfinite()."
    )
    # Twin parity: HIL evidence on this path only transfers if both sides
    # reject the same set of values.
    m = re.search(
        r"if not all\(math\.isfinite\(v\) for v in \(([^)]*)\)\)", _TWIN_PY
    )
    assert m, "the twin's all(math.isfinite(...)) PZEM guard is gone"
    assert {s.strip() for s in m.group(1).split(",") if s.strip()} == {
        "power_w",
        "voltage",
        "current",
        "pf",
    }, "the twin guards a different set of PZEM reads than main.cpp"


def test_overcurrent_latch_replaced_the_core1_level_check():
    """The core-0 -> core-1 overcurrent LATCH must exist, and the old
    level-triggered core-1 check must be GONE.

    This is the specific hole the latch closes, and the one a well-meaning
    "simplification" would silently reopen: core 0 opens the relay on the very
    sample that offends, which collapses the reading. A spike that tripped
    core 0 and cleared before core 1's next pass therefore failed
    `powerWatts > criticalWatts` when core 1 finally looked, so the 5-minute
    anti-thrashing lockout was NEVER TAKEN — and the next `ON` re-closed the
    relay straight into the fault, cycling the contacts into a live overload.
    The latch is set by core 0 on every 100 ms sample it trips on, so it
    strictly dominates anything the level check could observe.
    """
    core0 = _fn_body("SafetySamplingTask")
    core1 = _fn_body("loop")

    assert re.search(
        r"volatile bool\s+sharedOvercurrentLatch\s*=\s*false\s*;", _MAIN_CPP_STRUCT
    ), "the volatile bool sharedOvercurrentLatch declaration is gone from main.cpp"

    # Raised by core 0, in the same branch that opens the relay.
    cond, blk = _guarded_block(core0, "sharedOvercurrentLatch = true;")
    assert re.search(r"powerWatts\s*>\s*criticalWatts", cond), (
        "the overcurrent latch is no longer raised by the "
        f"`powerWatts > criticalWatts` cutoff in core 0: cond = {cond!r}"
    )
    assert "setRelay(false);" in blk, (
        "core 0 raises the overcurrent latch without opening the relay — the "
        "cutoff must be complete before core 1 ever hears about it"
    )
    assert "relayLocked" not in cond, (
        "the core-0 overcurrent cutoff must stay UNCONDITIONAL — gating it on "
        "the core-1 lockout state would skip the cutoff during a lockout"
    )

    # Consumed and acknowledged by core 1.
    assert "sharedOvercurrentLatch" in core1, (
        "core 1 no longer consumes the overcurrent latch: core 0 opens the "
        "relay but the 5-minute lockout is never taken, so the next `ON` "
        "re-closes into the fault"
    )
    assert "sharedOvercurrentLatch = false;" in core1, (
        "core 1 reads the overcurrent latch but never acknowledges it — the "
        "node would re-lock on every subsequent loop() pass"
    )

    # The regression itself: the level check must not come back.
    assert not re.search(
        r"(shared)?[Pp]owerWatts\s*>\s*criticalWatts", core1
    ), (
        "the LEVEL-triggered overcurrent check is back in loop(). It misses "
        "any spike that core 0 already cut off (the relay-open collapses the "
        "reading before core 1 looks), so the lockout is skipped and the next "
        "`ON` re-energises a faulted circuit. Consume sharedOvercurrentLatch "
        "instead."
    )


def test_pzem_fail_trip_count_matches_twin():
    """PZEM_FAIL_TRIP_COUNT is a hardware tuning knob (blind-window length),
    so main.cpp and the twin must move together in one commit — exactly like
    the RATED_WATTS drift guard above.

    Drift here is invisible and corrosive: every twin-based trip-timing result
    (HIL runs, the stress scripts, the bring-up expectations) would be
    measuring a different blind window than the firmware actually tolerates.
    """
    assert TWIN_PZEM_FAIL_TRIP_COUNT == FW_PZEM_FAIL_TRIP_COUNT, (
        f"twin PZEM_FAIL_TRIP_COUNT={TWIN_PZEM_FAIL_TRIP_COUNT} but firmware "
        f"PZEM_FAIL_TRIP_COUNT={FW_PZEM_FAIL_TRIP_COUNT} — the twin models a "
        f"{TWIN_PZEM_FAIL_TRIP_COUNT * 0.1:.1f}s blind window while the "
        f"firmware tolerates {FW_PZEM_FAIL_TRIP_COUNT * 0.1:.1f}s"
    )
    assert 0 < FW_PZEM_FAIL_TRIP_COUNT <= 100, (
        f"PZEM_FAIL_TRIP_COUNT={FW_PZEM_FAIL_TRIP_COUNT} leaves the socket "
        f"energised and unmeasured for {FW_PZEM_FAIL_TRIP_COUNT * 0.1:.1f}s "
        "with BOTH overcurrent and arc-fault blind. Raise past ~100 (10 s) "
        "only with evidence, and never to 0 (which would trip on any read)."
    )


def test_pzem_watchdog_trip_requires_all_three_conditions():
    """The loss-of-measurement watchdog must stay gated on all three
    conditions, and the counter must saturate and rewind.

    Each condition carries a distinct cost if dropped:
      - pzemFailCount >= PZEM_FAIL_TRIP_COUNT: without it a single transient
        Modbus CRC error opens the relay and locks the node out for 5 minutes.
      - relayClosed: without it a node whose PZEM is dead re-trips on every
        100 ms sample, spamming PZEM_FAULT and restarting the lockout forever
        on a relay that is already open — there is no energised socket to
        protect.
      - pzemEverValid: without it documented bring-up GATE 5 (USB power, no
        mains, PZEM NaN) self-lockouts, and GATE 7 (dry relay close, no mains)
        becomes UNPERFORMABLE — the saturated counter re-opens the contacts
        within 100 ms of the close, before COM-NO continuity can be metered
        (claude_debug/BRINGUP_RUNBOOK.md Gates 5/7).
    Conversely the trip must remain a conjunction: an `||` here would open the
    relay on any one of them.
    """
    core0 = _fn_body("SafetySamplingTask")
    cond, blk = _guarded_block(core0, "sharedPzemFault = true;")
    for token, why in (
        (
            "pzemFailCount >= PZEM_FAIL_TRIP_COUNT",
            "a single transient bad read would now open the relay and lock the "
            "node out for 5 minutes",
        ),
        (
            "pzemEverValid",
            "bring-up GATE 5 self-lockouts and GATE 7 (dry close, no mains) "
            "becomes unperformable — the contacts re-open before continuity "
            "can be metered",
        ),
        (
            "relayClosed",
            "an already-open relay re-trips every 100 ms, restarting the "
            "5-minute lockout forever and spamming PZEM_FAULT",
        ),
    ):
        assert token in cond, (
            f"the PZEM watchdog trip is no longer gated on `{token}`: "
            f"cond = {cond!r}. Consequence: {why}."
        )
    assert "||" not in cond, (
        f"the PZEM watchdog trip conditions are OR-ed: cond = {cond!r}. All "
        "three must hold — an OR opens the relay on any one of them."
    )
    assert "setRelay(false);" in blk, (
        "the PZEM watchdog raises sharedPzemFault without opening the relay — "
        "core 0 must complete the cutoff itself, with zero network dependency"
    )

    # The counter contract the three conditions rest on.
    assert re.search(
        r"if \(pzemFailCount < PZEM_FAIL_TRIP_COUNT\) pzemFailCount\+\+;", core0
    ), (
        "the blind-read counter no longer saturates at PZEM_FAIL_TRIP_COUNT. "
        "An unbounded int on a permanently dead sensor is signed-overflow UB "
        "(~2.5 s per 1000 samples to INT_MAX), and saturation is also what "
        "makes the node re-trip within ONE sample of the next `ON` instead of "
        "waiting another 3 s blind."
    )
    assert "pzemFailCount = 0;" in core0, (
        "the blind-read counter is never reset — the watchdog must count "
        "CONSECUTIVE failures, not lifetime ones, or a healthy node "
        "accumulates its way to a trip"
    )
    assert "pzemEverValid = true;" in core0, (
        "pzemEverValid is never set, so the watchdog can never arm: a real "
        "dead-PZEM fault on a live circuit would never trip"
    )

    # relayClosed is only trustworthy if setRelay() maintains it.
    assert "relayClosed = on;" in _fn_body("setRelay"), (
        "setRelay() no longer records relayClosed, so the PZEM watchdog's "
        "arming condition is stale — a disarmed watchdog on an energised "
        "socket, or a re-trip loop on an open one"
    )


def test_runtime_relay_actuation_has_one_owner_and_health_gate():
    """Core 1 queues commands; Core 0 alone writes the runtime relay."""
    callback = _fn_body("callback")
    core0 = _fn_body("SafetySamplingTask")

    assert "setRelay(" not in callback, (
        "MQTT callback regained direct GPIO authority; it must queue a request"
    )
    assert "pendingRelayOn = true;" in callback
    assert "pendingRelayOff = true;" in callback
    assert "setRelay(true);" in core0
    assert "measurementFresh" in core0
    assert "safetyInhibit" in core0
    assert re.search(
        r"if \(safetyTaskRunning\s*&&\s*!safetyInhibit\s*&&\s*"
        r"measurementFresh\s*&&\s*!relayLocked\)",
        core0,
    ), "Core-0 ON consumption lost its health/inhibit/lockout gate"


def test_pzem_watchdog_tracks_successful_read_age_and_task_creation():
    """H3 must not rely only on a getter-loop count or unchecked task startup."""
    code = _MAIN_CPP_STRUCT
    assert "PZEM_MAX_BLIND_MS" in code
    assert "lastSuccessfulPzemReadMs" in code
    assert re.search(
        r"lastSuccessfulPzemReadMs\s*=\s*nowMs\s*;", code
    ), "successful PZEM reads do not update a freshness timestamp"
    assert "(unsigned long)(nowMs - lastSuccessfulPzemReadMs)" in code
    assert ">= PZEM_MAX_BLIND_MS" in code, (
        "watchdog does not evaluate elapsed successful-read age"
    )
    assert "BaseType_t safetyTaskResult" in code
    assert "safetyTaskResult != pdPASS" in code
