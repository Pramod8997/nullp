"""Shared ESP32 diagnostic telemetry wire contract.

The firmware publishes a JSON object every ten seconds on
``home/sensor/{device_id}/telemetry``.  Device identity is carried by the MQTT
topic; the payload deliberately has no source timestamp or sequence number.
Consumers attach their receive time when they ingest a frame.

Wire fields are active voltage (``v``, volts), current (``i``, amps), active
power (``w``, watts), and power factor (``pf``, unitless).  All four fields are
required finite JSON numbers.  Unknown fields are ignored for forward
compatibility, while missing, boolean, string, non-finite, or malformed values
are rejected.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any


TELEMETRY_KEYS = ("v", "i", "w", "pf")
TELEMETRY_PRECISION = {"v": 1, "i": 2, "w": 1, "pf": 2}


class TelemetryContractError(ValueError):
    """Raised when an ESP32 diagnostic frame violates the wire contract."""


def parse_telemetry_frame(payload: Any) -> dict[str, float]:
    """Decode and validate one ESP32 diagnostic frame.

    The returned mapping contains only the four contract fields.  This keeps
    untrusted extension keys out of downstream state and WebSocket messages.
    """
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TelemetryContractError("payload is not valid UTF-8") from exc

    if isinstance(payload, str):
        try:
            frame = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise TelemetryContractError("payload is not valid JSON") from exc
    else:
        frame = payload

    if not isinstance(frame, Mapping):
        raise TelemetryContractError("payload must be a JSON object")

    missing = [key for key in TELEMETRY_KEYS if key not in frame]
    if missing:
        raise TelemetryContractError(
            f"missing required telemetry field(s): {', '.join(missing)}"
        )

    values: dict[str, float] = {}
    for key in TELEMETRY_KEYS:
        raw = frame[key]
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise TelemetryContractError(f"{key} must be a JSON number")
        value = float(raw)
        if not math.isfinite(value):
            raise TelemetryContractError(f"{key} must be finite")
        values[key] = value
    return values


def encode_telemetry_frame(
    voltage: float, current: float, power: float, power_factor: float
) -> str:
    """Format diagnostics exactly like the firmware's ``snprintf`` payload."""
    values = parse_telemetry_frame(
        {"v": voltage, "i": current, "w": power, "pf": power_factor}
    )
    return (
        f'{{"v":{values["v"]:.1f},"i":{values["i"]:.2f},'
        f'"w":{values["w"]:.1f},"pf":{values["pf"]:.2f}}}'
    )
