import json
import asyncio
import hmac
import logging
import logging.handlers
import math
import time
import os
import io
import csv
import random
from datetime import datetime, timezone
import re
from typing import List, Dict, Any, Optional, TypedDict
from contextlib import asynccontextmanager

import aiosqlite
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    HTTPException, Depends, Header,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
import aiomqtt


# ─── Issue #17: Structured JSON Logging ─────────────────────────────
class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production observability."""
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


json_handler = logging.StreamHandler()
json_handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[json_handler])
logger = logging.getLogger(__name__)


# MQTT is an untrusted transport boundary.  Keep limits explicit and shared
# by the decoder, event models, and REST validators so malformed frames cannot
# become unbounded state or expensive downstream work.
MAX_MQTT_PAYLOAD_BYTES = 256 * 1024
MAX_DEVICE_ID_LENGTH = 64
MAX_POWER_WATTS = 50_000.0
MAX_TRACKED_DEVICES = 200
DEVICE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$"
VALID_RELAY_COMMANDS = {"ON", "OFF", "WARNING"}


def _finite_number(value: Any, *, minimum: float, maximum: float, field_name: str) -> float:
    """Validate an event number without accepting bool/string coercion."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a JSON number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    if value < minimum or value > maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}")
    return value


def _validate_device_id(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_DEVICE_ID_LENGTH:
        raise ValueError("device_id must be a non-empty identifier of at most 64 characters")
    import re
    if re.fullmatch(DEVICE_ID_PATTERN, value) is None:
        raise ValueError("device_id contains unsupported characters")
    return value


class _StrictEventModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _validate_segment_list(value: List[List[float]]) -> List[List[float]]:
    if len(value) > 100:
        raise ValueError("Maximum 100 segments allowed")
    for index, segment in enumerate(value):
        if len(segment) != 128:
            raise ValueError(f"Segment {index} must have exactly 128 dimensions")
        for sample in segment:
            if isinstance(sample, bool) or not isinstance(sample, (int, float)):
                raise ValueError(f"Segment {index} contains a non-numeric sample")
            sample = float(sample)
            if not math.isfinite(sample) or sample < 0.0:
                raise ValueError(f"Segment {index} contains an invalid power sample")
        if max(segment) < 20.0:
            raise ValueError(f"Segment {index} never exceeds 20 W")
    return value

# ─── Issue #18: Audit Logger ────────────────────────────────────────
audit_logger = logging.getLogger("ems.audit")
audit_handler = logging.handlers.RotatingFileHandler(
    "safety_events.log", maxBytes=10_000_000, backupCount=5,
)
audit_handler.setFormatter(JSONFormatter())
audit_logger.addHandler(audit_handler)
audit_logger.setLevel(logging.INFO)


# ─── Issue #21: TypedDict for system_state ──────────────────────────
class DeviceState(TypedDict, total=False):
    power: float
    state: str
    classification: str
    confidence: float
    pmv: float
    last_seen: float
    relay_state: str
    last_ack: str
    command_id: int


class SystemState(TypedDict):
    devices: Dict[str, DeviceState]
    phantom_loads: Dict[str, float]
    total_phantom: float
    pmv_score: float
    analytics: Dict[str, Any]
    active_mitigations: List[Dict[str, Any]]
    pipeline_status: str
    pending_labels: List[Dict[str, Any]]
    low_confidence_log: List[Dict[str, Any]]
    safety_warnings: List[Dict[str, Any]]


# ─── Shared State ────────────────────────────────────────────────────
# Updated by the MQTT listener, read by REST endpoints and WebSocket broadcasts.
system_state: SystemState = {
    "devices": {},          # device_id -> {power, state, classification, confidence, last_seen}
    "phantom_loads": {},    # device_id -> watts
    "total_phantom": 0.0,
    "pmv_score": 0.0,
    "analytics": {},        # daily summary
    "active_mitigations": [],
    "pipeline_status": "initializing",
    "pending_labels": [],   # devices awaiting user label
    "low_confidence_log": [],  # recent low-confidence events
    "safety_warnings": [],  # recent safety warnings
}

# Issue #2: asyncio.Lock for compound state mutations
state_lock = asyncio.Lock()

# Issue #4: Separate dict for WebSocket throttle timestamps (not in system_state)
_ws_throttle: Dict[str, float] = {}

# Fix §4.3.3: WebSocket power aggregation buffer.
# Instead of broadcasting each 1Hz power reading individually (which floods
# browsers at 100 devices = 100 msg/sec), readings are accumulated here and
# broadcast as a single batched update every 1 second.
_ws_power_buffer: Dict[str, float] = {}

# MQTT delivery is at-least-once. These watermarks prevent retained or
# out-of-order events from rolling the dashboard back to an older state.
_device_event_watermarks: Dict[str, float] = {}


def _event_epoch(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Normalize numeric or ISO-8601 timestamps to finite UTC epoch seconds."""
    if value is None or isinstance(value, bool):
        return default
    try:
        if isinstance(value, (int, float)):
            result = float(value)
        else:
            text = str(value).strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            result = parsed.astimezone(timezone.utc).timestamp()
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default

# API-side state is fed by untrusted MQTT topic names.  Keep the fan-out and
# snapshot stores bounded even when a compromised/buggy publisher invents
# device IDs faster than the normal one-second aggregation loop can flush.
MAX_API_DEVICES = 200
API_DEVICE_TTL_SECONDS = 3600.0
MAX_MQTT_EVENT_BYTES = 256 * 1024
MAX_WS_QUEUE_SIZE = 64
_api_device_last_seen: Dict[str, float] = {}


# ─── Issue #24: Pydantic Models for MQTT Event Validation ───────────
class DeviceStatusEvent(BaseModel):
    type: str = "DEVICE_STATUS"
    device_id: str = ""
    power: float = 0
    state: str = "unknown"
    classification: str = "unknown"
    confidence: float = 0
    pmv: float = 0
    timestamp: Optional[Any] = None

    @field_validator("device_id")
    @classmethod
    def validate_device_id(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value) > 64 or "/" in value:
            raise ValueError("device_id must be a non-empty MQTT-safe identifier")
        return value


class PhantomLoadEvent(BaseModel):
    type: str = "PHANTOM_LOAD"
    loads: Dict[str, float] = {}
    total: float = 0


class AnalyticsUpdateEvent(BaseModel):
    type: str = "ANALYTICS_UPDATE"
    summary: Dict[str, Any] = {}


class PMVUpdateEvent(BaseModel):
    type: str = "PMV_UPDATE"
    pmv: float = 0


class LabelRequestEvent(BaseModel):
    type: str = "LABEL_REQUEST"
    device_id: str = ""
    power: float = 0
    confidence: float = 0
    # Raw 128-sample power windows in watts. This is what enrollment consumes;
    # without it the dashboard has nothing valid to submit and falls back to
    # sending the embedding, which enrolls a meaningless prototype.
    segments: List[List[float]] = Field(default_factory=list)
    embedding: List[float] = Field(default_factory=list)
    suggested_label: str = ""
    message: str = ""

    @field_validator("segments")
    @classmethod
    def validate_segments(cls, value):
        if len(value) > 100:
            raise ValueError("too many segments")
        for segment in value:
            if len(segment) != 128 or not all(math.isfinite(x) for x in segment):
                raise ValueError("segments must contain finite 128-sample windows")
        return value


class LowConfidenceEvent(BaseModel):
    type: str = "LOW_CONFIDENCE"
    device_id: str = ""
    classified_as: str = ""
    confidence: float = 0
    threshold: float = 0.90
    message: str = ""


class SafetyEventModel(BaseModel):
    type: str
    device_id: str = ""
    severity: str = "warning"
    message: str = ""


class ActionEventModel(BaseModel):
    type: str
    action: str = ""
    device_id: str = ""
    pmv: float = 0


# ─── WebSocket Manager ──────────────────────────────────────────────
# Issue #3: Lock-protected connection list with concurrent broadcast
# Issue #16: Per-client backpressure via snapshot + gather
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
        self._queues: Dict[WebSocket, asyncio.Queue] = {}
        self._sender_tasks: Dict[WebSocket, asyncio.Task] = {}
        self.max_connections = 256

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            if len(self.active_connections) >= self.max_connections:
                await websocket.close(code=1013, reason="WebSocket capacity reached")
                return False
            self.active_connections.append(websocket)
            queue = asyncio.Queue(maxsize=MAX_WS_QUEUE_SIZE)
            self._queues[websocket] = queue
            self._sender_tasks[websocket] = asyncio.create_task(
                self._sender_loop(websocket, queue),
                name="ws-sender",
            )
            total = len(self.active_connections)
        logger.info("WebSocket client connected. Total: %d", total)
        return True

    async def _sender_loop(self, websocket: WebSocket, queue: asyncio.Queue) -> None:
        """Serialize writes per client and evict clients that stop draining.

        A shared broadcast task must never await a slow socket.  The bounded
        queue deliberately drops the oldest update when a browser falls
        behind; the next power batch/init state repairs the client's snapshot.
        """
        try:
            while True:
                message = await queue.get()
                try:
                    await asyncio.wait_for(
                        websocket.send_json(message), timeout=0.5
                    )
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("WebSocket sender stopped: %s", exc)
            await self.disconnect(websocket, _from_sender=True)

    async def disconnect(self, websocket: WebSocket, *, _from_sender: bool = False):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            queue = self._queues.pop(websocket, None)
            sender = self._sender_tasks.pop(websocket, None)
            total = len(self.active_connections)
        if sender is not None and sender is not asyncio.current_task() and not sender.done():
            sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass
        if queue is not None:
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break
        logger.info("WebSocket client disconnected. Total: %d", total)

    async def broadcast(self, message: dict):
        """Enqueue one bounded update per client without blocking the producer."""
        async with self._lock:
            queues = list(self._queues.items())

            for websocket, queue in queues:
                if queue.full():
                    try:
                        queue.get_nowait()
                        queue.task_done()
                    except asyncio.QueueEmpty:
                        pass
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    # A concurrent sender can refill the queue between the
                    # full check and put_nowait.  Drop this update rather than
                    # allowing an unbounded waiter to retain the payload.
                    logger.debug("Dropping WebSocket update for slow client")

    async def send(self, websocket: WebSocket, message: dict) -> bool:
        """Queue a per-client message, preserving ordering with broadcasts."""
        async with self._lock:
            queue = self._queues.get(websocket)
            if queue is None:
                return False
            if queue.full():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    return False
            try:
                queue.put_nowait(message)
                return True
            except asyncio.QueueFull:
                return False

    async def close_all(self) -> None:
        async with self._lock:
            connections = list(self.active_connections)
        for websocket in connections:
            await self.disconnect(websocket)

    def connection_count(self) -> int:
        return len(self.active_connections)


manager = ConnectionManager()

# ─── Issue #13: Global MQTT client reference ────────────────────────
_shared_mqtt_client: Optional[aiomqtt.Client] = None
_shared_db: Optional[aiosqlite.Connection] = None
_mqtt_listener_alive = False
_mqtt_subscriptions_ready = False


def _evict_api_devices(now: Optional[float] = None) -> None:
    """Bound API device/index/batch state under hostile topic churn."""
    now = time.time() if now is None else now
    stale = {
        device_id for device_id, seen in _api_device_last_seen.items()
        if now - seen > API_DEVICE_TTL_SECONDS
    }
    if len(_api_device_last_seen) > MAX_API_DEVICES:
        oldest = sorted(_api_device_last_seen, key=_api_device_last_seen.get)
        stale.update(oldest[:len(_api_device_last_seen) - MAX_API_DEVICES])
    for device_id in stale:
        _api_device_last_seen.pop(device_id, None)
        system_state["devices"].pop(device_id, None)
        _ws_power_buffer.pop(device_id, None)


def _mqtt_connect_kwargs() -> dict:
    """
    Broker address, port and credentials for every aiomqtt client in this module.

    Both call sites previously hardcoded `port=1883` and passed no credentials,
    so against a broker with `allow_anonymous false` — which is what this repo's
    own mosquitto/config/mosquitto.conf sets and docker-compose.yml mounts — the
    dashboard bridge and the label-publish path both failed to authenticate.
    docker-compose.yml already exports MQTT_USERNAME/MQTT_PASSWORD; nothing read
    them. None stays None, so an anonymous dev broker behaves exactly as before.
    """
    return {
        "hostname": os.environ.get("MQTT_BROKER", "localhost"),
        "port": int(os.environ.get("MQTT_PORT", "1883")),
        "username": os.environ.get("MQTT_USERNAME") or None,
        "password": os.environ.get("MQTT_PASSWORD") or None,
    }


# ─── Issue #7: API Key Authentication ───────────────────────────────
async def verify_api_key(x_api_key: str = Header(None)):
    """Dependency that checks X-API-Key header against EMS_API_KEY env var."""
    expected = os.environ.get("EMS_API_KEY")
    # compare_digest on bytes: constant-time and no ASCII-only TypeError on
    # a hostile non-ASCII header value.
    if not expected or not hmac.compare_digest(
        (x_api_key or "").encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ─── MQTT → WebSocket Bridge ────────────────────────────────────────
async def mqtt_listener_task():
    """
    Subscribes to all EMS MQTT topics and forwards messages to
    connected WebSocket clients AND updates shared system_state.
    """
    global _shared_mqtt_client, _mqtt_listener_alive, _mqtt_subscriptions_ready
    attempt = 0  # Issue #6: Exponential backoff counter

    while True:
        try:
            # Bug 4.5 fix: Use MQTT_BROKER env var instead of hardcoded localhost
            async with aiomqtt.Client(**_mqtt_connect_kwargs()) as client:
                _shared_mqtt_client = client
                _mqtt_listener_alive = True
                _mqtt_subscriptions_ready = False
                logger.info("FastAPI MQTT bridge connected.")
                system_state["pipeline_status"] = "mqtt_subscribing"
                attempt = 0  # Reset backoff on successful connection

                await client.subscribe("home/sensor/+/power")
                await client.subscribe("home/plug/+/command")
                await client.subscribe("home/ui/events")
                _mqtt_subscriptions_ready = True
                system_state["pipeline_status"] = "connected"

                async for message in client.messages:
                    topic = str(message.topic)
                    raw_payload = message.payload
                    if isinstance(raw_payload, (bytes, bytearray)):
                        if len(raw_payload) > MAX_MQTT_EVENT_BYTES:
                            logger.warning("Dropping oversized MQTT frame on %s", topic)
                            continue
                    elif len(str(raw_payload).encode("utf-8")) > MAX_MQTT_EVENT_BYTES:
                        logger.warning("Dropping oversized MQTT frame on %s", topic)
                        continue
                    try:
                        payload = (raw_payload.decode()
                                   if isinstance(raw_payload, bytes)
                                   else str(raw_payload))
                    except (UnicodeDecodeError, TypeError) as exc:
                        logger.warning(
                            "Dropping undecodable MQTT frame on %s: %s", topic, exc
                        )
                        continue

                    # ── UI Events (structured JSON from the pipeline) ──
                    if "home/ui/events" in topic:
                        try:
                            event_data = json.loads(payload)
                            event_type = event_data.get("type", "")

                            if event_type == "DEVICE_STATUS":
                                evt = DeviceStatusEvent(**event_data)
                                did = evt.device_id
                                # Issue #5: Standardize timestamp to epoch float.
                                # A supplied zero is valid and must be preserved.
                                ts = _event_epoch(evt.timestamp, time.time())
                                previous_ts = _device_event_watermarks.get(did)
                                if previous_ts is not None and ts <= previous_ts:
                                    logger.debug(
                                        "Ignoring duplicate/stale DEVICE_STATUS for %s "
                                        "(%.3f <= %.3f)", did, ts, previous_ts
                                    )
                                    continue
                                _device_event_watermarks[did] = ts
                                system_state["devices"][did] = {
                                    "power": evt.power,
                                    "state": evt.state,
                                    "classification": evt.classification,
                                    "confidence": evt.confidence,
                                    "pmv": evt.pmv,
                                    "last_seen": ts,
                                }
                                _api_device_last_seen[did] = time.time()
                                _evict_api_devices()
                            elif event_type == "PHANTOM_LOAD":
                                evt = PhantomLoadEvent(**event_data)
                                system_state["phantom_loads"] = evt.loads
                                system_state["total_phantom"] = evt.total
                            elif event_type == "ANALYTICS_UPDATE":
                                evt = AnalyticsUpdateEvent(**event_data)
                                system_state["analytics"] = evt.summary
                            elif event_type == "PMV_UPDATE":
                                evt = PMVUpdateEvent(**event_data)
                                system_state["pmv_score"] = evt.pmv

                            elif event_type == "LABEL_REQUEST":
                                evt = LabelRequestEvent(**event_data)
                                label_entry = {
                                    "device_id": evt.device_id,
                                    "power": evt.power,
                                    "confidence": evt.confidence,
                                    # The raw power windows enrollment needs. A
                                    # dashboard that reloads and reads this from
                                    # /api/pending-labels must be able to submit
                                    # a label without the original WebSocket
                                    # frame, so the watts have to be kept here.
                                    "segments": evt.segments,
                                    # Bug 3.2 fix: Preserve the 128D embedding
                                    # (display/clustering only — NOT enrollment)
                                    "embedding": evt.embedding,
                                    "suggested_label": evt.suggested_label,
                                    "message": evt.message,
                                    "timestamp": time.time(),
                                }
                                async with state_lock:
                                    # Reconnects can replay the same request;
                                    # retain one current entry per device.
                                    system_state["pending_labels"] = [
                                        p for p in system_state["pending_labels"]
                                        if p.get("device_id") != evt.device_id
                                    ]
                                    system_state["pending_labels"].append(label_entry)
                                    system_state["pending_labels"] = system_state["pending_labels"][-50:]
                                # Open-set hook: buffer the raw windows behind
                                # this unrecognised signature so
                                # POST /api/v1/appliances/label-unrecognized
                                # can enroll them when the user names it.
                                await _capture_signature(evt.device_id, label_entry)
                                logger.info(f"📋 LABEL_REQUEST: {evt.device_id}")
                                audit_logger.info(json.dumps({
                                    "event": "LABEL_REQUEST",
                                    "device_id": evt.device_id,
                                    "confidence": evt.confidence,
                                }))

                            elif event_type == "LOW_CONFIDENCE":
                                evt = LowConfidenceEvent(**event_data)
                                lc_entry = {
                                    "device_id": evt.device_id,
                                    "classified_as": evt.classified_as,
                                    "confidence": evt.confidence,
                                    "threshold": evt.threshold,
                                    "message": evt.message,
                                    "timestamp": time.time(),
                                }
                                async with state_lock:
                                    system_state["low_confidence_log"].append(lc_entry)
                                    system_state["low_confidence_log"] = system_state["low_confidence_log"][-100:]

                            elif event_type in ["SAFETY_WARNING", "SAFETY_CUTOFF"]:
                                evt = SafetyEventModel(**event_data)
                                sw_entry = {
                                    "device_id": evt.device_id,
                                    "severity": evt.severity,
                                    "message": evt.message,
                                    "timestamp": time.time(),
                                }
                                async with state_lock:
                                    system_state["safety_warnings"].append(sw_entry)
                                    system_state["safety_warnings"] = system_state["safety_warnings"][-50:]
                                audit_logger.warning(json.dumps({
                                    "event": event_type,
                                    "device_id": evt.device_id,
                                    "severity": evt.severity,
                                    "message": evt.message,
                                }))

                            elif event_type in ["EMPATHY_ACTION", "RL_ACTION"]:
                                evt = ActionEventModel(**event_data)
                                async with state_lock:
                                    system_state["active_mitigations"].append({
                                        "type": event_type,
                                        "action": evt.action,
                                        "device_id": evt.device_id,
                                        "pmv": evt.pmv,
                                        "timestamp": time.time(),
                                    })
                                    system_state["active_mitigations"] = system_state["active_mitigations"][-50:]
                                audit_logger.info(json.dumps({
                                    "event": event_type,
                                    "action": evt.action,
                                    "device_id": evt.device_id,
                                }))

                            await manager.broadcast(event_data)
                        except Exception as exc:
                            # A malformed or schema-invalid UI event is an
                            # untrusted frame, not a listener-fatal error.
                            # Keep the bridge alive and leave shared state
                            # unchanged for that frame.
                            logger.warning(
                                "Dropping invalid UI event on %s: %s", topic, exc
                            )

                    # ── Raw Power Readings ──
                    elif "/power" in topic:
                        device_id = topic.split("/")[-2]
                        try:
                            power_event_ts = None
                            if payload.lstrip().startswith("{"):
                                power_data = json.loads(payload)
                                raw_power = power_data.get(
                                    "power", power_data.get("watts")
                                )
                                power_watts = float(raw_power)
                                power_event_ts = _event_epoch(
                                    power_data.get("timestamp")
                                )
                            else:
                                power_watts = float(payload)
                            # NaN/Inf payloads parse cleanly as floats; never
                            # let them into the state or the WS broadcast.
                            if not math.isfinite(power_watts):
                                raise ValueError("non-finite power")
                            if power_event_ts is not None:
                                previous_ts = _device_event_watermarks.get(device_id)
                                if previous_ts is not None and power_event_ts <= previous_ts:
                                    continue
                                _device_event_watermarks[device_id] = power_event_ts
                            if device_id not in system_state["devices"]:
                                system_state["devices"][device_id] = {}
                            system_state["devices"][device_id]["power"] = power_watts

                            # Fix §4.3.3: Accumulate into aggregation buffer
                            # instead of broadcasting each reading individually.
                            # The ws_aggregation_task flushes this buffer every 1s.
                            _ws_power_buffer[device_id] = power_watts
                            _api_device_last_seen[device_id] = time.time()
                            _evict_api_devices()
                        except ValueError:
                            # Issue #11: Log instead of silently swallowing
                            logger.debug(f"Non-numeric power payload on {topic}: {payload[:50]}")

                    # ── Relay acknowledgements ──
                    elif "/ack" in topic:
                        device_id = topic.split("/")[-2]
                        match = re.fullmatch(
                            r"(ON_CONFIRMED|OFF_CONFIRMED|LOCKOUT_NACK|STALE_COMMAND)\|([1-9][0-9]{0,19})",
                            payload,
                        )
                        if match is None:
                            logger.warning("Dropping invalid relay ACK on %s", topic)
                            continue
                        ack, command_id_text = match.groups()
                        command_id = int(command_id_text)
                        relay_state = (
                            "ON" if ack == "ON_CONFIRMED" else
                            "OFF" if ack == "OFF_CONFIRMED" else
                            "LOCKED" if ack == "LOCKOUT_NACK" else
                            "STALE"
                        )
                        system_state["devices"].setdefault(device_id, {})
                        system_state["devices"][device_id].update({
                            "relay_state": relay_state,
                            "last_ack": ack,
                            "command_id": command_id,
                            "last_seen": time.time(),
                        })
                        await manager.broadcast({
                            "type": "HARDWARE_ACK",
                            "device_id": device_id,
                            "ack": ack,
                            "command_id": command_id,
                            "relay_state": relay_state,
                            "accepted": ack in {"ON_CONFIRMED", "OFF_CONFIRMED"},
                        })

                    # ── Relay command observations ──
                    elif "/command" in topic:
                        device_id = topic.split("/")[-2]
                        if re.fullmatch(
                            r"(?:ON|OFF)\|[1-9][0-9]{0,19}", payload
                        ) is None:
                            logger.warning("Dropping invalid relay command on %s", topic)
                            continue
                        await manager.broadcast({
                            "type": "RELAY_COMMAND",
                            "device_id": device_id,
                            "command": payload.split("|", 1)[0],
                            "command_id": int(payload.split("|", 1)[1]),
                        })

        except aiomqtt.MqttError as e:
            _shared_mqtt_client = None
            _mqtt_listener_alive = False
            _mqtt_subscriptions_ready = False
            # Issue #6: Exponential backoff with jitter
            attempt += 1
            backoff = min(60, 2 ** attempt) + random.uniform(0, 1)
            logger.error(f"MQTT bridge connection failed: {e}. Retrying in {backoff:.1f}s (attempt {attempt})...")
            system_state["pipeline_status"] = "mqtt_reconnecting"
            await asyncio.sleep(backoff)
        except asyncio.CancelledError:
            _shared_mqtt_client = None
            _mqtt_listener_alive = False
            _mqtt_subscriptions_ready = False
            break
        except Exception as e:
            # A malformed frame, iterator termination, or unexpected callback
            # must never leave stale connected/readiness state behind.
            _shared_mqtt_client = None
            _mqtt_listener_alive = False
            _mqtt_subscriptions_ready = False
            system_state["pipeline_status"] = "mqtt_reconnecting"
            attempt += 1
            backoff = min(60, 2 ** attempt) + random.uniform(0, 1)
            logger.exception(
                "MQTT bridge task failed; reconnecting in %.1fs: %s", backoff, e
            )
            await asyncio.sleep(backoff)


# ─── Heartbeat ───────────────────────────────────────────────────────
async def heartbeat_task():
    """Send periodic heartbeat to keep WebSocket connections alive.
    Issue #23: Heartbeat only sends pipeline_status, NOT full init state.
    Full init state is sent once on WebSocket connect only."""
    while True:
        await asyncio.sleep(5)
        if manager.active_connections:
            await manager.broadcast({"type": "heartbeat", "status": system_state["pipeline_status"]})


# ─── Fix §4.3.3: WebSocket Telemetry Aggregation Task ────────────────
async def ws_aggregation_task():
    """Flush the power aggregation buffer as a single batched broadcast
    every 1 second. This replaces per-device per-message broadcasting
    and reduces WebSocket traffic by ~100x for multi-device deployments.

    At 100 devices × 1Hz, this sends 1 batched message/sec instead of
    100 individual messages/sec, preventing browser JS thread saturation."""
    while True:
        await asyncio.sleep(1.0)
        if not manager.active_connections:
            # Do not send stale readings to a client that connects after an
            # outage; a future live sample will repopulate the batch.
            _ws_power_buffer.clear()
            continue
        if not _ws_power_buffer:
            continue
        # Atomically snapshot and clear the buffer
        snapshot = dict(_ws_power_buffer)
        _ws_power_buffer.clear()
        # Broadcast single aggregated message
        await manager.broadcast({
            "type": "power_batch",
            "readings": snapshot,
            "device_count": len(snapshot),
            "timestamp": time.time(),
        })


# ─── App Lifecycle ───────────────────────────────────────────────────
# Issue #10 & #22: Properly await cancelled tasks; catch and log errors
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _shared_db, _shared_mqtt_client
    db_path = os.path.join(os.getcwd(), "data", "ems_state.db")
    try:
        if os.path.exists(db_path):
            _shared_db = await aiosqlite.connect(db_path)
            _shared_db.row_factory = aiosqlite.Row
            logger.info("Shared DB connection established.")
    except Exception as e:
        logger.error(f"Failed to open shared DB connection: {e}")

    mqtt_task = asyncio.create_task(mqtt_listener_task())
    hb_task = asyncio.create_task(heartbeat_task())
    # Fix §4.3.3: WebSocket aggregation task
    agg_task = asyncio.create_task(ws_aggregation_task())
    try:
        yield
    finally:
        for task, name in [(mqtt_task, "mqtt_listener"), (hb_task, "heartbeat"),
                           (agg_task, "ws_aggregation")]:
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.CancelledError:
                logger.info(f"{name} task cancelled cleanly.")
            except asyncio.TimeoutError:
                logger.warning(f"{name} task did not finish within 5s timeout.")
            except Exception as e:
                logger.error(f"{name} task shutdown error: {e}")
        if _shared_db:
            try:
                await _shared_db.close()
                logger.info("Shared DB connection closed.")
            except Exception as e:
                logger.error(f"Error closing DB connection: {e}")
            finally:
                _shared_db = None
        _shared_mqtt_client = None
        _ws_power_buffer.clear()
        _device_event_watermarks.clear()
        async with manager._lock:
            manager.active_connections.clear()


app = FastAPI(title="Digital Twin EMS", lifespan=lifespan)
app.state.broadcast = manager.broadcast

# Issue #8: Restrict CORS to specific trusted origins. 127.0.0.1 variants
# included because browsers treat localhost and 127.0.0.1 as distinct
# origins (and the WS origin check reuses this list) — LAN-IP origins must
# be added via CORS_ORIGINS when the dashboard is viewed from another
# device (see claude_debug/BRINGUP_RUNBOOK.md, demo-day notes).
allowed_origins = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,"
    "http://127.0.0.1:3000,http://127.0.0.1:5173"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Issue #15: Rate Limiting (graceful degradation if slowapi not installed)
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
except ImportError:
    limiter = None
    logger.warning("slowapi not installed — rate limiting disabled.")


# ─── REST Endpoints ─────────────────────────────────────────────────
class StatusResponse(BaseModel):
    status: str
    message: str


@app.get("/health", response_model=StatusResponse)
async def health_check() -> StatusResponse:
    return StatusResponse(status="ok", message="EMS API is running")


# Issue #14: Readiness probe that checks dependencies
@app.get("/ready")
async def readiness_check():
    """Readiness probe: checks MQTT and DB connectivity."""
    checks = {
        "mqtt": "connected" if (
            _shared_mqtt_client is not None
            and _mqtt_listener_alive
            and _mqtt_subscriptions_ready
        ) else "disconnected",
        "database": "available" if _shared_db is not None else "unavailable",
    }
    all_ok = all(v in ("connected", "available") for v in checks.values())
    if not all_ok:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": checks})
    return {"status": "ready", "checks": checks}


@app.get("/api/devices")
async def get_devices():
    """Current power state of all monitored devices."""
    async with state_lock:
        return {"devices": {k: dict(v) for k, v in system_state["devices"].items()}}


@app.get("/api/analytics")
async def get_analytics():
    """Daily usage summary and estimated cost."""
    return {"analytics": system_state["analytics"]}


@app.get("/api/phantom")
async def get_phantom():
    """Phantom (vampire) load report."""
    return {
        "phantom_loads": system_state["phantom_loads"],
        # Issue #20: Consistent field name
        "total_phantom_watts": system_state["total_phantom"],
    }


@app.get("/api/status")
async def get_status():
    """Full system status snapshot."""
    return {
        "pipeline_status": system_state["pipeline_status"],
        "device_count": len(system_state["devices"]),
        "pmv_score": system_state["pmv_score"],
        "total_phantom_watts": system_state["total_phantom"],
        "active_ws_clients": len(manager.active_connections),
    }


@app.get("/api/pending-labels")
async def get_pending_labels():
    """Devices awaiting user classification label."""
    async with state_lock:
        return {"pending_labels": [dict(v) for v in system_state["pending_labels"]]}


@app.get("/api/low-confidence")
async def get_low_confidence():
    """Recent low-confidence classification events."""
    return {"low_confidence_log": system_state["low_confidence_log"]}


@app.get("/api/safety-warnings")
async def get_safety_warnings():
    """Recent safety warning and cutoff events."""
    return {"safety_warnings": system_state["safety_warnings"]}


# Issue #9: Input validation with size constraints
# Bug 3.5 fix: Accept segments in LabelSubmission for ProtoNet registry update
class LabelSubmission(BaseModel):
    device_id: str = Field(..., max_length=64)
    # Log-forging guard: the label ends up in logs, MQTT payloads and audit
    # entries — no newlines, control chars or path-ish content.
    label: str = Field(
        ..., max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9 _-]*$"
    )
    segments: List[List[float]] = Field(
        default=[],
        description=(
            "Raw 128-sample power windows in watts, taken from the "
            "LABEL_REQUEST event's `segments` field. NOT embeddings — the "
            "orchestrator runs the CNN over these, so an embedding here "
            "enrolls a meaningless prototype."
        ),
    )

    @field_validator("segments")
    @classmethod
    def validate_segments(cls, v):
        if len(v) > 100:
            raise ValueError("Maximum 100 segments allowed")
        for i, seg in enumerate(v):
            if len(seg) != 128:
                raise ValueError(
                    f"Segment {i} must have exactly 128 dimensions, got {len(seg)}"
                )
            # Power, not an embedding. Both are length-128 float lists, so the
            # length check alone let the dashboard submit embeddings for months.
            # Watts are non-negative and an appliance event clears the 20 W
            # on-threshold; embeddings are zero-centred and fail both.
            if not all(math.isfinite(x) for x in seg):
                raise ValueError(f"Segment {i} contains NaN/Inf")
            if min(seg) < -1e-3:
                raise ValueError(
                    f"Segment {i} has negative values ({min(seg):.3f} W): expected "
                    "raw power in watts, not an embedding"
                )
            if max(seg) < 20.0:
                raise ValueError(
                    f"Segment {i} never exceeds 20 W (max {max(seg):.3f}): expected "
                    "a real appliance power window, not an embedding"
                )
        return v


# Issue #7: Require API key for write endpoints
@app.post("/api/submit-label")
async def submit_label(submission: LabelSubmission, _: None = Depends(verify_api_key)):
    """Submit a user-provided label for an unknown device."""
    # Issue #2: Lock around compound state mutation
    async with state_lock:
        system_state["pending_labels"] = [
            p for p in system_state["pending_labels"]
            if p["device_id"] != submission.device_id
        ]
    logger.info(f"Label submitted: {submission.device_id} → {submission.label}")
    # Issue #18: Audit log
    audit_logger.info(json.dumps({
        "event": "LABEL_SUBMITTED",
        "device_id": submission.device_id,
        "label": submission.label,
        "segments_count": len(submission.segments),
    }))

    # Bug 3.1 fix: Publish to MQTT so the orchestrator can update ProtoNet
    # Issue #13: Reuse shared MQTT client when available
    try:
        mqtt_payload = json.dumps({
            "class_name": submission.label,
            "segments": submission.segments,
        })
        if _shared_mqtt_client:
            await _shared_mqtt_client.publish("home/ml/label", mqtt_payload)
        else:
            async with aiomqtt.Client(**_mqtt_connect_kwargs()) as client:
                await client.publish("home/ml/label", mqtt_payload)
    except Exception as e:
        logger.error(f"Failed to publish label to MQTT: {e}")
        # Never report success when the orchestrator never received the label:
        # the dashboard would show "successfully enrolled" for a no-op.
        raise HTTPException(
            status_code=503,
            detail="MQTT publish failed — label NOT enrolled, retry",
        )

    await manager.broadcast({
        "type": "LABEL_SUBMITTED",
        "device_id": submission.device_id,
        "label": submission.label,
    })
    return {"status": "ok", "message": f"Label '{submission.label}' applied to {submission.device_id}"}


# Bug 3.4 fix: Removed /api/unknown_devices and pending_unknowns_store.
# The never-populated pending_unknowns_store was always returning empty arrays.
# Use /api/pending-labels exclusively for unknown device management.


# ─── Label-Unrecognized (open-set labeling hook) ────────────────────
# Signature buffer for unrecognised electrical signatures. Populated from
# LABEL_REQUEST events (keyed by signature_id == device_id of the unknown
# node), so a label can be submitted later without holding the original
# WebSocket frame. Raw 128-sample POWER windows in watts — the format
# PrototypeRegistry.add_class / handle_label_submitted enroll from.
signature_buffer: Dict[str, Dict[str, Any]] = {}
_signature_lock = asyncio.Lock()
SIGNATURE_BUFFER_MAX = 200  # bound memory; keep most recent

# Classes recognised by this deployment (the profile's top-level
# `appliances:` list; the old `protonet.classes` key was deleted 2026-09-10).
_target_classes: List[str] = []
try:
    import yaml as _yaml
    _cfg_path = os.environ.get("EMS_CONFIG", "config/config.demo.yaml")
    if not os.path.exists(_cfg_path):
        _cfg_path = "config/config.yaml"
    with open(_cfg_path) as _fh:
        _cfg = _yaml.safe_load(_fh) or {}
    _target_classes = list(_cfg.get("appliances") or [])
except Exception:
    _target_classes = []


async def _capture_signature(device_id: str, entry: Dict[str, Any]) -> None:
    """Buffer the feature vector behind an unrecognised signature (open-set hook).

    Called from the LABEL_REQUEST branch of the MQTT listener: keeps the raw
    power windows so a later POST /api/v1/appliances/label-unrecognized can
    enroll them, plus the embedding for display/cluster identity.
    """
    async with _signature_lock:
        signature_buffer[device_id] = {
            "segments": list(entry.get("segments") or []),
            "embedding": list(entry.get("embedding") or []),
            "power": entry.get("power", 0.0),
            "confidence": entry.get("confidence", 0.0),
            "captured_at": time.time(),
        }
        while len(signature_buffer) > SIGNATURE_BUFFER_MAX:
            signature_buffer.pop(next(iter(signature_buffer)))


class LabelUnrecognizedRequest(BaseModel):
    """POST /api/v1/appliances/label-unrecognized body."""
    signature_id: str = Field(..., max_length=64)
    # Same charset contract as LabelSubmission.label (log-forging guard).
    label: str = Field(
        ..., max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9 _-]*$"
    )
    # Optional direct segments (K x 128 raw watts). When absent the buffered
    # windows captured at LABEL_REQUEST time are used.
    segments: List[List[float]] = Field(
        default=[],
        description=(
            "Raw 128-sample power windows in watts. When omitted, the buffered "
            "windows for signature_id (captured when the signature was flagged "
            "Unrecognized) are enrolled."
        ),
    )

    @field_validator("segments")
    @classmethod
    def validate_segments(cls, v):
        if len(v) > 100:
            raise ValueError("Maximum 100 segments allowed")
        for i, seg in enumerate(v):
            if len(seg) != 128:
                raise ValueError(f"Segment {i} must have exactly 128 dimensions, got {len(seg)}")
            if not all(math.isfinite(x) for x in seg):
                raise ValueError(f"Segment {i} contains NaN/Inf")
            if min(seg) < -1e-3:
                raise ValueError(f"Segment {i} has negative values: expected raw power in watts")
            if max(seg) < 20.0:
                raise ValueError(f"Segment {i} never exceeds 20 W: expected an appliance power window")
        return v


class LabelUnrecognizedResponse(BaseModel):
    status: str
    signature_id: str
    label: str
    enrolled: bool
    classes: List[str]
    message: str


@app.post(
    "/api/v1/appliances/label-unrecognized",
    response_model=LabelUnrecognizedResponse,
)
async def label_unrecognized(req: LabelUnrecognizedRequest,
                             _: None = Depends(verify_api_key)):
    """
    Label an Unrecognised/Unknown electrical signature and register it into
    the runtime prototype registry via few-shot prototype averaging.

    Body: {"signature_id": "<id>", "label": "<new_appliance_name>"}.
    The 128-sample power windows buffered when the signature was flagged
    Unrecognized are enrolled (PrototypeRegistry.add_class), and the registry
    is persisted to the configured registry_path. The enrolled class is live
    for the *next* classification event.
    """
    label = req.label.strip()
    if not label:
        raise HTTPException(status_code=400, detail="label must not be empty")

    async with _signature_lock:
        buffered = signature_buffer.get(req.signature_id) or {}

    segments = req.segments or buffered.get("segments") or []
    source = "request" if req.segments else "buffer"

    if not segments:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "no_segments",
                "message": (
                    f"No buffered feature windows for signature_id "
                    f"'{req.signature_id}'. Provide `segments` (K x 128 raw "
                    f"watts) or wait for a LABEL_REQUEST event to buffer them."
                ),
                "signature_id": req.signature_id,
            },
        )

    # Enroll into the runtime prototype registry. Same watts-not-embedding
    # validation as handle_label_submitted, which does the actual
    # few-shot prototype averaging (add_class) and persistence.
    # Construct a FullPipeline lazily so the heavy weights load only once
    # and are shared across requests.
    global _label_pipeline
    try:
        if _label_pipeline is None:
            _label_pipeline = await asyncio.to_thread(_build_label_pipeline)
        pipeline = _label_pipeline
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ML pipeline unavailable: {e}")

    enrolled = await asyncio.to_thread(
        _enroll_segments, pipeline, label, segments
    )

    classes_after = await asyncio.to_thread(
        lambda: list(pipeline.prototype_registry.class_names())
    ) if pipeline.prototype_registry is not None else []

    # Consume the buffered signature on success
    if enrolled:
        async with _signature_lock:
            signature_buffer.pop(req.signature_id, None)
        await manager.broadcast({
            "type": "LABEL_SUBMITTED",
            "device_id": req.signature_id,
            "label": label,
            "signature_id": req.signature_id,
        })
        audit_logger.info(json.dumps({
            "event": "LABEL_UNRECOGNIZED",
            "signature_id": req.signature_id,
            "label": label,
            "segments_count": len(segments),
            "source": source,
        }))

    return LabelUnrecognizedResponse(
        status="ok" if enrolled else "error",
        signature_id=req.signature_id,
        label=label,
        enrolled=enrolled,
        classes=classes_after,
        message=(
            f"Registered '{label}' into the runtime prototype registry"
            if enrolled else
            f"Failed to enroll '{label}' — segments rejected (see server log)"
        ),
    )


# Shared lazily-initialised orchestrator for the label-unrecognized endpoint.
_label_pipeline = None


def _build_label_pipeline():
    """Load a FullPipeline from the active config for enrollment."""
    import yaml
    cfg_path = os.environ.get("EMS_CONFIG", "config/config.demo.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = "config/config.yaml"
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    from scripts.run_pipeline import FullPipeline
    return FullPipeline(config=cfg)


def _enroll_segments(pipeline, label: str, segments: List[List[float]]) -> bool:
    """Run handle_label_submitted in a worker thread. True on enrollment."""
    before = set(pipeline.prototype_registry.class_names()) \
        if pipeline.prototype_registry is not None else set()
    try:
        pipeline.handle_label_submitted(label, segments)
    except Exception as e:
        logger.error(f"label-unrecognized enrollment failed: {e}")
        return False
    after = set(pipeline.prototype_registry.class_names()) \
        if pipeline.prototype_registry is not None else set()
    return label in after or (before != after)


@app.get("/api/v1/appliances/signatures/unrecognized")
async def list_unrecognized_signatures(_: None = Depends(verify_api_key)):
    """Buffered unrecognised signatures awaiting a label."""
    async with _signature_lock:
        out = [
            {
                "signature_id": sid,
                "power": v.get("power", 0.0),
                "confidence": v.get("confidence", 0.0),
                "captured_at": v.get("captured_at"),
                "window_count": len(v.get("segments") or []),
            }
            for sid, v in signature_buffer.items()
        ]
    return {"unrecognized_signatures": out, "target_classes": _target_classes}


@app.get("/api/export-csv")
async def export_csv():
    """Stream the last 24 hours of power measurements as a downloadable CSV."""
    db_path = os.path.join(os.getcwd(), "data", "ems_state.db")
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database file not found")

    cutoff = time.time() - 86400  # 24 hours ago

    async def _generate():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["timestamp", "datetime", "device_id", "power_watts"])
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate(0)

        # Issue #1: Use fetchmany() with batching instead of async iteration
        try:
            async with aiosqlite.connect(db_path) as conn:
                conn.row_factory = aiosqlite.Row
                async with conn.execute(
                    "SELECT timestamp, device_id, power FROM measurements "
                    "WHERE timestamp >= ? ORDER BY timestamp ASC, id ASC",
                    (cutoff,),
                ) as cursor:
                    while True:
                        rows = await cursor.fetchmany(500)
                        if not rows:
                            break
                        for row in rows:
                            ts = row["timestamp"]
                            dt_str = time.strftime(
                                "%Y-%m-%d %H:%M:%S", time.localtime(ts)
                            )
                            writer.writerow(
                                [f"{ts:.3f}", dt_str, row["device_id"], f"{row['power']:.2f}"]
                            )
                        yield buf.getvalue()
                        buf.seek(0)
                        buf.truncate(0)
        except Exception as e:
            logger.error(f"CSV export error: {e}")
            writer.writerow(["ERROR", str(e), "", ""])
            yield buf.getvalue()

    return StreamingResponse(
        _generate(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ems_24h_export.csv"},
    )


# ─── WebSocket Endpoint ─────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # CORS middleware only guards HTTP, not the WebSocket handshake. Browsers
    # always send Origin on a WS handshake; non-browser clients (our own
    # tooling) usually omit it and are allowed.
    origin = websocket.headers.get("origin")
    if origin is not None and origin not in allowed_origins:
        logger.warning(f"Rejected WebSocket handshake: disallowed Origin {origin!r}")
        await websocket.close(code=1008)
        return
    connected = await manager.connect(websocket)
    if not connected:
        return
    # Issue #23: Send initial state snapshot ONCE on connection only
    try:
        await manager.send(websocket, {
            "type": "init_state",
            "devices": system_state["devices"],
            "pmv_score": system_state["pmv_score"],
            "phantom_loads": system_state["phantom_loads"],
            "pipeline_status": system_state["pipeline_status"],
        })
    except Exception as e:
        # Issue #11: Log instead of silently swallowing
        logger.debug(f"Failed to send init state to WebSocket: {e}")

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except RuntimeError as e:
        logger.warning(f"WebSocket disconnected with error: {e}")
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Unexpected WebSocket error: {e}")
        await manager.disconnect(websocket)
