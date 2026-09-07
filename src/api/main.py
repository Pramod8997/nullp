import json
import asyncio
import logging
import logging.handlers
import math
import time
import os
import io
import csv
import random
from typing import List, Dict, Any, Optional, TypedDict
from contextlib import asynccontextmanager

import aiosqlite
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    HTTPException, Depends, Header,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
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
    segments: List[List[float]] = []
    embedding: List[float] = []
    suggested_label: str = ""
    message: str = ""


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

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Issue #3 fix: Snapshot list under lock, concurrent send via gather.
        Issue #16: Disconnects slow/dead clients automatically."""
        async with self._lock:
            snapshot = list(self.active_connections)

        if not snapshot:
            return

        disconnected: List[WebSocket] = []

        async def _send(ws: WebSocket):
            try:
                await asyncio.wait_for(ws.send_json(message), timeout=0.5)
            except (asyncio.TimeoutError, Exception):
                disconnected.append(ws)

        await asyncio.gather(*[_send(ws) for ws in snapshot], return_exceptions=True)

        for ws in disconnected:
            await self.disconnect(ws)


manager = ConnectionManager()

# ─── Issue #13: Global MQTT client reference ────────────────────────
_shared_mqtt_client: Optional[aiomqtt.Client] = None
_shared_db: Optional[aiosqlite.Connection] = None


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
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ─── MQTT → WebSocket Bridge ────────────────────────────────────────
async def mqtt_listener_task():
    """
    Subscribes to all EMS MQTT topics and forwards messages to
    connected WebSocket clients AND updates shared system_state.
    """
    global _shared_mqtt_client
    attempt = 0  # Issue #6: Exponential backoff counter

    while True:
        try:
            # Bug 4.5 fix: Use MQTT_BROKER env var instead of hardcoded localhost
            async with aiomqtt.Client(**_mqtt_connect_kwargs()) as client:
                _shared_mqtt_client = client
                logger.info("FastAPI MQTT bridge connected.")
                system_state["pipeline_status"] = "connected"
                attempt = 0  # Reset backoff on successful connection

                await client.subscribe("home/sensor/+/power")
                await client.subscribe("home/plug/+/command")
                await client.subscribe("home/ui/events")

                async for message in client.messages:
                    topic = str(message.topic)
                    payload = message.payload.decode() if isinstance(message.payload, bytes) else str(message.payload)

                    # ── UI Events (structured JSON from the pipeline) ──
                    if "home/ui/events" in topic:
                        try:
                            event_data = json.loads(payload)
                            event_type = event_data.get("type", "")

                            if event_type == "DEVICE_STATUS":
                                evt = DeviceStatusEvent(**event_data)
                                did = evt.device_id
                                # Issue #5: Standardize timestamp to epoch float
                                ts = evt.timestamp if evt.timestamp else time.time()
                                try:
                                    ts = float(ts)
                                except (ValueError, TypeError):
                                    ts = time.time()
                                system_state["devices"][did] = {
                                    "power": evt.power,
                                    "state": evt.state,
                                    "classification": evt.classification,
                                    "confidence": evt.confidence,
                                    "pmv": evt.pmv,
                                    "last_seen": ts,
                                }
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
                        except json.JSONDecodeError:
                            # Issue #11: Log instead of silently swallowing
                            logger.debug(f"Invalid JSON on topic {topic}, payload length={len(payload)}")

                    # ── Raw Power Readings ──
                    elif "/power" in topic:
                        device_id = topic.split("/")[-2]
                        try:
                            power_watts = float(payload)
                            if device_id not in system_state["devices"]:
                                system_state["devices"][device_id] = {}
                            system_state["devices"][device_id]["power"] = power_watts

                            # Fix §4.3.3: Accumulate into aggregation buffer
                            # instead of broadcasting each reading individually.
                            # The ws_aggregation_task flushes this buffer every 1s.
                            _ws_power_buffer[device_id] = power_watts
                        except ValueError:
                            # Issue #11: Log instead of silently swallowing
                            logger.debug(f"Non-numeric power payload on {topic}: {payload[:50]}")

                    # ── Relay Commands ──
                    elif "/command" in topic:
                        device_id = topic.split("/")[-2]
                        await manager.broadcast({
                            "type": "safety_alert",
                            "severity": "critical",
                            "device_id": device_id,
                            "message": f"Relay Cutoff: {device_id} forced {payload}",
                            "command": payload,
                        })

        except aiomqtt.MqttError as e:
            _shared_mqtt_client = None
            # Issue #6: Exponential backoff with jitter
            attempt += 1
            backoff = min(60, 2 ** attempt) + random.uniform(0, 1)
            logger.error(f"MQTT bridge connection failed: {e}. Retrying in {backoff:.1f}s (attempt {attempt})...")
            system_state["pipeline_status"] = "mqtt_reconnecting"
            await asyncio.sleep(backoff)
        except asyncio.CancelledError:
            _shared_mqtt_client = None
            break


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
        if not manager.active_connections or not _ws_power_buffer:
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
    global _shared_db
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


app = FastAPI(title="Digital Twin EMS", lifespan=lifespan)
app.state.broadcast = manager.broadcast

# Issue #8: Restrict CORS to specific trusted origins
allowed_origins = os.environ.get(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
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
        "mqtt": "connected" if _shared_mqtt_client is not None else "disconnected",
        "database": "available" if _shared_db is not None else "unavailable",
    }
    all_ok = all(v in ("connected", "available") for v in checks.values())
    if not all_ok:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": checks})
    return {"status": "ready", "checks": checks}


@app.get("/api/devices")
async def get_devices():
    """Current power state of all monitored devices."""
    return {"devices": system_state["devices"]}


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
    return {"pending_labels": system_state["pending_labels"]}


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
    label: str = Field(..., max_length=64)
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

# Classes recognised by this deployment (config protonet.classes).
_target_classes: List[str] = []
try:
    import yaml as _yaml
    _cfg_path = os.environ.get("EMS_CONFIG", "config/config.demo.yaml")
    if not os.path.exists(_cfg_path):
        _cfg_path = "config/config.yaml"
    with open(_cfg_path) as _fh:
        _cfg = _yaml.safe_load(_fh) or {}
    _target_classes = list((_cfg.get("protonet") or {}).get("classes") or [])
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
    label: str = Field(..., max_length=64)
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
                    "WHERE timestamp >= ? ORDER BY timestamp ASC",
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
    await manager.connect(websocket)
    # Issue #23: Send initial state snapshot ONCE on connection only
    try:
        await websocket.send_json({
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
