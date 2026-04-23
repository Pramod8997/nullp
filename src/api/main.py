import json
import asyncio
import logging
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiomqtt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Shared State ────────────────────────────────────────────────────
# Updated by the MQTT listener, read by REST endpoints and WebSocket broadcasts.
system_state: Dict[str, Any] = {
    "devices": {},          # device_id -> {power, state, classification, last_seen}
    "phantom_loads": {},    # device_id -> watts
    "total_phantom": 0.0,
    "pmv_score": 0.0,
    "analytics": {},        # daily summary
    "active_mitigations": [],
    "pipeline_status": "initializing",
}


# ─── WebSocket Manager ──────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        disconnected = []
        for ws in self.active_connections:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)


manager = ConnectionManager()


# ─── MQTT → WebSocket Bridge ────────────────────────────────────────
async def mqtt_listener_task():
    """
    Subscribes to all EMS MQTT topics and forwards messages to
    connected WebSocket clients AND updates shared system_state.
    """
    while True:
        try:
            async with aiomqtt.Client("localhost", port=1883) as client:
                logger.info("FastAPI MQTT bridge connected.")
                system_state["pipeline_status"] = "connected"

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

                            # Update shared state based on event type
                            if event_type == "DEVICE_STATUS":
                                did = event_data.get("device_id", "")
                                system_state["devices"][did] = {
                                    "power": event_data.get("power", 0),
                                    "state": event_data.get("state", "unknown"),
                                    "classification": event_data.get("classification", "unknown"),
                                    "pmv": event_data.get("pmv", 0),
                                    "last_seen": event_data.get("timestamp", ""),
                                }
                            elif event_type == "PHANTOM_LOAD":
                                system_state["phantom_loads"] = event_data.get("loads", {})
                                system_state["total_phantom"] = event_data.get("total", 0)
                            elif event_type == "ANALYTICS_UPDATE":
                                system_state["analytics"] = event_data.get("summary", {})
                            elif event_type == "PMV_UPDATE":
                                system_state["pmv_score"] = event_data.get("pmv", 0)

                            await manager.broadcast(event_data)
                        except json.JSONDecodeError:
                            pass

                    # ── Raw Power Readings ──
                    elif "/power" in topic:
                        device_id = topic.split("/")[-2]
                        try:
                            power_watts = float(payload)
                            # Update shared state
                            if device_id not in system_state["devices"]:
                                system_state["devices"][device_id] = {}
                            system_state["devices"][device_id]["power"] = power_watts

                            await manager.broadcast({
                                "type": "power_reading",
                                "device_id": device_id,
                                "power": power_watts,
                            })
                        except ValueError:
                            pass

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
            logger.error(f"MQTT bridge connection failed: {e}. Retrying in 3s...")
            system_state["pipeline_status"] = "mqtt_reconnecting"
            await asyncio.sleep(3)
        except asyncio.CancelledError:
            break


# ─── Heartbeat ───────────────────────────────────────────────────────
async def heartbeat_task():
    """Send periodic heartbeat to keep WebSocket connections alive."""
    while True:
        await asyncio.sleep(5)
        if manager.active_connections:
            await manager.broadcast({"type": "heartbeat", "status": system_state["pipeline_status"]})


# ─── App Lifecycle ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    mqtt_task = asyncio.create_task(mqtt_listener_task())
    hb_task = asyncio.create_task(heartbeat_task())
    yield
    mqtt_task.cancel()
    hb_task.cancel()


app = FastAPI(title="Digital Twin EMS", lifespan=lifespan)

# CORS — allow React dev server on any port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── REST Endpoints ─────────────────────────────────────────────────
class StatusResponse(BaseModel):
    status: str
    message: str


@app.get("/health", response_model=StatusResponse)
async def health_check() -> StatusResponse:
    return StatusResponse(status="ok", message="EMS API is running")


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
        "total_watts": system_state["total_phantom"],
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


# ─── WebSocket Endpoint ─────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    # Send initial state snapshot on connection
    try:
        await websocket.send_json({
            "type": "init_state",
            "devices": system_state["devices"],
            "pmv_score": system_state["pmv_score"],
            "phantom_loads": system_state["phantom_loads"],
            "pipeline_status": system_state["pipeline_status"],
        })
    except Exception:
        pass

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
