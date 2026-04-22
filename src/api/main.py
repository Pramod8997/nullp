from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import asyncio
import aiomqtt
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Digital Twin EMS")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info("WebSocket disconnected.")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to websocket: {e}")

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    # Start the background task to listen to MQTT and broadcast
    asyncio.create_task(mqtt_listener_task())

async def mqtt_listener_task():
    # Connect to the local MQTT broker and forward messages to WebSockets
    while True:
        try:
            async with aiomqtt.Client("localhost", port=1883) as client:
                logger.info("FastAPI connected to MQTT Broker for WebSocket broadcasting.")
                
                # Listen for power readings, cutoff commands, and UI events
                await client.subscribe("home/sensor/+/power")
                await client.subscribe("home/plug/+/command")
                await client.subscribe("home/ui/events")
                
                async for message in client.messages:
                    topic = str(message.topic)
                    payload = message.payload.decode() if isinstance(message.payload, bytes) else message.payload
                    
                    if "home/ui/events" in topic:
                        try:
                            event_data = json.loads(payload)
                            await manager.broadcast(event_data)
                        except json.JSONDecodeError:
                            pass
                            
                    elif "/power" in topic:
                        device_id = topic.split("/")[-2]
                        try:
                            power_watts = float(payload)
                            await manager.broadcast({
                                "type": "power_reading",
                                "device_id": device_id,
                                "power": power_watts
                            })
                        except ValueError:
                            pass
                    
                    elif "/command" in topic and payload == "OFF":
                        device_id = topic.split("/")[-2]
                        await manager.broadcast({
                            "type": "safety_alert",
                            "message": f"Safety Cutoff Triggered for {device_id}!"
                        })
                        
        except aiomqtt.MqttError as e:
            logger.error(f"FastAPI MQTT connection failed: {e}. Retrying in 5s...")
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            break

class StatusResponse(BaseModel):
    status: str
    message: str

@app.get("/health", response_model=StatusResponse)
async def health_check() -> StatusResponse:
    return StatusResponse(status="ok", message="EMS API is running")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, we only push from the server
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
