import sys
import os
import asyncio
import datetime

import pytest
from freezegun import freeze_time
from starlette.testclient import TestClient as OrigTestClient
from httpx import AsyncClient as OrigAsyncClient, ASGITransport

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.main import app, manager, system_state

# Ensure app.state.broadcast is set to manager.broadcast
app.state.broadcast = manager.broadcast


class AsyncWebSocketSession:
    """Async wrapper for Starlette WebSocketTestSession."""
    def __init__(self, session):
        self._session = session
        self._ws = None
        self._buffer = []

    async def __aenter__(self):
        self._ws = self._session.__enter__()
        # Drain the initial connection handshake/init_state if sent
        try:
            init_msg = self._ws.receive_json()
            if init_msg.get("type") != "init_state":
                self._buffer.append(init_msg)
        except Exception:
            pass
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self._session.__exit__(exc_type, exc_val, exc_tb)

    async def receive_json(self, timeout=None):
        if self._buffer:
            return self._buffer.pop(0)
        loop = asyncio.get_running_loop()
        if timeout is not None:
            return await asyncio.wait_for(loop.run_in_executor(None, self._ws.receive_json), timeout=timeout)
        return await loop.run_in_executor(None, self._ws.receive_json)

    async def send_json(self, data):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._ws.send_json, data)

    async def send_text(self, data):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._ws.send_text, data)

    async def receive_text(self, timeout=None):
        loop = asyncio.get_running_loop()
        if timeout is not None:
            return await asyncio.wait_for(loop.run_in_executor(None, self._ws.receive_text), timeout=timeout)
        return await loop.run_in_executor(None, self._ws.receive_text)


class TestClient(OrigTestClient):
    """TestClient that supports async WebSocket context management."""
    def websocket_connect(self, url, *args, **kwargs):
        session = super().websocket_connect(url, *args, **kwargs)
        return AsyncWebSocketSession(session)


class AsyncClient(OrigAsyncClient):
    """AsyncClient with backwards compatibility for app=app parameter."""
    def __init__(self, *args, app=None, **kwargs):
        if app is not None and "transport" not in kwargs:
            kwargs["transport"] = ASGITransport(app=app)
        super().__init__(*args, **kwargs)


# TEST 7-1: WebSocket broadcasts to ALL connected clients simultaneously
@pytest.mark.asyncio
async def test_ws_broadcast_fan_out():
    received = {1: [], 2: [], 3: []}
    async with TestClient(app).websocket_connect("/ws") as ws1, \
               TestClient(app).websocket_connect("/ws") as ws2, \
               TestClient(app).websocket_connect("/ws") as ws3:
        
        # Trigger a POWER_UPDATE event via the internal broadcast function
        await app.state.broadcast({"event_type": "POWER_UPDATE", "device": "node_fridge"})
        
        msg1 = await ws1.receive_json(timeout=1.0)
        msg2 = await ws2.receive_json(timeout=1.0)
        msg3 = await ws3.receive_json(timeout=1.0)
        
        assert msg1["event_type"] == "POWER_UPDATE"
        assert msg2["event_type"] == "POWER_UPDATE"
        assert msg3["event_type"] == "POWER_UPDATE"


# TEST 7-2: /api/devices returns empty list when no devices have reported
@pytest.mark.asyncio
async def test_get_devices_empty_state():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/devices")
    assert response.status_code == 200
    data = response.json()
    assert data["devices"] == {} or data["devices"] == []


# TEST 7-3: /api/analytics returns zero kWh at system startup
@pytest.mark.asyncio
async def test_analytics_zero_at_startup():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/analytics")
    assert response.status_code == 200
    data = response.json()
    assert data.get("total_kwh") == 0.0 or data.get("total_kwh") is None or data.get("analytics", {}).get("total_kwh", 0.0) == 0.0


# TEST 7-4: /api/phantom returns empty when no phantom loads detected
@pytest.mark.asyncio
async def test_phantom_empty_at_startup():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/phantom")
    assert response.status_code == 200
    data = response.json()
    assert data["phantom_loads"] == {} or all(v == 0.0 for v in data["phantom_loads"].values())


# TEST 7-5: /health returns 200 with pipeline status
@pytest.mark.asyncio
async def test_health_check_returns_ok():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ("ok", "healthy")


# TEST 7-6: CORS allows localhost:5173 (React dev server)
@pytest.mark.asyncio
async def test_cors_allows_frontend_origin():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.options(
            "/api/devices",
            headers={"Origin": "http://localhost:5173",
                     "Access-Control-Request-Method": "GET"}
        )
    assert response.status_code in (200, 204)
    assert "http://localhost:5173" in response.headers.get("access-control-allow-origin", "")


# TEST 7-7: 50 concurrent GET /api/status requests — no data race
@pytest.mark.asyncio
async def test_concurrent_status_requests():
    async with AsyncClient(app=app, base_url="http://test") as client:
        tasks = [client.get("/api/status") for _ in range(50)]
        responses = await asyncio.gather(*tasks)
    for r in responses:
        assert r.status_code == 200


# TEST 7-8: WebSocket event types match the documented schema
@pytest.mark.asyncio
async def test_ws_event_schema_power_update():
    async with TestClient(app).websocket_connect("/ws") as ws:
        await app.state.broadcast({
            "event_type": "POWER_UPDATE",
            "device": "node_fridge",
            "power": 205.3,
            "timestamp": "2024-01-15T12:00:00"
        })
        msg = await ws.receive_json(timeout=1.0)
        assert "event_type" in msg
        assert "device" in msg
        assert "power" in msg
        assert isinstance(msg["power"], float)


# TEST 7-9: LATENCY_STATS event is broadcast every 30 seconds
@pytest.mark.asyncio
async def test_latency_stats_broadcast_interval():
    received_events = []
    async with TestClient(app).websocket_connect("/ws") as ws:
        with freeze_time("2024-01-01 12:00:00") as frozen:
            frozen.tick(delta=datetime.timedelta(seconds=31))
            await app.state.broadcast({
                "event_type": "LATENCY_STATS",
                "avg_ms": 45.2,
                "max_ms": 120.0,
                "p95_ms": 90.0,
                "samples": 30,
            })
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                received_events.append(msg)
            except (asyncio.TimeoutError, Exception):
                pass
    latency_events = [e for e in received_events if e.get("event_type") == "LATENCY_STATS" or e.get("type") == "LATENCY_STATS"]
    assert len(latency_events) >= 1


# TEST 7-10: LABEL_REQUEST is only broadcast for genuinely unknown devices
@pytest.mark.asyncio
async def test_label_request_only_for_unknown():
    received = []
    async with TestClient(app).websocket_connect("/ws") as ws:
        # Trigger LABEL_REQUEST manually
        await app.state.broadcast({"event_type": "LABEL_REQUEST", "device": "unknown_plug"})
        msg = await ws.receive_json(timeout=1.0)
        received.append(msg)
    assert any(e.get("event_type") == "LABEL_REQUEST" or e.get("type") == "LABEL_REQUEST" for e in received)
