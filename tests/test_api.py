"""
API endpoint tests — Issue #19.
Run with: make test
"""
import sys
import os

import pytest
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.main import app, system_state, state_lock


# ─── Helpers ─────────────────────────────────────────────────────────
@pytest.fixture
def transport():
    return ASGITransport(app=app)


# ─── Health / Readiness ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_health_endpoint(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_ready_endpoint_returns_503_when_deps_unavailable(transport):
    """Issue #14: /ready should return 503 when MQTT/DB not connected."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ready")
        # With no MQTT or DB connected in test, expect 503
        assert response.status_code == 503


# ─── Device / Analytics / Status Endpoints ──────────────────────────
@pytest.mark.asyncio
async def test_devices_endpoint(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/devices")
        assert response.status_code == 200
        assert "devices" in response.json()


@pytest.mark.asyncio
async def test_analytics_endpoint(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/analytics")
        assert response.status_code == 200
        assert "analytics" in response.json()


@pytest.mark.asyncio
async def test_phantom_endpoint_uses_consistent_field_name(transport):
    """Issue #20: Field should be total_phantom_watts, not total_watts."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/phantom")
        assert response.status_code == 200
        data = response.json()
        assert "phantom_loads" in data
        assert "total_phantom_watts" in data
        assert "total_watts" not in data


@pytest.mark.asyncio
async def test_status_endpoint(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "pipeline_status" in data
        assert "device_count" in data
        assert "total_phantom_watts" in data


# ─── Pending Labels / Low-Confidence / Safety ───────────────────────
@pytest.mark.asyncio
async def test_pending_labels_endpoint(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/pending-labels")
        assert response.status_code == 200
        assert "pending_labels" in response.json()


@pytest.mark.asyncio
async def test_low_confidence_endpoint(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/low-confidence")
        assert response.status_code == 200
        assert "low_confidence_log" in response.json()


@pytest.mark.asyncio
async def test_safety_warnings_endpoint(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/safety-warnings")
        assert response.status_code == 200
        assert "safety_warnings" in response.json()


# ─── Submit Label — Auth (Issue #7) ─────────────────────────────────
@pytest.mark.asyncio
async def test_submit_label_without_api_key_when_key_is_set(transport):
    """Issue #7: Should reject if EMS_API_KEY is set but no header provided."""
    os.environ["EMS_API_KEY"] = "test-secret-key"
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/submit-label",
                json={"device_id": "test_device", "label": "fridge"},
            )
            assert response.status_code == 401
    finally:
        del os.environ["EMS_API_KEY"]


@pytest.mark.asyncio
async def test_submit_label_with_valid_api_key(transport):
    """Issue #7: Should accept with correct API key."""
    os.environ["EMS_API_KEY"] = "test-secret-key"
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/submit-label",
                json={"device_id": "test_device", "label": "fridge"},
                headers={"X-API-Key": "test-secret-key"},
            )
            # 200 OK (MQTT publish may fail in test, but endpoint logic succeeds)
            assert response.status_code == 200
    finally:
        del os.environ["EMS_API_KEY"]


@pytest.mark.asyncio
async def test_submit_label_no_auth_required_when_key_unset(transport):
    """When EMS_API_KEY is not set, any request should be accepted."""
    # Ensure key is not set
    os.environ.pop("EMS_API_KEY", None)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/submit-label",
            json={"device_id": "test_device", "label": "fridge"},
        )
        assert response.status_code == 200


# ─── Input Validation (Issue #9) ────────────────────────────────────
@pytest.mark.asyncio
async def test_submit_label_rejects_oversized_segments(transport):
    """Issue #9: Reject more than 100 segments."""
    os.environ.pop("EMS_API_KEY", None)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/submit-label",
            json={
                "device_id": "test",
                "label": "fridge",
                "segments": [[0.0] * 128] * 101,  # 101 > 100 limit
            },
        )
        assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_submit_label_rejects_wrong_dimension_segments(transport):
    """Issue #9: Reject segments with wrong dimension (not 128)."""
    os.environ.pop("EMS_API_KEY", None)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/submit-label",
            json={
                "device_id": "test",
                "label": "fridge",
                "segments": [[0.0] * 64],  # 64 != 128
            },
        )
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_submit_label_rejects_long_device_id(transport):
    """Issue #9: Reject device_id longer than 64 chars."""
    os.environ.pop("EMS_API_KEY", None)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/submit-label",
            json={
                "device_id": "x" * 65,
                "label": "fridge",
            },
        )
        assert response.status_code == 422


# ─── CSV Export (Issue #1) ──────────────────────────────────────────
@pytest.mark.asyncio
async def test_export_csv_returns_404_when_no_db(transport):
    """CSV export should return 404 when database file doesn't exist."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/export-csv")
        # If data/ems_state.db doesn't exist in test env, expect 404
        if not os.path.exists(os.path.join(os.getcwd(), "data", "ems_state.db")):
            assert response.status_code == 404
        else:
            assert response.status_code == 200
            assert "text/csv" in response.headers.get("content-type", "")
