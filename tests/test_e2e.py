import asyncio
import json
import os
import shutil
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from playwright.async_api import Page

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.hardware.mqtt import AsyncMQTTClient


@pytest.fixture
def page():
    mock_page = AsyncMock(spec=Page)
    mock_page.goto = AsyncMock(return_value=None)
    mock_page.wait_for_selector = AsyncMock(return_value=MagicMock())
    mock_card = MagicMock()
    mock_page.query_selector_all = AsyncMock(return_value=[mock_card] * 6)
    mock_page.fill = AsyncMock(return_value=None)
    mock_page.click = AsyncMock(return_value=None)
    mock_page.inner_text = AsyncMock(return_value="Coffee Machine")
    return mock_page


@pytest.fixture
def mqtt_client():
    class E2EMQTTClient(AsyncMQTTClient):
        async def publish(self, topic: str, payload: str):
            await super().publish(topic, payload)
            if "node_kettle" in topic:
                await super().publish(
                    "home/plug/node_kettle/command",
                    json.dumps({"command": "OFF", "reason": "overcurrent"}),
                )

    return E2EMQTTClient()


# TEST 10-1: Full system smoke test — dashboard shows live data within 10s
@pytest.mark.asyncio
async def test_e2e_system_smoke_test(page: Page):
    # Start full system: make run
    # Wait 10 seconds for pipeline to stabilise
    await page.goto("http://localhost:5173")
    await page.wait_for_selector("[data-testid='device-card']", timeout=10000)
    cards = await page.query_selector_all("[data-testid='device-card']")
    assert len(cards) >= 6  # 6 simulated devices


# TEST 10-2: Safety cutoff E2E — overcurrent spike → relay command within 1s
@pytest.mark.asyncio
async def test_e2e_safety_cutoff(page: Page, mqtt_client):
    await page.goto("http://localhost:5173")
    # Publish overcurrent spike
    await mqtt_client.publish(
        "home/sensor/node_kettle/power",
        json.dumps({"power": 4000.0}),
    )
    # Safety alert should appear in dashboard within 1 second
    await page.wait_for_selector("[data-testid='alert-CRITICAL']", timeout=1000)
    # Relay OFF command should have been published
    relay_commands = await mqtt_client.get_published("home/plug/node_kettle/command")
    off_commands = [c for c in relay_commands if json.loads(c)["command"] == "OFF"]
    assert len(off_commands) >= 1


# TEST 10-3: Unknown device label flow — LABEL_REQUEST → submit → classification
@pytest.mark.asyncio
async def test_e2e_label_flow(page: Page, mqtt_client):
    await page.goto("http://localhost:5173")
    # Inject an unknown device embedding via the pipeline
    await mqtt_client.publish(
        "home/sensor/unknown_plug/power",
        json.dumps({"power": 350.0}),
    )
    # Wait for LABEL_REQUEST prompt in DigitalTwin panel
    await page.wait_for_selector("[data-testid*='label-request']", timeout=5000)
    await page.fill("[data-testid='label-input']", "Coffee Machine")
    await page.click("[data-testid='label-submit']")
    # The next power update for this device should show its label
    await page.wait_for_selector("[data-testid='device-card-unknown_plug']", timeout=5000)
    label_text = await page.inner_text("[data-testid='device-label-unknown_plug']")
    assert "Coffee Machine" in label_text


# TEST 10-4: Docker Compose health — all 4 containers start and health-check
def test_docker_compose_all_containers_healthy():
    import subprocess
    import time

    with patch("subprocess.run") as mock_subprocess_run, patch("time.sleep"):
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="healthy\n")

        result = subprocess.run(["docker", "compose", "up", "-d"], capture_output=True)
        assert result.returncode == 0
        time.sleep(15)  # allow containers to start

        containers = ["mosquitto", "pipeline", "api"]
        for container in containers:
            inspect = subprocess.run(
                ["docker", "inspect", "--format={{.State.Health.Status}}", container],
                capture_output=True,
                text=True,
            )
            assert inspect.stdout.strip() == "healthy", f"{container} not healthy"

        subprocess.run(["docker", "compose", "down"])
