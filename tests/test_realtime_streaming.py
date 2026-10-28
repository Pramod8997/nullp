import asyncio

import pytest

from scripts.run_pipeline import FullPipeline, load_config
from src.api.main import ConnectionManager, MAX_WS_QUEUE_SIZE
from src.database.session import DatabaseSession
from src.hardware.mqtt import MQTTClientManager


class _FakeWebSocket:
    def __init__(self, delay=0.0, fail=False):
        self.delay = delay
        self.fail = fail
        self.sent = []
        self.closed = False

    async def accept(self):
        return None

    async def send_json(self, message):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("socket closed")
        self.sent.append(message)

    async def close(self, **_kwargs):
        self.closed = True


@pytest.mark.asyncio
async def test_websocket_broadcast_is_bounded_for_slow_client():
    manager = ConnectionManager()
    websocket = _FakeWebSocket(delay=1.0)
    assert await manager.connect(websocket)

    for index in range(MAX_WS_QUEUE_SIZE * 4):
        await manager.broadcast({"type": "power_batch", "index": index})

    queue = manager._queues[websocket]
    assert queue.maxsize == MAX_WS_QUEUE_SIZE
    assert queue.qsize() <= MAX_WS_QUEUE_SIZE

    # A stalled socket is removed by the sender timeout and cannot retain a
    # sender task or queue after disconnect.
    await asyncio.sleep(0.6)
    assert manager.connection_count() == 0
    assert websocket not in manager._queues


@pytest.mark.asyncio
async def test_mqtt_ingress_queue_is_bounded_and_bad_callback_isolated():
    manager = MQTTClientManager("localhost", ingress_queue_size=2)
    seen = []

    async def callback(topic, _payload):
        seen.append(topic)
        if topic.endswith("/bad"):
            raise RuntimeError("malformed frame")

    manager.set_read_callback(callback)
    manager._enqueue_message("home/sensor/a/power", "1")
    manager._enqueue_message("home/sensor/b/power", "2")
    manager._enqueue_message("home/sensor/c/power", "3")
    assert manager.ingress_queue_size <= 2
    assert manager.dropped_messages == 1

    manager._enqueue_message("home/sensor/bad/power", "4")
    dispatcher = asyncio.create_task(manager._dispatch_ingress())
    await asyncio.sleep(0.05)
    dispatcher.cancel()
    await asyncio.gather(dispatcher, return_exceptions=True)
    assert manager.ingress_queue_size == 0
    assert seen


@pytest.mark.asyncio
async def test_database_measurement_queue_has_hard_capacity(tmp_path):
    session = DatabaseSession(
        str(tmp_path / "measurements.db"),
        fallback_csv=str(tmp_path / "fallback.csv"),
        write_queue_size=3,
    )
    session._running = True
    for index in range(20):
        await session.insert_measurement(float(index), "node", float(index))

    assert session.max_queued_writes == 3
    assert session.queued_writes <= 3
    await session.close()


@pytest.mark.asyncio
async def test_telemetry_topic_flood_evicts_pipeline_state():
    config = load_config()
    config.setdefault("system", {})["max_tracked_devices"] = 3
    pipeline = FullPipeline(config=config)

    async def no_broadcast(_event):
        return None

    pipeline._broadcast_event = no_broadcast

    for index in range(12):
        await pipeline._handle_mqtt_message(
            f"home/sensor/flood_{index}/telemetry",
            '{"v": 230, "i": 0.2, "w": 40, "pf": 0.9}',
        )

    assert len(pipeline._device_last_seen) <= 3
    assert len(pipeline.device_telemetry) <= 3
