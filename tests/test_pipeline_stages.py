import time
import datetime
import pytest
import numpy as np
import torch
from freezegun import freeze_time

from src.database.session import DBSession, load_config
from src.pipeline import (
    FleetDiagnosticsMonitor,
    Watchdog,
    NILMPreprocessor,
    ProtoNetClassifier,
    OpenMaxStage,
    TemperatureScalingStage,
    ConfidenceGateStage,
    DeltaStabilityStage,
    PhantomTrackerStage,
    AnalyticsStage,
    DigitalTwinStage,
    RLAgentStage,
    FullPipeline,
    BroadcastStage,
    mock_power_event,
    mock_power_event_with_window,
    mock_power_event_low_confidence,
    mock_thermal_event,
)

# TEST 5A-0: Fleet diagnostics (Stage 0) — parallel task does not block main pipeline
@pytest.mark.asyncio
async def test_stage0_fleet_diagnostics_nonblocking():
    fleet_monitor = FleetDiagnosticsMonitor(config=load_config())
    start = time.perf_counter()
    await fleet_monitor.process(mock_power_event("node_fridge", 200.0))
    elapsed = time.perf_counter() - start
    assert elapsed < 0.01  # must not block the pipeline

# TEST 5A-1: Watchdog (Stage 1) — returns anomaly event on z-score spike
@pytest.mark.asyncio
async def test_stage1_watchdog_anomaly_on_spike():
    watchdog = Watchdog(window=30, threshold=3.0)
    # Prime watchdog with 30 baseline readings
    for _ in range(30):
        await watchdog.process(mock_power_event("device", 100.0))
    # Send spike
    result = await watchdog.process(mock_power_event("device", 500.0))
    assert result.event_type in ("WATCHDOG_ANOMALY", "SOFT_OVERRIDE")

# TEST 5A-1b: NILM preprocessing (Stage 1b) — SG filter applied before derivative
@pytest.mark.asyncio
async def test_stage1b_nilm_sg_filter_applied():
    nilm = NILMPreprocessor(window_length=11, polyorder=2, threshold=50.0)
    noisy_signal = [100.0 + np.random.normal(0, 3) for _ in range(50)]
    noisy_signal += [300.0 + np.random.normal(0, 3) for _ in range(50)]
    events = await nilm.process_window(noisy_signal)
    # Should detect the 200W step despite noise
    assert len(events) == 1

# TEST 5A-2: ProtoNet CNN (Stage 2) — returns embedding of shape (128,)
@pytest.mark.asyncio
async def test_stage2_protonet_embedding_shape():
    protonet = ProtoNetClassifier(config=load_config())
    event = mock_power_event_with_window("node_fridge", window=[200.0]*128)
    result = await protonet.process(event)
    assert result.embedding.shape == (128,)

# TEST 5A-2b: OpenMax (Stage 2b) — unknown device does NOT produce class label
@pytest.mark.asyncio
async def test_stage2b_openmax_rejects_unknown():
    openmax = OpenMaxStage(config=load_config())
    unknown_embedding = np.random.randn(128) * 50.0  # far from all prototypes
    result = await openmax.process(embedding=unknown_embedding)
    assert result.label == "unknown" or result.rejected

# TEST 5A-2c: Temperature scaling (Stage 2c) — output confidence is well-calibrated
@pytest.mark.asyncio
async def test_stage2c_temp_scaling_calibrates_confidence():
    ts = TemperatureScalingStage(T=2.0)  # T>1 softens, reduces overconfidence
    logits = torch.tensor([[8.0, 0.5, 0.3, 0.1]])
    raw_conf = torch.softmax(logits, dim=1).max().item()
    calibrated = await ts.scale(logits)
    assert calibrated.confidence < raw_conf  # scaling softened the confidence

# TEST 5A-3: Confidence gate (Stage 3) — LOW_CONFIDENCE short-circuits RL stages
@pytest.mark.asyncio
async def test_stage3_confidence_gate_short_circuits():
    gate = ConfidenceGateStage(threshold=0.90)
    result = await gate.process(confidence=0.75, embedding=np.zeros(128))
    assert result.event_type == "LOW_CONFIDENCE"
    assert result.pipeline_action == "STOP"  # downstream stages should not run

# TEST 5A-4: Delta stability (Stage 4) — stable unknown emits LABEL_REQUEST
@pytest.mark.asyncio
async def test_stage4_delta_stability_emits_label_request():
    stage = DeltaStabilityStage(config=load_config())
    emb = np.ones(128, dtype=np.float32)
    for _ in range(3):
        result = await stage.process(emb + np.random.normal(0, 0.05, 128))
    assert result.event_type == "LABEL_REQUEST"

# TEST 5A-5: Phantom tracker (Stage 5) — EMA updated only when device is OFF
@pytest.mark.asyncio
async def test_stage5_phantom_updates_on_off_state():
    stage = PhantomTrackerStage(config=load_config())
    await stage.process(mock_power_event("esp32_tv", 8.0, state="OFF"))
    ema = stage.get_ema("esp32_tv")
    assert ema > 0.0

@pytest.mark.asyncio
async def test_stage5_phantom_does_not_update_on_on_state():
    stage = PhantomTrackerStage(config=load_config())
    await stage.process(mock_power_event("esp32_tv", 150.0, state="ON"))
    ema = stage.get_ema("esp32_tv")
    assert ema == 0.0  # should not count ON-state reading as phantom

# TEST 5A-6: Database write (Stage 6) — record persisted within 10s batch window
@pytest.mark.asyncio
async def test_stage6_database_batch_write():
    session = DBSession(config=load_config(), batch_interval_s=10)
    await session.queue_write({"device": "node_fridge", "power": 200.0})
    # Nothing written yet (batch window hasn't elapsed)
    assert await session.count_records("node_fridge") == 0
    # Advance time 10s
    with freeze_time(datetime.datetime.now() + datetime.timedelta(seconds=11)):
        await session.flush()
    assert await session.count_records("node_fridge") == 1

# TEST 5A-7: Analytics (Stage 7) — ToU cost updates after each reading
@pytest.mark.asyncio
async def test_stage7_analytics_updates_cost():
    analytics = AnalyticsStage(config=load_config())
    await analytics.process(mock_power_event("node_kettle", 2500.0))
    cost = analytics.get_accumulated_cost("node_kettle")
    assert cost > 0.0

# TEST 5A-8: Digital twin / PMV (Stage 8) — PMV emitted with correct range
@pytest.mark.asyncio
async def test_stage8_digital_twin_pmv_in_range():
    twin = DigitalTwinStage(config=load_config())
    result = await twin.process(mock_thermal_event(ta=22.0, tr=22.0, var=0.1,
                                                    rh=60.0, Icl=1.0, M=70.0))
    assert -3.0 <= result.pmv <= 3.0

# TEST 5A-9: RL agent (Stage 9) — output is an action command
@pytest.mark.asyncio
async def test_stage9_rl_agent_produces_action():
    agent_stage = RLAgentStage(config=load_config())
    result = await agent_stage.process(state={"node_hvac": 2000.0}, pmv=0.8,
                                        confidence=0.95)
    assert result.action in ("ON", "OFF", "NO_ACTION")

# TEST 5A-10: Latency monitor (Stage 10) — latency below 200ms SLA
@pytest.mark.asyncio
async def test_stage10_latency_sla():
    pipeline = FullPipeline(config=load_config())
    start = time.perf_counter()
    await pipeline.process(mock_power_event("node_fridge", 200.0))
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 200.0, f"Pipeline latency {elapsed_ms:.1f}ms exceeds 200ms SLA"

# TEST 5A-11: Broadcast (Stage 11) — JSON event sent to WebSocket clients
@pytest.mark.asyncio
async def test_stage11_broadcast_emits_json():
    broadcast_calls = []
    broadcaster = BroadcastStage(ws_broadcast_fn=lambda e: broadcast_calls.append(e))
    await broadcaster.broadcast({"event_type": "POWER_UPDATE", "device": "node_fridge"})
    assert len(broadcast_calls) == 1
    assert "event_type" in broadcast_calls[0]

# TEST 5A-12: Stage ordering is enforced — confidence gate runs before delta stability
@pytest.mark.asyncio
async def test_stage_ordering_confidence_before_delta():
    execution_order = []
    pipeline = FullPipeline(config=load_config(),
                             stage_hook=lambda name: execution_order.append(name))
    await pipeline.process(mock_power_event("node_fridge", 200.0))
    conf_idx   = execution_order.index("confidence_gate")
    delta_idx  = execution_order.index("delta_stability")
    assert conf_idx < delta_idx

# TEST 5A-13: LOW_CONFIDENCE event short-circuits RL stage execution
@pytest.mark.asyncio
async def test_low_confidence_skips_rl():
    rl_called = []
    pipeline = FullPipeline(config=load_config(),
                             rl_hook=lambda: rl_called.append(True))
    # Inject very low confidence (will fire LOW_CONFIDENCE after stage 3)
    await pipeline.process(mock_power_event_low_confidence("device", power=100.0))
    assert len(rl_called) == 0, "RL must not be called after LOW_CONFIDENCE"
