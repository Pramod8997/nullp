import asyncio
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rl.agent import TabularQLearningAgent
from src.rl.dqn_agent import DQNAgent
from src.pipeline.safety import FleetDiagnosticsMonitor
from src.pipeline.watchdog import SoftAnomalyWatchdog

print("========================================")
print("CHAOS TEST REPORT")
print("========================================\n")

async def test_rl_adversarial():
    print("--- Phase 2: RL Agent Adversarial Analysis ---")
    agent = TabularQLearningAgent()
    dqn = DQNAgent()
    
    state = {"devices": {"esp32_hvac": 1000}, "price_tier": 1, "pmv_zone": 1, "tod": 2}
    
    # 1. NaN PMV
    try:
        act1 = agent.act(state, pmv=float('nan'), confidence=0.99, classified_device="esp32_hvac")
        act2 = dqn.act(state, pmv=float('nan'), confidence=0.99, classified_device="esp32_hvac")
        print(f"NaN PMV -> Tabular: {act1}, DQN: {act2} (FAIL: should explicitly handle NaN)")
    except Exception as e:
        print(f"NaN PMV caused crash: {e}")

    # 2. Extremely large inputs
    try:
        act1 = agent.act(state, pmv=100000.0, confidence=0.99, classified_device="esp32_hvac")
        print(f"Extremely large PMV -> Tabular: {act1}")
    except Exception as e:
        print(f"Extremely large PMV caused crash: {e}")

    # 3. Malformed state dictionary (missing keys)
    try:
        bad_state = {}
        act1 = agent.act(bad_state, pmv=0.0, confidence=0.99, classified_device="esp32_hvac")
        act2 = dqn.act(bad_state, pmv=0.0, confidence=0.99, classified_device="esp32_hvac")
        print(f"Empty state -> Tabular: {act1}, DQN: {act2}")
    except Exception as e:
        print(f"Empty state caused crash: {e} (FAIL)")

async def test_safety_boundary():
    print("\n--- Phase 4: Safety Module Boundary Conditions ---")
    safety = FleetDiagnosticsMonitor(3500.0, {"esp32_hvac": 1000.0})
    
    class MockMessage:
        def __init__(self, topic, payload):
            self.topic = type('obj', (object,), {'__str__': lambda self_: topic})()
            self.payload = payload if isinstance(payload, bytes) else payload.encode()
            
    class AsyncIterList:
        def __init__(self, items):
            self.items = items
            self.idx = 0
        def __aiter__(self):
            return self
        async def __anext__(self):
            if self.idx < len(self.items):
                item = self.items[self.idx]
                self.idx += 1
                return item
            raise StopAsyncIteration
            
    class MockClient:
        def __init__(self, messages):
            self.messages = AsyncIterList(messages)
            
    actions = []
    async def relay_cb(device_id, action):
        actions.append((device_id, action))
        
    # Test boundary conditions: negative watts, NaN, Inf
    msgs = [
        MockMessage("home/sensor/esp32_hvac/power", "-5000"), # Negative power bypasses threshold?
        MockMessage("home/sensor/esp32_hvac/power", "-10000"),
        MockMessage("home/sensor/esp32_hvac/power", "NaN"),
        MockMessage("home/sensor/esp32_hvac/power", "Inf")
    ]
    
    client = MockClient(msgs)
    
    try:
        await asyncio.wait_for(safety.run_forever(client, relay_cb), timeout=0.5)
    except asyncio.TimeoutError:
        pass
    except Exception as e:
        print(f"Safety Monitor crashed: {e}")
        
    print(f"Safety Actions emitted for negative/NaN/Inf: {actions} (FAIL: should flag negative as anomaly)")

def test_watchdog():
    print("\n--- Phase 4: Watchdog Broker Failure ---")
    watchdog = SoftAnomalyWatchdog(window_size=60, z_score_threshold=3.0)
    
    # 1. NaN reading
    try:
        res = watchdog.check_reading("sensor1", float('nan'))
        print(f"Watchdog with NaN -> {res} (FAIL: should handle gracefully)")
    except Exception as e:
        print(f"Watchdog NaN crashed: {e}")

    # 2. Inf reading
    try:
        res = watchdog.check_reading("sensor1", float('inf'))
        print(f"Watchdog with Inf -> {res} (FAIL: should handle gracefully)")
    except Exception as e:
        print(f"Watchdog Inf crashed: {e}")
        
    # Broker failure: what if broker sends None or malformed strings?
    try:
        res = watchdog.check_reading("sensor1", "not_a_number")
        print(f"Watchdog with malformed -> {res}")
    except Exception as e:
        print(f"Watchdog string crashed: {e} (FAIL: should handle missing/malformed)")

async def main():
    await test_rl_adversarial()
    await test_safety_boundary()
    test_watchdog()

if __name__ == "__main__":
    asyncio.run(main())
