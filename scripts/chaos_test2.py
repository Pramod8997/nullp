import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rl.agent import TabularQLearningAgent
from src.rl.dqn_agent import DQNAgent
from src.pipeline.safety import FleetDiagnosticsMonitor

async def test_rl_nan():
    print("\n--- Testing RL Agents with NaN/Inf ---")
    
    agent = TabularQLearningAgent()
    dqn = DQNAgent()
    # Force exploit for DQN
    dqn.epsilon = 0.0
    agent.epsilon = 0.0
    
    state = {"devices": {"esp32_hvac": 1000}, "price_tier": 1, "pmv_zone": 1, "tod": 2}
    
    # Test NaN
    act1 = agent.act(state, pmv=float('nan'), confidence=0.99, classified_device="esp32_hvac")
    print(f"Tabular Agent action for NaN PMV: {act1}")

    act2 = dqn.act(state, pmv=float('nan'), confidence=0.99, classified_device="esp32_hvac")
    print(f"DQN Agent action for NaN PMV: {act2}")

async def test_safety_nan():
    print("\n--- Testing Safety Monitor with NaN/Inf ---")
    safety = FleetDiagnosticsMonitor(3500.0, {"esp32_hvac": 1000.0})
    
    class MockMessage:
        def __init__(self, topic, payload):
            self.topic = type('obj', (object,), {'__str__': lambda self_: topic})()
            self.payload = payload.encode()
            
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
        
    msgs = [
        MockMessage("home/sensor/esp32_hvac/power", "NaN"),
        MockMessage("home/sensor/esp32_hvac/power", "Inf"),
        MockMessage("home/sensor/esp32_hvac/power", "-5000"),
        MockMessage("home/sensor/esp32_hvac/power", "3000")
    ]
    
    client = MockClient(msgs)
    
    try:
        await asyncio.wait_for(safety.run_forever(client, relay_cb), timeout=0.5)
    except asyncio.TimeoutError:
        pass
    except Exception as e:
        print(f"Safety Error: {e}")
        
    print(f"Safety Actions emitted: {actions}")

async def main():
    await test_rl_nan()
    await test_safety_nan()

if __name__ == "__main__":
    asyncio.run(main())
