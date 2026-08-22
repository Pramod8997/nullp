import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.safety import FleetDiagnosticsMonitor

async def test_safety_negative():
    print("\n--- Testing Safety Monitor with Negative values ---")
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
        MockMessage("home/sensor/esp32_hvac/power", "-10000"), # -10000W
        MockMessage("home/sensor/esp32_hvac/power", "-10000")
    ]
    
    client = MockClient(msgs)
    
    try:
        await asyncio.wait_for(safety.run_forever(client, relay_cb), timeout=0.5)
    except asyncio.TimeoutError:
        pass
        
    print(f"Safety Actions emitted: {actions}")

async def main():
    await test_safety_negative()

if __name__ == "__main__":
    asyncio.run(main())
