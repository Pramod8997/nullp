import asyncio
import signal
import sys
import logging
from typing import Union
from src.database.session import DatabaseSession
from src.hardware.mqtt import MQTTClientManager
from src.pipeline.safety import SafetyMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EMSOrchestrator:
    def __init__(self) -> None:
        self.db = DatabaseSession("data/ems_state.db")
        self.mqtt = MQTTClientManager("localhost", 1883)
        
        import yaml
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        safety_config = config.get("system_safety", {})
        max_agg = safety_config.get("max_aggregate_wattage", 3500.0)
        dev_limits = safety_config.get("device_wattage_limits", {})
        
        self.safety = SafetyMonitor(max_agg, dev_limits, self._trigger_cutoff)
        self._running = False

        from src.rl.agent import TabularQLearningAgent
        import pickle
        import os
        self.agent = TabularQLearningAgent()
        weights_path = "backend/models/weights/q_table.pkl"
        if os.path.exists(weights_path):
            with open(weights_path, "rb") as f:
                self.agent.q_table = pickle.load(f)
        
        self.device_states = {"fridge": 0, "microwave": 0, "kettle": 0, "default": 0}
        self.last_action_time = {"fridge": 0, "microwave": 0, "kettle": 0, "default": 0}
        
    async def _trigger_cutoff(self, device_id: str) -> None:
        await self.mqtt.publish_command(f"home/plug/{device_id}/command", "OFF")

    async def _handle_mqtt_message(self, topic: str, payload: Union[str, bytes, bytearray, int, float, None]) -> None:
        try:
            device_id = topic.split("/")[-2]
            power_watts = float(payload) if payload else 0.0
            
            # [INV-7] Safety Check First
            if await self.safety.process_reading(device_id, power_watts):
                # Valid reading, store it
                import time
                await self.db.insert_measurement(time.time(), device_id, power_watts)
                
                # Update local device state
                self.device_states[device_id] = 1 if power_watts > 10 else 0
                
                # FEATURE 1: The Active Learning Loop (Unknown Devices)
                # Mock ProtoNet confidence evaluation
                import random
                import json
                if power_watts > 50 and random.random() < 0.05:
                    logger.warning(f"ProtoNet Anomaly: UNKNOWN_DEVICE detected on {device_id} at {power_watts}W")
                    await self.mqtt.publish_command(
                        "home/ui/events", 
                        json.dumps({"type": "UNKNOWN_DEVICE", "device_id": device_id, "power": power_watts})
                    )
                
                # FEATURE 2: RL Agent MQTT Dispatch (Control Loop)
                current_hour = time.localtime().tm_hour
                power_bin = min(9, int(power_watts / 500))
                dev_tuple = tuple(self.device_states.get(k, 0) for k in ["fridge", "microwave", "kettle", "default"])
                
                state = self.agent.get_state_tuple(current_hour, power_bin, dev_tuple)
                action = self.agent.get_action(state)
                
                # Decode action for the current device
                device_names = ["fridge", "microwave", "kettle", "default"]
                device_idx = device_names.index(device_id) if device_id in device_names else 3
                
                temp_act = action
                action_decoded = []
                for _ in range(self.agent.MAX_RL_DEVICES):
                    action_decoded.append(temp_act % 3)
                    temp_act //= 3
                
                dev_action = action_decoded[device_idx]
                
                if dev_action == 0 and self.device_states[device_id] == 1:
                    current_time = time.time()
                    if current_time - self.last_action_time.get(device_id, 0) > 10.0:
                        # TURN_OFF action
                        logger.info(f"RL Agent decision: Turning off {device_id}")
                        await self.mqtt.publish_command(f"home/plug/{device_id}/command", "OFF")
                        await self.mqtt.publish_command(
                            "home/ui/events", 
                            json.dumps({"type": "RL_ACTION", "message": f"Agent turned off {device_id} to optimize grid."})
                        )
                        
                        # Apply cooldown
                        self.last_action_time[device_id] = current_time
                        
                        # Update Q-table
                        self.device_states[device_id] = 0
                        next_dev_tuple = tuple(self.device_states.get(k, 0) for k in ["fridge", "microwave", "kettle", "default"])
                        next_state = self.agent.get_state_tuple(current_hour, power_bin, next_dev_tuple)
                        
                        # Comfort Penalty logic
                        reward = -10.0 if "fridge" in device_id else 1.0
                        
                        self.agent.update(state, action, reward=reward, next_state=next_state)
                
        except Exception as e:
            logger.error(f"Failed handling message on {topic}: {e}")

    async def run(self) -> None:
        self._running = True
        await self.db.connect()
        self.mqtt.set_read_callback(self._handle_mqtt_message)
        
        # Start MQTT listener in background
        mqtt_task = asyncio.create_task(self.mqtt.run("home/sensor/+/power"))
        
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            mqtt_task.cancel()
            await self.db.close()

    def shutdown(self) -> None:
        logger.info("Initiating graceful shutdown...")
        self._running = False

async def main() -> None:
    orchestrator = EMSOrchestrator()
    
    loop = asyncio.get_running_loop()
    if sys.platform != 'win32':
        loop.add_signal_handler(signal.SIGINT, lambda: orchestrator.shutdown())
        loop.add_signal_handler(signal.SIGTERM, lambda: orchestrator.shutdown())
    
    await orchestrator.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
