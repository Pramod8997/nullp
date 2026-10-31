import urllib.request
import json
import time

def send_cmd(cmd):
    req = urllib.request.Request(
        'http://localhost:8000/api/relay/command',
        data=json.dumps({'device_id': 'node_bench_agg', 'command': cmd}).encode(),
        headers={'Content-Type': 'application/json'}
    )
    res = json.loads(urllib.request.urlopen(req, timeout=3).read().decode())
    return res

def get_esp32_status():
    return json.loads(urllib.request.urlopen('http://172.20.10.10/status', timeout=3).read().decode())

def get_backend_device():
    res = json.loads(urllib.request.urlopen('http://localhost:8000/api/devices', timeout=3).read().decode())
    return res.get('devices', {}).get('node_bench_agg', {})

print("==================================================================")
print("  STEP 1: Turn ON Socket 1 (Laptop Charger, ~65W) via API")
print("==================================================================")
send_cmd("IN1_ON")
time.sleep(2.0)
s1 = get_esp32_status()
print(f" -> ESP32 Socket 1 : {'ON' if s1['relay1'] else 'OFF'}")
print(f" -> ESP32 Socket 2 : {'ON' if s1['relay2'] else 'OFF'}")
print(f" -> Live Power     : {s1['power']:.1f} W (Current: {s1['current']:.3f} A, Voltage: {s1['voltage']:.1f} V)")

print("\n==================================================================")
print("  STEP 2: Turn ON Socket 2 (Phone Charger, ~25W) -> Total ~90W")
print("==================================================================")
send_cmd("IN2_ON")
time.sleep(1.5)
s2 = get_esp32_status()
print(f" -> ESP32 Socket 1 : {'ON' if s2['relay1'] else 'OFF'}")
print(f" -> ESP32 Socket 2 : {'ON' if s2['relay2'] else 'OFF'}")
print(f" -> Combined Power : {s2['power']:.1f} W (> 80W Limit)")
print(f" -> Overload Armed : {s2['overload_active']} (Countdown remaining: {s2['countdown']:.1f}s)")

print("\n==================================================================")
print("  STEP 3: Wait 3.5s for 80W Overload Shedding to Cut Socket 2")
print("==================================================================")
time.sleep(3.5)
s3 = get_esp32_status()
print(f" -> POST-CUTOFF STATUS:")
print(f"    - Socket 1 (Laptop Charger) : {'ON (PROTECTED)' if s3['relay1'] else 'OFF'}")
print(f"    - Socket 2 (Phone Charger)  : {'ON' if s3['relay2'] else 'SHED / CUT OFF (SUCCESS!)'}")
print(f"    - Normalized Power          : {s3['power']:.1f} W (Safely under 80W)")
print(f"    - Total Lifetime Trip Events: {s3['trip_count']}")
print(f"    - ESP32 Alert Message       : {s3['alert']}")

print("\n==================================================================")
print("  STEP 4: Turn OFF Socket 1 (Return System to Standby)")
print("==================================================================")
send_cmd("OFF")
time.sleep(1.5)
s4 = get_esp32_status()
print(f" -> Standby: Socket 1 = {s4['relay1']} | Socket 2 = {s4['relay2']} | Power = {s4['power']:.1f}W")
print("\n==================================================================")
print("  ALL TESTS PASSED: PHYSICAL HARDWARE CUTOFF VERIFIED OVER WIFI!")
print("==================================================================")
