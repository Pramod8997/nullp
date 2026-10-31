import urllib.request
import json
import time
import sys

ESP_URL = 'http://172.20.10.10'
API_URL = 'http://127.0.0.1:8000'

def get_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode('utf-8'))

def post_json(url, data):
    body = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=body,
        headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'},
        method='POST'
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode('utf-8'))

def main():
    print('=== 1. VERIFYING WI-FI CONNECTIVITY ===')
    for _ in range(10):
        status = get_json(f'{ESP_URL}/status')
        if status.get('mqtt_connected'):
            break
        print('Waiting for MQTT broker handshake...')
        time.sleep(1.0)
    print(f"WiFi Mode: {status.get('wifi_mode')}")
    print(f"Node IP: {status.get('ip')}")
    print(f"MQTT Connected: {status.get('mqtt_connected')}")
    print(f"Initial Telemetry: V={status.get('voltage')}V, P={status.get('power')}W, R1={status.get('relay1')}, R2={status.get('relay2')}")

    # Clean initialize: set both relays to OFF
    get_json(f'{ESP_URL}/toggle?relay=1&state=0')
    get_json(f'{ESP_URL}/toggle?relay=2&state=0')
    time.sleep(1.0)

    print('\n=== 2. TESTING RELAY 1 (LAPTOP ~65W) ENERGIZATION ===')
    get_json(f'{ESP_URL}/toggle?relay=1&state=1')
    time.sleep(1.2)
    s1 = get_json(f'{ESP_URL}/status')
    print(f"Relay 1 ON -> R1={s1.get('relay1')}, R2={s1.get('relay2')}, Power={s1.get('power')}W, Current={s1.get('current')}A")
    assert s1.get('relay1') is True, "Relay 1 must be true"

    print('\n=== 3. TESTING RELAY 2 (PHONE ~25W) ENERGIZATION -> TOTAL ~90W OVERLOAD (>80W) ===')
    get_json(f'{ESP_URL}/toggle?relay=2&state=1')
    time.sleep(1.0)
    s2 = get_json(f'{ESP_URL}/status')
    print(f"Both Relays ON -> R1={s2.get('relay1')}, R2={s2.get('relay2')}, Power={s2.get('power')}W")
    print(f"Overload Active: {s2.get('overload_active')}, Countdown: {s2.get('countdown')}s")

    print('\n=== 4. WAITING 3.5s FOR AUTOMATIC 80W CUTOFF / LOAD-SHEDDING ===')
    time.sleep(3.5)
    s3 = get_json(f'{ESP_URL}/status')
    print(f"After 3.5s -> R1={s3.get('relay1')}, R2={s3.get('relay2')}, Power={s3.get('power')}W")
    print(f"Trip Count: {s3.get('trip_count')}")
    # Remove HTML tags and non-ascii from alert
    alert = s3.get('alert', '').replace('<b>', '').replace('</b>', '')
    safe_alert = alert.encode('ascii', errors='ignore').decode('ascii')
    print(f"Alert: {safe_alert}")
    assert s3.get('relay2') is False, 'Relay 2 must be shed!'
    assert s3.get('relay1') is True, 'Relay 1 (high priority) must stay ON!'
    print('==> AUTOMATIC 80W CUTOFF VERIFIED SUCCESSFUL!')

    print('\n=== 5. TESTING FASTAPI / MQTT RELAY CUTOFF COMMAND ===')
    cmd_res = post_json(f'{API_URL}/api/relay/command', {'device_id': 'node_bench_agg', 'command': 'OFF'})
    print(f"FastAPI POST /api/relay/command (OFF) -> {cmd_res}")
    time.sleep(2.0)
    s4 = get_json(f'{ESP_URL}/status')
    print(f"After Cutoff Command -> R1={s4.get('relay1')}, R2={s4.get('relay2')}, Power={s4.get('power')}W")
    assert s4.get('relay1') is False and s4.get('relay2') is False, 'Both relays must be OFF!'
    print('==> MANUAL CUTOFF VIA FASTAPI & MQTT VERIFIED SUCCESSFUL!')

    print('\n=== ALL HARDWARE, WI-FI & CUTOFF CHECKS PASSED PERFECTLY! ===')

if __name__ == '__main__':
    main()
