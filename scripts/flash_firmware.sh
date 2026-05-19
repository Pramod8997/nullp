#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Flash ESP32 firmware to all physical nodes
#
# Usage:
#   ./scripts/flash_firmware.sh
#
# Prerequisites:
#   pip install esptool
#   Connect each ESP32 to a USB port (/dev/ttyUSB0..3)
#
# The script flashes the compiled binary, then waits briefly
# to verify that each node connects to the Mosquitto broker.
# ─────────────────────────────────────────────────────────────

set -euo pipefail

BINARY="firmware/esp32_node/build/esp32_node.bin"

if [[ ! -f "$BINARY" ]]; then
    echo "❌ Binary not found at $BINARY"
    echo "   Build the firmware first (e.g., platformio run) or place the .bin here."
    exit 1
fi

# Map: device_id → serial port
# Adjust these ports to match your physical USB wiring.
declare -A DEVICES=(
    ["node_fridge"]="/dev/ttyUSB0"
    ["node_microwave"]="/dev/ttyUSB1"
    ["node_kettle"]="/dev/ttyUSB2"
    ["node_hvac"]="/dev/ttyUSB3"
)

FAILED=0

for DEV in "${!DEVICES[@]}"; do
    PORT="${DEVICES[$DEV]}"

    if [[ ! -e "$PORT" ]]; then
        echo "⚠️  $DEV: port $PORT not found — skipping"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "──────────────────────────────────────"
    echo "🔧 Flashing $DEV on $PORT ..."
    echo "──────────────────────────────────────"

    esptool.py --chip esp32 \
               --port "$PORT" \
               --baud 460800 \
               write_flash -z 0x1000 "$BINARY"

    if [[ $? -ne 0 ]]; then
        echo "❌ Flash FAILED for $DEV"
        FAILED=$((FAILED + 1))
    else
        echo "✅ $DEV flashed successfully"
    fi
done

echo ""
echo "══════════════════════════════════════"
if [[ $FAILED -eq 0 ]]; then
    echo "✅ All devices flashed successfully!"
else
    echo "⚠️  $FAILED device(s) failed or were skipped."
fi
echo "══════════════════════════════════════"

echo ""
echo "Waiting 5s for nodes to boot and connect to MQTT..."
sleep 5
echo "Done. Check Mosquitto logs to confirm topic registration:"
echo "  docker-compose exec mosquitto cat /mosquitto/log/mosquitto.log | tail -20"
