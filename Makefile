# ═════════════════════════════════════════════════════════════════════════════
# Confidence-Aware Digital Twin EMS — Root Makefile
#
# The targets below follow the current repository entry points:
#   - scripts/demo_full_system.py       software demo orchestration
#   - config/config.*.yaml               runtime profiles
#   - docker-compose.yml                 production service stack
#   - firmware/tests and PlatformIO      host and ESP32 firmware checks
# ═════════════════════════════════════════════════════════════════════════════

.DEFAULT_GOAL := help

# Allow callers to override tools, while selecting the checked-in virtualenv
# when one exists on either Unix or Windows.
ifeq ($(origin PYTHON),undefined)
ifeq ($(OS),Windows_NT)
PYTHON := python
VENV_PYTHON := $(firstword $(wildcard .venv/Scripts/python.exe venv/Scripts/python.exe))
else
PYTHON := python3
VENV_PYTHON := $(firstword $(wildcard .venv/bin/python3 .venv/bin/python venv/bin/python3 venv/bin/python))
endif
ifneq ($(strip $(VENV_PYTHON)),)
PYTHON := $(VENV_PYTHON)
endif
endif

PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
NPM ?= npm
DOCKER ?= docker
PIO ?= pio
CMAKE ?= cmake
ROOT := $(CURDIR)
FIRMWARE_TEST_BUILD ?= build/firmware-tests

ifeq ($(OS),Windows_NT)
FIRMWARE_TEST_BIN := $(FIRMWARE_TEST_BUILD)/Release/run_host_tests.exe
else
FIRMWARE_TEST_BIN := $(FIRMWARE_TEST_BUILD)/run_host_tests
endif

.PHONY: help install install-backend install-frontend \
        run dev live hardware demo production prod dev-backend dev-frontend \
        dev-pipeline dev-pipeline-demo dev-pipeline-hardware \
        broker dev-sim dev-sim-demo \
        relay-on relay-off relay-in1-on relay-in1-off relay-in2-on relay-in2-off \
        hil-test stress-test sim-test test-firmware-sim test-firmware-host \
        generate_data train_all train_synthetic train_demo enroll_demo \
        test test-all test-backend test-frontend test-safety test-e2e \
        verify-ml verify-chaos \
        lint lint-frontend build-frontend \
        firmware-build firmware-upload firmware-monitor \
        docker-build docker-up docker-down docker-logs \
        graph-update clean

# ── Default Target ──────────────────────────────────────────────────────────
help:
	@echo "=================================================================="
	@echo "  ⚡ Confidence-Aware Digital Twin EMS — Command Reference"
	@echo "=================================================================="
	@echo "  Setup & Installation:"
	@echo "    make install           - Install all Python & Frontend dependencies"
	@echo "    make install-backend   - Install Python dependencies only"
	@echo "    make install-frontend  - Install locked Frontend npm dependencies"
	@echo ""
	@echo "  Live Hardware & Development:"
	@echo "    make run / make dev    - Start full live stack connected to ESP32 (Broker, Pipeline, Server, Frontend)"
	@echo "    make live / hardware   - Alias to start the physical ESP32 rig stack"
	@echo "    make demo              - Start software demo with virtual fleet simulator"
	@echo "    make broker            - Start standalone MQTT broker on 0.0.0.0:1883"
	@echo "    make dev-backend       - Start FastAPI backend (Port 8000)"
	@echo "    make dev-frontend      - Start Vite Tailwind frontend (Port 5173)"
	@echo "    make dev-pipeline-hardware - Start hardware pipeline (config.hardware.yaml)"
	@echo ""
	@echo "  Hardware Relay Controls:"
	@echo "    make relay-off         - Emergency Cutoff: cuts BOTH IN1 & IN2 relays"
	@echo "    make relay-on          - Master ON: energizes BOTH IN1 & IN2 relays"
	@echo "    make relay-in1-off     - Cut Relay 1 (IN1: Laptop Charger, GPIO 26)"
	@echo "    make relay-in1-on      - Turn ON Relay 1 (IN1: Laptop Charger, GPIO 26)"
	@echo "    make relay-in2-off     - Cut Relay 2 (IN2: Phone Charger, GPIO 27)"
	@echo "    make relay-in2-on      - Turn ON Relay 2 (IN2: Phone Charger, GPIO 27)"
	@echo ""
	@echo "  Data & Model Training:"
	@echo "    make generate_data     - Generate synthetic & mock UK-DALE data"
	@echo "    make train_all         - Train ProtoNet + OpenMax on all datasets (ukdale redd synd)"
	@echo "    make train_synthetic   - Train models on synthetic dataset"
	@echo "    make train_demo        - Train the demo-profile consumer-electronics model"
	@echo "    make enroll_demo CAPTURE=file.npz - Enroll measured demo-device windows"
	@echo ""
	@echo "  Testing & Quality:"
	@echo "    make test              - Run backend pytest & frontend vitest tests"
	@echo "    make test-all          - Run application tests and host firmware tests"
	@echo "    make test-backend      - Run the CI-aligned Python pytest suite"
	@echo "    make test-frontend     - Run Frontend vitest suite"
	@echo "    make test-safety       - Run safety cutoff & arc fault test"
	@echo "    make test-e2e          - Run end-to-end integration tests"
	@echo "    make test-firmware-host - Build and run host-side firmware tests"
	@echo "    make test-firmware-sim  - Run firmware/AI simulation integration test"
	@echo "    make verify-ml         - Run ML verification checks"
	@echo "    make verify-chaos      - Run adversarial/chaos verification checks"
	@echo "    make lint              - Run frontend ESLint"
	@echo "    make build-frontend    - Build production Tailwind frontend bundle"
	@echo ""
	@echo "  Firmware, Docker & Maintenance:"
	@echo "    make firmware-build    - Build ESP32 firmware with PlatformIO"
	@echo "    make firmware-upload   - Flash ESP32 firmware with PlatformIO"
	@echo "    make docker-up         - Start services with docker-compose"
	@echo "    make docker-down       - Stop docker-compose services"
	@echo "    make docker-logs       - Follow docker-compose service logs"
	@echo "    make graph-update      - Update Graphify knowledge graph"
	@echo "    make clean             - Remove DB state, pycache, dist & build artifacts"
	@echo "=================================================================="

# ── Installation ────────────────────────────────────────────────────────────
install: install-backend install-frontend

install-backend:
	$(PIP) install -r requirements.txt

install-frontend:
	cd frontend && $(NPM) ci

# ── Development & Full Stack Runner ─────────────────────────────────────────
run: dev

dev:
	$(PYTHON) scripts/run_hardware_system.py

live: dev

hardware: dev

broker:
	$(PYTHON) scripts/start_broker.py 0.0.0.0

dev-backend:
	$(PYTHON) scripts/start_api.py

dev-frontend:
	cd frontend && $(NPM) run dev

dev-pipeline:
	$(PYTHON) scripts/run_pipeline.py

dev-pipeline-demo:
	$(PYTHON) scripts/run_pipeline.py --config config/config.demo.yaml

dev-pipeline-hardware:
	$(PYTHON) scripts/run_pipeline.py --config config/config.hardware.yaml

# ── Direct Relay Controls via MQTT ──────────────────────────────────────────
relay-on:
	$(PYTHON) -c "import paho.mqtt.publish as pub; pub.single('home/plug/node_bench_agg/command', 'ON', hostname='127.0.0.1')"

relay-off:
	$(PYTHON) -c "import paho.mqtt.publish as pub; pub.single('home/plug/node_bench_agg/command', 'OFF', hostname='127.0.0.1')"

relay-in1-on:
	$(PYTHON) -c "import paho.mqtt.publish as pub; pub.single('home/plug/node_bench_agg/command', 'IN1_ON', hostname='127.0.0.1')"

relay-in1-off:
	$(PYTHON) -c "import paho.mqtt.publish as pub; pub.single('home/plug/node_bench_agg/command', 'IN1_OFF', hostname='127.0.0.1')"

relay-in2-on:
	$(PYTHON) -c "import paho.mqtt.publish as pub; pub.single('home/plug/node_bench_agg/command', 'IN2_ON', hostname='127.0.0.1')"

relay-in2-off:
	$(PYTHON) -c "import paho.mqtt.publish as pub; pub.single('home/plug/node_bench_agg/command', 'IN2_OFF', hostname='127.0.0.1')"

demo:
	$(PYTHON) scripts/demo_full_system.py --demo

production: docker-build docker-up

prod: production

hil-test:
	$(PYTHON) scripts/hil_hardware_test.py

stress-test:
	$(PYTHON) scripts/stress_test_hardware_sim.py

sim-test:
	$(PYTHON) scripts/test_firmware_and_ai_e2e.py

test-firmware-sim: sim-test

test-firmware-host:
	$(CMAKE) -S firmware/tests -B $(FIRMWARE_TEST_BUILD) -DCMAKE_BUILD_TYPE=Release
	$(CMAKE) --build $(FIRMWARE_TEST_BUILD) --config Release
	$(FIRMWARE_TEST_BIN)

dev-sim:
	$(PYTHON) backend/scripts/simulate_esp32.py --all

dev-sim-demo:
	$(PYTHON) backend/scripts/simulate_esp32.py --demo

# ── Data Generation & Training ──────────────────────────────────────────────
generate_data:
	$(PYTHON) scripts/generate_mock_ukdale.py

train_all:
	$(PYTHON) scripts/train_models.py --sources ukdale redd synd --episodes 2000

train_synthetic:
	$(PYTHON) scripts/train_models.py --sources synd --episodes 2000

train_demo:
	$(PYTHON) scripts/train_demo_models.py $(TRAIN_DEMO_ARGS)

enroll_demo:
ifndef CAPTURE
	$(error CAPTURE is required, for example: make enroll_demo CAPTURE=bench.npz)
endif
	$(PYTHON) scripts/enroll_demo_devices.py --capture "$(CAPTURE)" $(ENROLL_ARGS)

# ── Testing ─────────────────────────────────────────────────────────────────
test: test-backend test-frontend

test-all: test test-firmware-host

test-backend:
	$(PYTEST) tests/ -v --tb=short --ignore=tests/test_e2e.py -m "not e2e"

test-frontend:
	cd frontend && $(NPM) test

test-safety:
	$(PYTHON) scripts/test_safety_cutoff.py --spike 4000

test-e2e:
	$(PYTEST) tests/test_e2e.py -v --tb=short

verify-ml:
	$(PYTHON) scripts/verify_ml_agent.py

verify-chaos:
	$(PYTHON) scripts/verify_chaos_agent.py

# ── Build & Lint ────────────────────────────────────────────────────────────
lint: lint-frontend

lint-frontend:
	cd frontend && $(NPM) run lint

build-frontend:
	cd frontend && $(NPM) run build

# ── Docker ──────────────────────────────────────────────────────────────────
docker-build:
	$(DOCKER) compose build

docker-up:
	$(DOCKER) compose up -d

docker-down:
	$(DOCKER) compose down

docker-logs:
	$(DOCKER) compose logs -f

# ── Firmware ────────────────────────────────────────────────────────────────
firmware-build:
	cd firmware/esp32_node && $(PIO) run

firmware-upload:
	cd firmware/esp32_node && $(PIO) run --target upload

firmware-monitor:
	cd firmware/esp32_node && $(PIO) device monitor -b 115200

# ── Knowledge Graph ─────────────────────────────────────────────────────────
graph-update:
	graphify update .

# ── Cleanup ─────────────────────────────────────────────────────────────────
clean:
	rm -f data/ems_state.db data/ems_state.db-shm data/ems_state.db-wal
	rm -rf frontend/dist
	rm -rf $(FIRMWARE_TEST_BUILD)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
