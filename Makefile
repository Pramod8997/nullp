# ═════════════════════════════════════════════════════════════════════════════
# Confidence-Aware Digital Twin EMS — Root Makefile
# ═════════════════════════════════════════════════════════════════════════════

PYTHON := $(shell which python3)
ifneq ($(wildcard .venv/bin/python3),)
	PYTHON := .venv/bin/python3
	PIP := .venv/bin/pip
else ifneq ($(wildcard venv/bin/python3),)
	PYTHON := venv/bin/python3
	PIP := venv/bin/pip
else
	PIP := pip3
endif

PWD := $(shell pwd)

.PHONY: help install install-backend install-frontend \
        run dev dev-backend dev-frontend dev-pipeline dev-sim \
        generate_data train_all train_synthetic \
        test test-backend test-frontend test-all test-safety test-e2e \
        lint lint-frontend build-frontend \
        docker-build docker-up docker-down \
        graph-update clean

# ── Default Target ──────────────────────────────────────────────────────────
help:
	@echo "=================================================================="
	@echo "  ⚡ Confidence-Aware Digital Twin EMS — Command Reference"
	@echo "=================================================================="
	@echo "  Setup & Installation:"
	@echo "    make install           - Install all Python & Frontend dependencies"
	@echo "    make install-backend   - Install Python dependencies only"
	@echo "    make install-frontend  - Install Frontend npm dependencies only"
	@echo ""
	@echo "  Development & Run:"
	@echo "    make run / make dev    - Start full stack (Mosquitto, Pipeline, API, Frontend, Sim)"
	@echo "    make demo              - Start full software demo (Pipeline, API, Virtual ESP32 Fleet)"
	@echo "    make hil-test          - Run 10-scenario Hardware-In-The-Loop test suite"
	@echo "    make dev-backend       - Start FastAPI backend (Port 8000)"
	@echo "    make dev-frontend      - Start Vite Tailwind frontend (Port 5173)"
	@echo "    make dev-pipeline      - Start real-time NILM ingestion pipeline"
	@echo "    make dev-sim           - Start ESP32 telemetry simulator"
	@echo ""
	@echo "  Data & Model Training:"
	@echo "    make generate_data     - Generate synthetic & mock UK-DALE data"
	@echo "    make train_all         - Train ProtoNet + OpenMax on all datasets (CUDA)"
	@echo "    make train_synthetic   - Train models on synthetic dataset"
	@echo ""
	@echo "  Testing & Quality:"
	@echo "    make test              - Run all backend pytest & frontend vitest tests"
	@echo "    make test-backend      - Run Python pytest suite"
	@echo "    make test-frontend     - Run Frontend vitest suite"
	@echo "    make test-safety       - Run safety cutoff & arc fault test"
	@echo "    make test-e2e          - Run end-to-end integration tests"
	@echo "    make lint              - Run frontend ESLint"
	@echo "    make build-frontend    - Build production Tailwind frontend bundle"
	@echo ""
	@echo "  Docker & Maintenance:"
	@echo "    make docker-up         - Start services with docker-compose"
	@echo "    make docker-down       - Stop docker-compose services"
	@echo "    make graph-update      - Update Graphify knowledge graph"
	@echo "    make clean             - Remove DB state, pycache, dist & build artifacts"
	@echo "=================================================================="

# ── Installation ────────────────────────────────────────────────────────────
install: install-backend install-frontend

install-backend:
	$(PIP) install -r requirements.txt

install-frontend:
	cd frontend && npm install

# ── Development & Full Stack Runner ─────────────────────────────────────────
run: dev

dev:
	@echo "⚡ Starting Confidence-Aware Digital Twin EMS..."
	@bash -c 'trap "kill 0" SIGINT SIGTERM EXIT; \
	if ss -tlnp 2>/dev/null | grep -q ":1883 "; then \
		echo "✅ Mosquitto broker already running on port 1883"; \
	elif command -v mosquitto >/dev/null 2>&1; then \
		mosquitto -c mosquitto/config/mosquitto.conf -d; \
		echo "✅ Mosquitto broker started"; \
	else \
		echo "⚠️  mosquitto not found — start via Docker or sudo apt-get install -y mosquitto"; \
	fi; \
	sleep 1; \
	export PYTHONPATH=$(PWD) && $(PYTHON) scripts/run_pipeline.py & \
	export PYTHONPATH=$(PWD) && $(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & \
	(cd frontend && npm run dev) & \
	sleep 2; \
	export PYTHONPATH=$(PWD) && $(PYTHON) backend/scripts/simulate_esp32.py & \
	wait'

dev-backend:
	export PYTHONPATH=$(PWD) && $(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dev-frontend:
	cd frontend && npm run dev

dev-pipeline:
	export PYTHONPATH=$(PWD) && $(PYTHON) scripts/run_pipeline.py

demo:
	export PYTHONPATH=$(PWD) && $(PYTHON) scripts/demo_full_system.py

hil-test:
	export PYTHONPATH=$(PWD) && $(PYTHON) scripts/hil_hardware_test.py

stress-test:
	export PYTHONPATH=$(PWD) && $(PYTHON) scripts/stress_test_hardware_sim.py

sim-test:
	export PYTHONPATH=$(PWD) && $(PYTHON) scripts/test_firmware_and_ai_e2e.py

dev-sim:
	export PYTHONPATH=$(PWD) && $(PYTHON) backend/scripts/simulate_esp32.py --all

# ── Data Generation & Training ──────────────────────────────────────────────
generate_data:
	export PYTHONPATH=$(PWD) && $(PYTHON) scripts/generate_mock_ukdale.py

train_all:
	export PYTHONPATH=$(PWD) && $(PYTHON) scripts/train_models.py --datasets synthetic ukdale redd --episodes 2000 --cuda

train_synthetic:
	export PYTHONPATH=$(PWD) && $(PYTHON) scripts/train_models.py --datasets synthetic --episodes 2000

# ── Testing ─────────────────────────────────────────────────────────────────
test: test-backend test-frontend

test-backend:
	export PYTHONPATH=$(PWD) && $(PYTHON) -m pytest tests/ -v --tb=short

test-frontend:
	cd frontend && npm test

test-safety:
	export PYTHONPATH=$(PWD) && $(PYTHON) scripts/test_safety_cutoff.py --spike 4000

test-e2e:
	export PYTHONPATH=$(PWD) && $(PYTHON) -m pytest tests/test_e2e.py -v

# ── Build & Lint ────────────────────────────────────────────────────────────
lint: lint-frontend

lint-frontend:
	cd frontend && npm run lint

build-frontend:
	cd frontend && npm run build

# ── Docker ──────────────────────────────────────────────────────────────────
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

# ── Knowledge Graph ─────────────────────────────────────────────────────────
graph-update:
	graphify update .

# ── Cleanup ─────────────────────────────────────────────────────────────────
clean:
	rm -f data/ems_state.db data/ems_state.db-shm data/ems_state.db-wal
	rm -rf frontend/dist
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
