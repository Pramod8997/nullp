PYTHON=venv/bin/python3
PIP=venv/bin/pip
PWD=$(shell pwd)

.PHONY: install train_all test_safety test run clean

install:
	$(PIP) install -r requirements.txt
	cd frontend && npm install

train_all:
	export PYTHONPATH=$(PWD) && $(PYTHON) scripts/train_models.py --datasets ukdale redd synd --cuda

test_safety:
	export PYTHONPATH=$(PWD) && $(PYTHON) scripts/test_safety_cutoff.py --spike 4000

test:
	export PYTHONPATH=$(PWD) && $(PYTHON) -m pytest tests/ -v --tb=short

run:
	@echo "Starting Confidence-Aware Digital Twin EMS..."
	@bash -c 'trap "kill 0" SIGINT SIGTERM EXIT; \
	$(PYTHON) scripts/start_broker.py & \
	sleep 2; \
	export PYTHONPATH=$(PWD) && $(PYTHON) scripts/run_pipeline.py & \
	export PYTHONPATH=$(PWD) && $(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & \
	(cd frontend && npm run dev) & \
	export PYTHONPATH=$(PWD) && $(PYTHON) backend/scripts/simulate_esp32.py & \
	wait'

clean:
	rm -f data/ems_state.db data/ems_state.db-shm data/ems_state.db-wal
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
