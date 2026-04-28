PYTHON=.venv/bin/python3
PIP=.venv/bin/pip
PWD=$(shell pwd)

.PHONY: install generate_data train_synthetic train_real train_all evaluate test_safety test run clean

install:
	$(PIP) install -r requirements.txt
	cd frontend && npm install

generate_data:
	export PYTHONPATH=$(PWD) && $(PYTHON) scripts/generate_mock_ukdale.py

# ── Training ─────────────────────────────────────────────────────────────────

train_synthetic:
	@echo "▶ Episodic meta-training on SyntheticUKDALE (10 classes, 10k episodes)..."
	export PYTHONPATH=$(PWD) && $(PYTHON) scripts/train_models.py

train_real:
	@echo "▶ Downloading + training on real UK-DALE & REDD datasets..."
	export PYTHONPATH=$(PWD) && $(PYTHON) scripts/train_real_datasets.py

train_all:
	@echo "▶ Full training pipeline: Synthetic → Real augmentation → Evaluation..."
	export PYTHONPATH=$(PWD) && $(PYTHON) scripts/train_models.py
	export PYTHONPATH=$(PWD) && $(PYTHON) scripts/train_real_datasets.py

evaluate:
	@echo "▶ Generating evaluation results & benchmark report..."
	export PYTHONPATH=$(PWD) && $(PYTHON) scripts/evaluate.py

# ── Testing ───────────────────────────────────────────────────────────────────

test_safety:
	export PYTHONPATH=$(PWD) && $(PYTHON) scripts/test_safety_cutoff.py --spike 4000

test:
	export PYTHONPATH=$(PWD) && $(PYTHON) -m pytest tests/ -v --tb=short

# ── Run (fixed: explicit PID tracking avoids kill-0 segfault) ─────────────────

run:
	@echo "Starting Confidence-Aware Digital Twin EMS..."
	@export PYTHONPATH=$(PWD); \
	$(PYTHON) scripts/start_broker.py & BROKER_PID=$$!; \
	sleep 2; \
	$(PYTHON) scripts/run_pipeline.py & PIPELINE_PID=$$!; \
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & API_PID=$$!; \
	(cd frontend && npm run dev) & FRONTEND_PID=$$!; \
	$(PYTHON) backend/scripts/simulate_esp32.py & SIM_PID=$$!; \
	trap "kill $$BROKER_PID $$PIPELINE_PID $$API_PID $$FRONTEND_PID $$SIM_PID 2>/dev/null; exit 0" SIGINT SIGTERM; \
	wait $$BROKER_PID $$PIPELINE_PID $$API_PID $$FRONTEND_PID $$SIM_PID

clean:
	rm -f data/ems_state.db data/ems_state.db-shm data/ems_state.db-wal
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
