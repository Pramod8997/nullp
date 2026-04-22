.PHONY: install train_all test_safety run

install:
	pip install -r requirements.txt
	cd frontend && npm install

train_all:
	python scripts/train_models.py --datasets ukdale redd synd --cuda

test_safety:
	python scripts/test_safety_cutoff.py --spike 4000

run:
	@echo "Starting Confidence-Aware Digital Twin EMS..."
	python scripts/run_pipeline.py & \
	cd frontend && npm run dev & \
	python scripts/esp32_simulator.py & \
	wait
