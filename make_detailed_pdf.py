import sys
from fpdf import FPDF
from fpdf.enums import XPos, YPos

class PDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 20)
        self.cell(0, 10, "Smart Home EMS - Project Overview", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(5)

    def chapter_title(self, title):
        self.set_font("helvetica", "B", 14)
        self.set_text_color(0, 51, 102)
        self.cell(0, 10, title, 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        self.ln(2)

    def chapter_body(self, body):
        self.set_font("helvetica", "", 11)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, body)
        self.ln(5)

pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

content = [
    ("1. config/ (System Configuration)",
     "- config.yaml: Central configuration file. Stores safety limits (e.g., 2500W limit), ML parameters (ProtoNet distance thresholds), Reinforcement Learning settings (cooldown periods, comfort bounds), and MQTT settings."),

    ("2. scripts/ (Execution & Utilities)",
     "- run_pipeline.py: The main orchestrator. It starts the 9-step processing pipeline (MQTT -> ML -> RL -> DB).\n"
     "- start_broker.py: Launches a local amqtt broker.\n"
     "- train_models.py: Offline training script for CNN, ProtoNet anchors, and Q-Table.\n"
     "- generate_mock_ukdale.py: Generates synthetic energy dataset (mock_ukdale.h5).\n"
     "- import_colab_weights.py: Utility to extract and load ML weights from Google Colab."),

    ("3. src/ (Core Backend & ML Pipeline)",
     "- api/main.py: FastAPI REST endpoints and WebSocket bridge for the frontend.\n"
     "- database/session.py: Asynchronous aiosqlite database logic with Write-Ahead Logging.\n"
     "- hardware/mqtt.py: Async MQTT client manager (aiomqtt).\n"
     "- models/protonet.py: CNN encoder and open-set classification logic.\n"
     "- models/thermodynamics.py: ISO 7730 Fanger PMV thermal comfort Digital Twin.\n"
     "- pipeline/safety.py: Real-time safety monitor (triggers emergency cutoffs).\n"
     "- pipeline/watchdog.py: Soft anomaly detector using z-scores.\n"
     "- pipeline/phantom_tracker.py: Tracks vampire loads when devices are off.\n"
     "- rl/agent.py: Tabular Q-Learning agent that optimizes energy vs. comfort."),

    ("4. backend/ (Simulation & Models)",
     "- scripts/simulate_esp32.py: Mocks 4 smart plugs generating 1Hz power readings.\n"
     "- models/weights/: Directory where trained ML models (.pt, .json, .pkl) are saved."),

    ("5. frontend/ (React Dashboard)",
     "A dark-mode Vite React application connecting to FastAPI WebSockets.\n"
     "- App.jsx / index.css: Main layout and design system.\n"
     "- components/DeviceCards.jsx: Live power state per device.\n"
     "- components/RealTimeChart.jsx: Power traces and safety thresholds.\n"
     "- components/DigitalTwin.jsx: Thermal comfort gauge and RL decisions."),

    ("6. notebooks/ & tests/",
     "- notebooks/colab_train.py: Notebook for training on real UK-DALE data via GPUs.\n"
     "- tests/test_pipeline.py: 45 integration tests verifying safety, ML, and API endpoints."),

    ("7. Root Level Files",
     "- Makefile: Command runner (make run, make test, make train_all).\n"
     "- README.md / requirements.txt: Documentation and dependencies.")
]

for title, body in content:
    pdf.chapter_title(title)
    pdf.chapter_body(body)

pdf.output("Project_Overview.pdf")
