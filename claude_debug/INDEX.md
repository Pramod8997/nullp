# Master Index & Navigation: Claude Debug Package

> **Smart Energy Monitoring & Edge Safety System (EMS)**  
> **Target Folder:** `claude_debug/`  
> **Prepared for:** Claude (Opus 4.5 / 5 / Sonnet) & Senior Engineering Agents  
> **Current Baseline:** 467/467 Tests Passing (100%) | Real UK-DALE & REDD Data Integrated | Demo Models Trained

---

## Directory Structure & Navigational Map

All documents in this folder (`claude_debug/`) provide zero-gap, token-efficient, fully grounded context:

| Document | Path | Purpose |
| :--- | :--- | :--- |
| **1. Agent Guide** | [`CLAUDE.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/CLAUDE.md) | Master development playbook, baseline commands, FreeRTOS pinout, and token-economy rules. |
| **2. Master Prompt** | [`PROMPT.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/PROMPT.md) | Ultra-dense, token-efficient prompt for Claude Opus 5 with exact API contracts and 467-test baseline. |
| **3. Product Requirements** | [`PRD.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/PRD.md) | Product Requirements Document (Functional & Non-Functional specifications, 10-appliance + demo class sets). |
| **4. Technical Review** | [`TECHNICAL_REVIEW.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/TECHNICAL_REVIEW.md) | Deep system architecture, FreeRTOS state machines, real-data NILM pipeline, centroid fallback. |
| **5. API Cheatsheet** | [`ARCHITECTURE_AND_APIS.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/ARCHITECTURE_AND_APIS.md) | Exhaustive class/method signatures to prevent token-wasting API hallucinations. |
| **6. Hardware Audit** | [`HARDWARE_DEPLOYMENT_GUIDE.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/HARDWARE_DEPLOYMENT_GUIDE.md) | 6 physical hazards, MOSFET level shifting, RC snubber, 10A PZEM selection, PE earth wiring, and BOM. |
| **7. Hardware Readiness Checklist** | [`HARDWARE_READINESS_CHECKLIST.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/HARDWARE_READINESS_CHECKLIST.md) | Pre-procurement BOM order checklist, 3 schematic blocking fixes, staged bring-up gates. |
| **8. Real-World Physical Testing** | [`REAL_WORLD_TESTING_PLAN.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/REAL_WORLD_TESTING_PLAN.md) | 8 physical bench tests (Variac brownouts, inductive arcing, thermal rise, THD noise). |
| **9. Debug Verification Status** | [`debug_status.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/debug_status.md) | 467/467 regression verification, 7/7 physical stress, 10/10 HIL, and modified tracked files review. |

---

## Reference Knowledge Base & Pre-built Graphs

Before asking broad architectural questions, query the pre-indexed AST knowledge graph:
* **Knowledge Graph Directory:** `graphify-out/` (2,266 nodes, 4,625 edges, 164 communities)
* **CLI Query Tool:** `graphify query "<question>"`
* **CLI Path Tool:** `graphify path "<nodeA>" "<nodeB>"`
* **Architecture Report:** [`graphify-out/GRAPH_REPORT.md`](file:///home/pramodsb/Downloads/mjr/graphify-out/GRAPH_REPORT.md)

---

## Quick Command Cheatsheet

```bash
# Activate Virtual Environment
source .venv/bin/activate

# Run Entire 467-Test Regression Suite (100% Pass)
python -m pytest tests/ -q

# Run Real-Data & Heuristic Fallback Suite (34 Tests)
python -m pytest tests/test_real_data_and_ml_fallback.py -v

# Run Real-World Physical & Electrical Stress Harness
python scripts/real_world_physical_stress.py
python scripts/hil_hardware_test.py

# Run Core 5 Production Stress & Chaos Suites (209 Tests)
python -m pytest tests/test_hil_uart_corruption.py \
                 tests/test_relay_safety_boot_brownout.py \
                 tests/test_ml_nilm_math_stress.py \
                 tests/test_security_penetration.py \
                 tests/test_chaos_engineering.py -v --tb=short

# Run Full Software Demo (Broker + Demo Pipeline + API + Fleet)
make demo

# Update Code Graph after modifying files
graphify update .
```
