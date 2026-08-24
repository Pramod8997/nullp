# Master Index & Navigation: Claude Context Package

> **Smart Energy Monitoring & Edge Safety System (EMS)**  
> **Prepared for:** Claude (Opus 4.5 / 5 / Sonnet) & Senior Engineering Agents  
> **Status:** 100% Simulation-Verified (433/433 Tests Passing) | Real-World Physical Bench Harness Ready

---

## Directory Structure & Navigational Map

All documents in this folder (`claude_context/`) are structured to provide zero-gap, token-efficient, fully grounded context for Claude Opus/Sonnet:

| Document | Path | Purpose |
| :--- | :--- | :--- |
| **1. Agent Guide** | [`CLAUDE.md`](file:///home/pramodsb/Downloads/mjr/claude_context/CLAUDE.md) | Master development playbook, commands, rules, conventions, and agent workflows. |
| **2. Master Prompt** | [`PROMPT.md`](file:///home/pramodsb/Downloads/mjr/claude_context/PROMPT.md) | Ultra-dense, token-efficient master prompt for zero-hallucination agent execution. |
| **3. Product Requirements** | [`PRD.md`](file:///home/pramodsb/Downloads/mjr/claude_context/PRD.md) | Complete Product Requirements Document (Functional & Non-Functional specifications). |
| **4. Technical Review** | [`TECHNICAL_REVIEW.md`](file:///home/pramodsb/Downloads/mjr/claude_context/TECHNICAL_REVIEW.md) | Deep system architecture, FreeRTOS state machines, pipeline stages, math formulas. |
| **5. API Cheatsheet** | [`ARCHITECTURE_AND_APIS.md`](file:///home/pramodsb/Downloads/mjr/claude_context/ARCHITECTURE_AND_APIS.md) | Exhaustive class/method signatures to prevent token-wasting API hallucinations. |
| **6. Hardware Audit** | [`HARDWARE_DEPLOYMENT_GUIDE.md`](file:///home/pramodsb/Downloads/mjr/claude_context/HARDWARE_DEPLOYMENT_GUIDE.md) | 6 real-world electrical hazards, level shifters, snubbers, PSU sizing, and BOM. |
| **7. Real-World Physical Testing** | [`REAL_WORLD_TESTING_PLAN.md`](file:///home/pramodsb/Downloads/mjr/claude_context/REAL_WORLD_TESTING_PLAN.md) | 8 physical bench tests (Variac brownouts, inductive arcing, thermal rise, THD noise). |

---

## Reference Knowledge Base & Pre-built Graphs

Before asking broad architectural questions, query the pre-indexed AST knowledge graph:
* **Knowledge Graph Directory:** `graphify-out/` (1,910 nodes, 4,085 edges, 146 communities)
* **CLI Query Tool:** `graphify query "<question>"`
* **CLI Path Tool:** `graphify path "<nodeA>" "<nodeB>"`
* **Architecture Report:** [`graphify-out/GRAPH_REPORT.md`](file:///home/pramodsb/Downloads/mjr/graphify-out/GRAPH_REPORT.md)

---

## Quick Command Cheatsheet

```bash
# Activate Virtual Environment
source .venv/bin/activate

# Run Entire 433-Test Regression Suite
python -m pytest tests/ -q

# Run Real-World Physical & Electrical Stress Harness
python scripts/real_world_physical_stress.py
python scripts/hil_hardware_test.py

# Run 5 New Production Stress & Chaos Suites (209 Tests)
python -m pytest tests/test_hil_uart_corruption.py \
                 tests/test_relay_safety_boot_brownout.py \
                 tests/test_ml_nilm_math_stress.py \
                 tests/test_security_penetration.py \
                 tests/test_chaos_engineering.py -v --tb=short

# Update Code Graph after modifying files
graphify update .
```
