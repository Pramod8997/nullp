# Master Index & Navigation: Claude Debug Package

> **Smart Energy Monitoring & Edge Safety System (EMS)**  
> **Target Folder:** `claude_debug/`  
> **Prepared for:** Claude (Opus 4.5 / 5 / Sonnet) & Senior Engineering Agents  
> **Current Baseline:** 626/626 Tests Passing (100%) | 20/20 Vitest Passing | Real UK-DALE & REDD Data Integrated | 3-Class Physical Scope (phone/laptop/projector; LED bulb phantom-tracked, fan dropped)

---

## Directory Structure & Navigational Map

All documents in this folder (`claude_debug/`) provide zero-gap, token-efficient, fully grounded context:

| Document | Path | Purpose |
| :--- | :--- | :--- |
| **1. Agent Guide** | [`CLAUDE.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/CLAUDE.md) | Master development playbook, baseline commands, FreeRTOS pinout, and token-economy rules. |
| **2. Master Prompt** | [`PROMPT.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/PROMPT.md) | Ultra-dense, token-efficient prompt for Claude Opus 5 with exact API contracts. **Note:** its stated 467-test baseline is historic — the live baseline is 626. |
| **3. Product Requirements** | [`PRD.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/PRD.md) | Product Requirements Document (Functional & Non-Functional specifications, 10-appliance + demo class sets). |
| **4. Technical Review** | [`TECHNICAL_REVIEW.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/TECHNICAL_REVIEW.md) | Deep system architecture, FreeRTOS state machines, real-data NILM pipeline, centroid fallback. |
| **5. API Cheatsheet** | [`ARCHITECTURE_AND_APIS.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/ARCHITECTURE_AND_APIS.md) | Exhaustive class/method signatures to prevent token-wasting API hallucinations. |
| **6. 🔒 FINAL Hardware Spec** | [`HARDWARE_FINAL_SPEC.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/HARDWARE_FINAL_SPEC.md) | **AUTHORITATIVE.** Locked physical build: single aggregate node, ~600 W consumer electronics, India 230 V. Final BOM, coordination ladder, bring-up gates, B-0…B-7 ledger. |
| **7. Hazard Analysis** | [`HARDWARE_DEPLOYMENT_GUIDE.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/HARDWARE_DEPLOYMENT_GUIDE.md) | 6 physical hazard root causes: MOSFET level shifting, RC snubber, brownout, PZEM refresh latency, creepage, inverter dynamics. **BOM superseded by the spec.** |
| **8. Readiness Checklist** | [`HARDWARE_READINESS_CHECKLIST.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/HARDWARE_READINESS_CHECKLIST.md) | Original pre-procurement review and blocking-issue analysis. **All issues resolved; order tables superseded by the spec.** |
| **9. Real-World Physical Testing** | [`REAL_WORLD_TESTING_PLAN.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/REAL_WORLD_TESTING_PLAN.md) | 8 physical bench tests (Variac brownouts, inductive arcing, thermal rise, THD noise). |
| **10. Debug Verification Status** | [`debug_status.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/debug_status.md) | 626/626 regression verification, 7/7 physical stress, 10/10 HIL. §0 = fixes from the 2026-08-25 hardware/NILM pass, §0b = ML recognition, §0c = five-class/label-API, §0d = 2026-09-10 hardware alignment, overlap delta & close-out. |
| **11. Debug Session Log (2026-08-25)** | [`DEBUG_SESSION_2026-08-25.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/DEBUG_SESSION_2026-08-25.md) | ML & hardware-integration debug pass: 6 defects (H-1…H-3, M-1/M-3/M-4) with root causes and `file:line` proofs. |
| **12. ML Pipeline Fix Log (2026-08-25)** | [`ML_PIPELINE_FIX_2026-08-25.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/ML_PIPELINE_FIX_2026-08-25.md) | The recognition rewire: M-5…M-8, measured evidence base, working label/enrollment loop, OpenMax resolution. |
| **13. Debug Session Log (2026-09-08)** | [`SESSION_2026-09-08.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/SESSION_2026-09-08.md) | 5-class scope pass (phone/laptop/bulb/projector/fan), `label-unrecognized` API, e2e open-set loop, 549-test baseline. |
| **14. Wiring Guide** | [`WIRING_STEP_BY_STEP.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/WIRING_STEP_BY_STEP.md) | 🔧 Beginner pin-to-pin bench wiring for 38-pin DevKit: jumper counts, B-4 level check, 100 kΩ pull-down, relay jumper, D13 rule. |
| **15. God-Tier Execution Plan (2026-09-10)** | [`GOD_TIER_PLAN_2026-09-10.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/GOD_TIER_PLAN_2026-09-10.md) | Master 3-wave execution plan, scope re-lock (phone, laptop, projector; LED bulb phantom-tracked, fan dropped), gate criteria. |
| **16. Hardware Alignment Contract** | [`HARDWARE_ALIGNMENT_CONTRACT.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/HARDWARE_ALIGNMENT_CONTRACT.md) | Formal hardware and protocol parity contract between ESP32 firmware (`main.cpp`), Digital Twin (`esp32_firmware_sim.py`), Backend API, and HIL. |
| **17. Debug Session Log (2026-09-10)** | [`SESSION_2026-09-10.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/SESSION_2026-09-10.md) | God-tier execution close-out narrative: twin parity, delta overlap NILM, 100% safety branch coverage, ACL hardening, 626 tests. |
| **18. Verification Ledger (2026-09-10)** | [`VERIFICATION_LEDGER_2026-09-10.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/VERIFICATION_LEDGER_2026-09-10.md) | Machine-checkable ledger of all verification gates, live command runs, test matrices, and proof outputs. |

---

## Reference Knowledge Base & Pre-built Graphs

Before asking broad architectural questions, query the pre-indexed AST knowledge graph:
* **Knowledge Graph Directory:** `graphify-out/` (2,783 nodes, 5,350 edges, 176 communities; rebuilt 2026-09-10)
* **CLI Query Tool:** `graphify query "<question>"`
* **CLI Path Tool:** `graphify path "<nodeA>" "<nodeB>"`
* **Architecture Report:** [`graphify-out/GRAPH_REPORT.md`](file:///home/pramodsb/Downloads/mjr/graphify-out/GRAPH_REPORT.md)

---

## Quick Command Cheatsheet

```bash
# Activate Virtual Environment
source .venv/bin/activate

# Run Entire 626-Test Regression Suite (100% Pass)
python -m pytest tests/ -q

# Run Hardware-Firmware-Twin Parity Suite (14 Tests)
python -m pytest tests/test_hardware_alignment.py -v

# Run Delta Overlap & Detector Path E2E Suites
python -m pytest tests/test_overlap_delta.py tests/test_detector_path_e2e.py -v

# Run Frontend Vitest Suite (20/20 Pass)
cd frontend && npm test -- --run

# Run Real-World Physical & Electrical Stress Harness
python scripts/real_world_physical_stress.py
python scripts/hil_hardware_test.py
python scripts/test_firmware_and_ai_e2e.py

# Run Full Software Demo (Broker + Demo Pipeline + API + Fleet)
make demo

# Update Code Graph after modifying files
graphify update .
```
