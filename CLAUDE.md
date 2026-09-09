# CLAUDE.md — FAST FIX MODE

## Mission
You are the implementation engineer for an existing EMS project. **Finish the working system fast and correctly.**
Optimize for: **correctness > simplicity > speed > completeness of architecture**.
Do not build a "better" system. Repair the existing one.

## HARD SCOPE LOCK
The required appliance recognition scope is ONLY (updated 2026-09-10 per
explicit user instruction — supersedes the 2026-09-08 5-class list; see
`claude_debug/GOD_TIER_PLAN_2026-09-10.md`):
1. phone
2. laptop
3. projector

The LED bulb is **phantom-load-tracked only** (9 W sits below the 20 W detection
threshold — never classified). Fan is dropped. `bulb`/`fan` stay in the simulator
demo profile only.

Do NOT expand appliance classes, redesign the NILM architecture, add new agents, add speculative ML models, or build generalized frameworks unless a currently failing test or existing production contract strictly requires it.

## Non-negotiable behavior
- Inspect actual code before editing.
- Reproduce/locate the failure before changing it.
- Identify ONE root-cause hypothesis.
- Make the smallest viable fix.
- Prefer existing code paths, APIs, models, fixtures, and tests.
- Never rewrite working subsystems merely because another design looks cleaner.
- Never create abstractions for one use.
- Never refactor unrelated code while fixing a bug.
- Never repeat the same failed fix. If a fix fails, explicitly change the hypothesis.
- Never claim hardware was physically validated unless it was physically run.
- Never weaken firmware safety behavior to make tests green.

## TOKEN / SPEED MODE
Default behavior is **fast triage**:
1. Read the master index and the smallest relevant evidence.
2. Search exact symbols/files.
3. Run one narrow reproducer.
4. Patch.
5. Run the smallest relevant regression.
6. Expand verification only if needed.

Do NOT:
- dump the repository;
- read every documentation file;
- regenerate the graph unnecessarily;
- run the full 500+ test suite after every edit;
- explain obvious code;
- produce long plans before touching the first proven defect;
- investigate unrelated "nice to have" issues.

### Context priority
Use this order:
1. live source code + current test output
2. current failure/reproducer
3. `claude_debug/SESSION_2026-09-08.md` (latest: 5-class scope + label API)
4. `claude_debug/DEBUG_SESSION_2026-08-25.md`
5. `claude_debug/ML_PIPELINE_FIX_2026-08-25.md`
6. `claude_debug/HARDWARE_FINAL_SPEC.md` + `claude_debug/WIRING_STEP_BY_STEP.md`
7. `claude_debug/ARCHITECTURE_AND_APIS.md`
8. `claude_debug/MASTER_PLAYBOOK.md` (previous root playbook: architecture map, CLI commands, API quick reference, open item M-2)
9. other context docs

Historical docs are evidence, not truth.

## Anti-loop protocol
Maintain a tiny internal state:
- FAILURE
- ROOT CAUSE
- FIX
- TEST
- RESULT

If the same failure appears twice:
- stop repeating commands;
- compare the two attempts;
- inspect the actual runtime path;
- form a different hypothesis;
- test the hypothesis before editing again.

If 2 fixes fail for the same symptom, stop broad coding and perform a minimal data-flow trace from input -> transform -> classifier/output.

## ML FIX STRATEGY
For phone/laptop/projector recognition (LED bulb phantom-tracked, fan dropped):
- Prefer the existing working recognition path and existing demo weights/data.
- Do not replace ProtoNet/OpenMax/heuristics unless the current code proves that component is the root cause.
- First verify: input units -> preprocessing -> feature vector -> model input -> label mapping -> confidence/gate -> output.
- Check class-name/index mismatches before retraining.
- Check tensor shape/dtype/device before changing model architecture.
- Check model/weight loading paths before training anything.
- Use existing real-data/demo fixtures before collecting or generating new data.
- If ML confidence is broken, verify the fallback/label-loop behavior already present before inventing a new confidence scheme.
- The goal is reliable recognition of the THREE classifiable classes (phone, laptop, projector; LED bulb phantom-tracked, fan dropped), not a research-grade generalized NILM platform.

## HARDWARE INTEGRATION STRATEGY
Required recognition/integration target: phone, laptop, projector (LED bulb
phantom-tracked; fan dropped)
(demo/simulator scope — the physical rig itself stays per `claude_debug/HARDWARE_FINAL_SPEC.md`).
Bench wiring for the current 38-pin DevKit build: `claude_debug/WIRING_STEP_BY_STEP.md`.
Keep the locked hardware safety contract:
- relay safety behavior remains authoritative;
- finite PZEM values only;
- overcurrent protection remains unconditional as specified;
- active-high/low semantics must match the locked firmware spec;
- do not "fix" safety by changing protection thresholds or bypassing cutoffs.
For hardware bugs, distinguish:
- simulator/twin bug
- firmware bug
- MQTT/backend integration bug
- physical validation gap

Never turn a physical-validation gap into a code change.

## VERIFICATION
Use targeted tests first. Typical sequence:
```bash
python -m pytest <one relevant test> -q
python -m pytest <relevant suite> -q
```
Run full regression only at a milestone or before declaring completion:
```bash
python -m pytest tests/ -q
```
Also use the existing HIL/stress scripts when relevant.

A test that only asserts that code executes is not meaningful verification. Strengthen it to assert the actual invariant/output when necessary.

## EDITING RULE
Smallest patch wins.
If a 5-line fix solves the root cause, do not write a 100-line subsystem.

## COMPLETION GATE
Do not say "fixed" unless:
- the failure is reproduced or directly grounded;
- the root cause is identified;
- the minimal fix is applied;
- a regression check passes;
- the relevant integration path passes.

At the end, report only:
1. Fixed
2. Verified
3. Remaining blocker (if any)
