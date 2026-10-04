# ASTRA Assumptions Register

| ID | Assumption | Evidence / owner | Status |
|---|---|---|---|
| A-001 | The actual physical rig may not be connected in this environment | no physical artifacts or instrument logs supplied | UNVERIFIED |
| A-002 | `claude_debug/HARDWARE_FINAL_SPEC.md` is intended to be authoritative, but its projector exclusion conflicts with current scope | document header versus config/plan | BLOCKED pending human decision |
| A-003 | Existing dirty working-tree changes belong to prior user work | git status/diff | ACCEPTED for preservation |
| A-004 | Simulator/HIL harnesses do not constitute physical validation | project audit and user instruction | ACCEPTED |
| A-005 | Current model/registry artifacts may be demo or synthetic unless provenance proves physical enrollment | hardware profile has no registry path | UNVERIFIED |
| A-006 | The installed PZEM library timeout behavior must be measured from source/runtime, not inferred from comments | audit estimate only | OPEN |
