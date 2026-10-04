# ASTRA Decision Log

| Date | Decision | Basis | Consequence |
|---|---|---|---|
| 2026-10-04 | Treat live source/tests as truth over historical “close-out” documents | Current dirty tree and contradictory specs | Baseline must be rerun before fixes or release claims |
| 2026-10-04 | Preserve all pre-existing uncommitted changes | `git status` shows firmware, simulator, and test edits | No reset/checkout of user work |
| 2026-10-04 | Physical status remains `NOT RUN` | No mains/instrument evidence supplied in this execution | No `[P] PHYSICAL VERIFIED` status allowed |
| 2026-10-04 | Hardware contradiction is a release blocker | 250 W laptop/phone spec conflicts with projector profile/plan | Human hardware decision required before ceiling/safety changes |
| 2026-10-04 | Existing dirty PZEM watchdog is not accepted as H1 closure | It retains a `pzemEverValid`-gated dead-at-boot ON path | Reproduce invariant directly and fix only after failing test |
