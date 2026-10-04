# ASTRA Omnirush Engineering Changelog

## CP-000 baseline — 2026-10-04

- Created the persistent release-gate/checkpoint workspace.
- Recorded the dirty working tree and historical-document contradictions.
- Defined the baseline test commands and H1–H5/B1/parser/ML/dashboard reproduction matrix.
- Marked physical validation `NOT RUN`; no source remediation performed in this baseline phase.
- Ran the matching-runtime baseline: 641 Python tests passed with 3 warnings; 20 frontend tests passed.
- Added four failing audit reproductions; all fail against the current implementation as expected.

## CP-001 safety/parser — 2026-10-04

- Core 0 is the sole runtime firmware/twin relay owner; MQTT ON/OFF requests are queued and consumed after safety evaluation.
- Production ON fails closed when PZEM health is unavailable and cannot win a safety-trip/Core-1-lockout interval.
- Live pipeline parsing rejects empty, missing, non-finite, negative, and nonnumeric values before state mutation.
- API MQTT frame decoding/validation failures are isolated per frame.
- Verification: focused **163 passed**; full Python **646 passed, 3 warnings**; frontend **20 passed**.
- Firmware compilation and physical gates: **NOT RUN**.
