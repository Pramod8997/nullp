# Physical Validation Status

**Current status:** NOT RUN. No hardware is assembled. Mains authorization is
BLOCKED.

## Evidence still required

1. ESP32 board marking, actual pin count, connector and shield fit.
2. Relay part marking, H/L jumper, VCC/JD-VCC wiring, input current and COM/NO
   truth table.
3. PZEM part marking proving 10 A direct/shunt, plus measured TX idle voltage.
4. Low-voltage common-ground, 5 V rail polarity and rail-sag check.
5. Fuse, PE continuity, enclosure bond, creepage, strain relief, terminals,
   socket and RCBO/RCCB inspection.
6. Load nameplates and staged relay-cutoff test within the 250 W envelope.

Record photos, meter readings and test results in
[`../omnirush/PHYSICAL_VALIDATION.md`](../omnirush/PHYSICAL_VALIDATION.md).
Do not mark any gate passed from simulator output, source inspection or a
continuity check alone.
