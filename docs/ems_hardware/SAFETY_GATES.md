# Mains Safety Gates

This file is a release gate, not permission to build or energize mains. A
qualified person must inspect the finished enclosure and wiring.

## Fixed load path

```text
Live → 5 A ceramic fuse → branch to PZEM L sense and relay COM
      → relay NO → CT around live conductor → socket Live
Neutral → PZEM N sense and socket Neutral
PE → bonded enclosure / PE barrier → socket PE (never switched)
```

This is the topology for the physically available **PZEM-004T-100A external-CT
variant**. The old `PZEM in-L → PZEM out-L` shunt path applies only to a 10 A
direct-connect unit and is not permitted for this build.

Required physical controls include a flame-retardant enclosure, mains-rated
terminals and socket, 1.0 mm² mains conductors with strain relief, PE
continuity, correctly rated fuse holder, and upstream 30 mA RCBO/RCCB.

## Release gates

- Low-voltage boot test passes with relay GPIO18 LOW through reset.
- PZEM TX voltage and level-shifting path are measured and recorded.
- PZEM model, CT terminal pair, CT polarity/orientation and `L/N` sense labels
  are checked against the exact manual.
- Relay truth table is verified with a multimeter, not by sound.
- USB/mains isolation rule is followed.
- Fuse, PE, enclosure, creepage, terminal torque and RCBO/RCCB are inspected.
- Actual load nameplates remain within the 250 W prototype envelope; projector
  is software-only and is not authorized on this socket.

Until every gate passes, keep the PZEM AC terminals, relay contacts and socket
de-energized. Software tests and simulator output do not validate insulation,
earthing, creepage, contact spacing or fuse interruption.
