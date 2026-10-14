# Mains Safety Gates

This file is a release gate, not permission to build or energize mains. A
qualified person must inspect the finished enclosure and wiring.

## Fixed load path

```text
Live → 5 A ceramic fuse → PZEM AC in/out → relay COM → relay NO → socket Live
Neutral → PZEM Neutral and socket Neutral
PE → bonded enclosure / PE barrier → socket PE (never switched)
```

Required physical controls include a flame-retardant enclosure, mains-rated
terminals and socket, 1.0 mm² mains conductors with strain relief, PE
continuity, correctly rated fuse holder, and upstream 30 mA RCBO/RCCB.

## Release gates

- Low-voltage boot test passes with relay GPIO18 LOW through reset.
- PZEM TX voltage and level-shifting path are measured and recorded.
- Relay truth table is verified with a multimeter, not by sound.
- USB/mains isolation rule is followed.
- Fuse, PE, enclosure, creepage, terminal torque and RCBO/RCCB are inspected.
- Actual load nameplates remain within the 250 W prototype envelope.

Until every gate passes, keep the PZEM AC terminals, relay contacts and socket
de-energized. Software tests and simulator output do not validate insulation,
earthing, creepage, contact spacing or fuse interruption.
