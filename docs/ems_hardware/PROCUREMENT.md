# Current Procurement

The complete itemized list, links, prices and GST arithmetic is maintained at
[`../../reports/ems_procurement_bom_under_2500.txt`](../../reports/ems_procurement_bom_under_2500.txt).

## Required order gates

- Available meter is recorded as PZEM-004T-100A(V4.0)-type external CT. Confirm
  the exact marking/manual, CT terminal pair, `L/N` sense terminals, CT
  polarity and UART behavior before any mains work. Do not substitute the old
  10 A shunt wiring.
- 38-pin breakout physically matches the owned ESP32 board; no 30-pin shield.
- The available relay board may be 4-channel. Use channel 1 only; verify
  H/L trigger, VCC/JD-VCC arrangement, input/coil current and COM/NO labels. A
  generic “30 A” label is not enough.
- The four-channel level shifter is not part of the approved wiring; leave it
  disconnected unless a separately reviewed relay interface requires it.
- PZEM TX voltage is measured before GPIO16 is connected.
- Final runtime power uses an isolated/BIS-marked 5 V 2 A supply and a
  USB-A-female-to-screw-terminal breakout. The breakout is not in the current
  priced subtotal and must be added before final assembly.
- Checkout shipping, tax and stock remain within the budget allowance.

The listed goods subtotal is ₹1,790.30 including GST before shipping. This is
not a complete mains-build price: enclosure, fuse/holder, PE hardware, RCBO/
RCCB, mains-rated socket/terminals, conductors and the runtime DC breakout are
outside that subtotal.
