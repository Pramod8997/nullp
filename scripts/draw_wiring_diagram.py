#!/usr/bin/env python3
"""
Draw the beginner wiring diagrams for the locked EMS hardware build.

Every pin, part and wire comes from claude_debug/HARDWARE_FINAL_SPEC.md
(decisions D1-D13) and firmware/esp32_node/src/main.cpp (pin constants).
Nothing here is invented. Regenerate with:

    python scripts/draw_wiring_diagram.py

Outputs into claude_debug/:
    wiring_1_lowvoltage.png   build and test this first, no mains
    wiring_2_mains.png        230 V - have an electrician check it
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "claude_debug")

LIVE = "#6b3410"      # brown        - line / live
NEUT = "#1f5fa8"      # blue         - neutral
EARTH = "#1f8a3c"     # green/yellow - protective earth
V5 = "#d62728"        # red          - +5 V
GND = "#222222"       # black        - 0 V / GND
SIG = "#e07b00"       # orange       - logic signal
BOX = "#f7f7f9"
EDGE = "#444444"


def dev(ax, x, y, w, h, title, fc=BOX, ec=EDGE, title_size=12, lw=1.8,
        sub=None):
    """Device outline with a centred title at the top and optional subtitle."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h - 0.26, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", zorder=3)
    if sub:
        n = title.count("\n") + 1
        ax.text(x + w / 2, y + h - 0.26 - n * 0.36, sub, ha="center", va="top",
                fontsize=9.2, color="#555555", zorder=3)


def pin(ax, x, y, label, side="right", color=EDGE, size=10.0, pad=0.16):
    """A terminal dot ON the box edge, labelled INSIDE the box."""
    ax.plot([x], [y], marker="o", ms=8.0, mfc="white", mec=color, mew=1.9,
            zorder=4)
    if side == "right":                      # wire leaves rightwards
        ax.text(x - pad, y, label, ha="right", va="center", fontsize=size,
                family="monospace", zorder=4)
    else:                                    # wire leaves leftwards
        ax.text(x + pad, y, label, ha="left", va="center", fontsize=size,
                family="monospace", zorder=4)


def wire(ax, pts, color, lw=2.8, ls="-", dots=True):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw,
            ls=ls, zorder=1, solid_capstyle="round")
    if dots:
        for p in pts[1:-1]:
            ax.plot([p[0]], [p[1]], marker="o", ms=3.6, color=color, zorder=2)


def wlabel(ax, x, y, text, color, size=9.2, ha="center", va="bottom"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color,
            fontweight="bold", zorder=6,
            bbox=dict(fc="white", ec="none", alpha=0.94, pad=1.8))


def junction(ax, x, y, color):
    ax.plot([x], [y], marker="o", ms=7.5, color=color, zorder=5)


def callout(ax, x, y, w, h, title, body, fc="#fff4d6", ec="#c47f00",
            tc="#b00020", title_size=10.6, body_size=9.3):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.9, zorder=4))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=tc, zorder=5)
    ax.text(x + w / 2, y + h - 0.16 - 0.40, body, ha="center", va="top",
            fontsize=body_size, color=tc, zorder=5)


def legend(ax, x, y, entries, size=9.4, dy=0.34):
    for i, (c, t) in enumerate(entries):
        ax.plot([x, x + 0.55], [y - i * dy, y - i * dy], color=c, lw=3.6,
                solid_capstyle="round")
        ax.text(x + 0.72, y - i * dy, t, va="center", fontsize=size)


# ══════════════════════════════════════════════════════════════════════
#  DIAGRAM 1 — LOW VOLTAGE. Build this FIRST. No mains.
# ══════════════════════════════════════════════════════════════════════
def diagram_lowvoltage():
    W, H = 18.0, 11.0
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    ax.text(W / 2, H - 0.16, "DIAGRAM 1 of 2   —   LOW-VOLTAGE WIRING (5 V / 3.3 V)",
            ha="center", va="top", fontsize=18.5, fontweight="bold")
    ax.text(W / 2, H - 0.60,
            "Build and test this ENTIRE page first, with NO mains connected anywhere. "
            "Nothing on this page can hurt you.",
            ha="center", va="top", fontsize=12, color="#1f8a3c", fontweight="bold")

    ax.add_patch(Rectangle((0.5, 9.40), W - 1.0, 0.74, fc="#fff4d6",
                           ec="#c47f00", lw=1.6, zorder=4))
    ax.text(W / 2, 10.04,
            "DO THIS BEFORE connecting PZEM TX to GPIO16  (spec open item B-4):   "
            "power the PZEM from 5 V, leave TX unconnected, measure TX-to-GND on your multimeter.",
            ha="center", va="top", fontsize=10.3, zorder=5)
    ax.text(W / 2, 9.75,
            "~3.3 V or floating  ->  wire it straight to GPIO16.          "
            "~5 V  ->  put a 1 kΩ / 2 kΩ divider on that one wire  (₹6 — order them with the 100 kΩ).",
            ha="center", va="top", fontsize=10.3, zorder=5)

    # ── ESP32 ─────────────────────────────────────────────────────────
    dev(ax, 0.80, 4.60, 4.20, 4.35, "ESP32 DevKit V1",
        sub="WROOM-32D, 30-pin\n+ screw-terminal shield", fc="#eef4fb")
    pin(ax, 5.00, 8.50, "GPIO16", "right", SIG)
    pin(ax, 5.00, 8.10, "GPIO17", "right", SIG)
    pin(ax, 5.00, 6.00, "GPIO18", "right", SIG)
    pin(ax, 5.00, 5.30, "5V", "right", V5)
    pin(ax, 5.00, 4.90, "GND", "right", GND)
    ax.text(2.90, 4.40,
            "Must be WROOM-32D, never WROVER — a WROVER\n"
            "uses GPIO16/17 for its own RAM and metering dies.",
            ha="center", va="top", fontsize=9.0, color="#b00020",
            fontweight="bold", zorder=3)

    # ── PZEM ──────────────────────────────────────────────────────────
    dev(ax, 8.40, 7.15, 5.00, 1.85, "PZEM-004T v3.0",
        sub="10 A direct-connect · TTL 4-pin header", fc="#f3eefb")
    pin(ax, 8.40, 8.68, "RX", "left", SIG)
    pin(ax, 8.40, 8.28, "TX", "left", SIG)
    pin(ax, 13.40, 7.80, "5V", "right", V5)
    pin(ax, 13.40, 7.45, "GND", "right", GND)

    # ── Relay ─────────────────────────────────────────────────────────
    dev(ax, 8.40, 4.60, 5.00, 1.95, "2-channel Relay Module",
        sub="SRD-05VDC-SL-C · opto-isolated", fc="#fbf3ee")
    pin(ax, 8.40, 6.05, "IN1", "left", SIG)
    pin(ax, 13.40, 5.40, "VCC", "right", V5)
    pin(ax, 13.40, 5.00, "GND", "right", GND)
    ax.text(10.90, 4.80, "set the JUMPER to  H  (high trigger)",
            ha="center", va="bottom", fontsize=9.6, fontweight="bold",
            color="#b00020", zorder=4)

    # ── UART cross-over, drawn crossing ───────────────────────────────
    wire(ax, [(5.00, 8.50), (6.30, 8.50), (6.30, 8.28), (8.40, 8.28)], SIG)
    wire(ax, [(5.00, 8.10), (5.85, 8.10), (5.85, 8.68), (8.40, 8.68)], SIG)
    wlabel(ax, 7.30, 8.80, "GPIO17  ---->  PZEM RX", SIG)
    wlabel(ax, 7.30, 8.02, "GPIO16  <----  PZEM TX", SIG)

    callout(ax, 5.28, 6.72, 3.00, 1.05, "THE #1 BEGINNER MISTAKE",
            "TX goes to RX — the two\nwires CROSS. Wired straight\nacross, you get 0 W for ever.",
            title_size=9.8, body_size=9.0)

    # ── GPIO18 -> IN1, and the mandatory pull-down ────────────────────
    wire(ax, [(5.00, 6.00), (8.40, 6.05)], SIG)
    wlabel(ax, 6.85, 6.10, "GPIO18  ---->  IN1", SIG)
    junction(ax, 6.40, 6.02, SIG)
    wire(ax, [(6.40, 6.02), (6.40, 3.10)], GND, lw=2.4)
    ax.add_patch(Rectangle((6.12, 4.35), 0.56, 0.72, fc="#ffffff", ec=GND,
                           lw=2.0, zorder=5))
    ax.text(6.40, 4.71, "100k", ha="center", va="center", fontsize=7.8,
            rotation=90, zorder=6, family="monospace")
    ax.text(7.05, 4.22,
            "100 kΩ from IN1 down to 0 V — MANDATORY.\n"
            "It is what holds the relay OPEN while the ESP32 boots.\n"
            "Push the resistor legs straight into the screw terminals.",
            ha="center", va="top", fontsize=9.0, color="#b00020",
            fontweight="bold", zorder=6,
            bbox=dict(fc="white", ec="none", alpha=0.94, pad=2.2))

    # ── distribution + charger ────────────────────────────────────────
    dev(ax, 6.00, 1.85, 5.40, 1.25, "5 V / 0 V distribution",
        sub="screw terminal block", fc="#f0f0f0", title_size=11)
    dev(ax, 0.80, 0.45, 4.60, 1.05, "5 V 2 A BIS USB charger",
        sub="+ USB screw-terminal breakout", fc="#fdeeee", title_size=11)
    ax.text(3.10, 0.30, "outside the enclosure, its own wall socket",
            ha="center", va="top", fontsize=8.8, color="#555555")

    wire(ax, [(5.40, 0.97), (8.70, 0.97), (8.70, 1.85)], V5)
    wlabel(ax, 7.05, 1.07, "5 V in", V5)

    # to ESP32 (left and down)
    wire(ax, [(6.00, 2.72), (5.62, 2.72), (5.62, 5.30), (5.00, 5.30)], V5)
    wire(ax, [(6.00, 2.28), (5.28, 2.28), (5.28, 4.90), (5.00, 4.90)], GND)
    wlabel(ax, 5.62, 3.62, "+5 V", V5)
    wlabel(ax, 5.28, 3.10, "0 V", GND)

    # to Relay (up the right-hand side of the distribution block)
    wire(ax, [(11.40, 2.72), (14.20, 2.72), (14.20, 5.40), (13.40, 5.40)], V5)
    wire(ax, [(11.40, 2.28), (14.95, 2.28), (14.95, 5.00), (13.40, 5.00)], GND)
    # to PZEM (continue up the same two verticals)
    wire(ax, [(14.20, 5.40), (14.20, 7.80), (13.40, 7.80)], V5)
    wire(ax, [(14.95, 5.00), (14.95, 7.45), (13.40, 7.45)], GND)
    junction(ax, 14.20, 5.40, V5)
    junction(ax, 14.95, 5.00, GND)
    wlabel(ax, 14.20, 6.75, "+5 V", V5)
    wlabel(ax, 14.95, 6.30, "0 V", GND)
    ax.text(14.60, 2.05, "one 5 V rail and one shared 0 V feed all three boards",
            ha="left", va="center", fontsize=8.8, color="#555555", zorder=5,
            bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.6))

    legend(ax, 15.55, 9.00, [(V5, "+5 V"), (GND, "0 V / GND"),
                             (SIG, "logic signal, 3.3 V")])

    ax.add_patch(Rectangle((15.45, 3.30), 2.25, 3.10, fc="#ffe3e3",
                           ec="#b00020", lw=1.9, zorder=3))
    ax.text(16.57, 6.26, "RULE D13", ha="center", va="top", fontsize=11,
            fontweight="bold", color="#b00020", zorder=4)
    ax.text(16.57, 5.86,
            "USB and mains\nare NEVER both\nconnected.\n\n"
            "Flashing:\nUSB in, mains\nplug OUT of\nthe wall.\n\n"
            "Running:\nUSB out, 5 V\ncharger only.",
            ha="center", va="top", fontsize=9.0, color="#b00020", zorder=4)

    p = os.path.join(OUT_DIR, "wiring_1_lowvoltage.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# ══════════════════════════════════════════════════════════════════════
#  DIAGRAM 2 — MAINS. 230 V.
# ══════════════════════════════════════════════════════════════════════
def diagram_mains():
    W, H = 18.0, 10.0
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    ax.text(W / 2, H - 0.16, "DIAGRAM 2 of 2   —   230 V MAINS WIRING",
            ha="center", va="top", fontsize=18.5, fontweight="bold")
    ax.add_patch(Rectangle((0.5, 8.60), W - 1.0, 0.80, fc="#ffe3e3",
                           ec="#b00020", lw=2.4, zorder=4))
    ax.text(W / 2, 9.31,
            "DANGER  —  230 V CAN KILL YOU.   Have a qualified electrician make or check every joint on this page.",
            ha="center", va="top", fontsize=11.6, fontweight="bold",
            color="#b00020", zorder=5)
    ax.text(W / 2, 8.98,
            "Wire it with the plug OUT of the wall.        Load order is FIXED:  "
            "fuse -> PZEM -> relay -> socket.        Earth is NEVER switched.",
            ha="center", va="top", fontsize=10.5, zorder=5)

    y_l, y_n, y_pe = 6.95, 4.70, 3.10

    ax.add_patch(Rectangle((3.45, 2.35), 9.85, 5.65, fc="none", ec="#888888",
                           lw=2.0, ls="--", zorder=1))
    ax.text(8.37, 7.94, "EMS NODE  —  inside the UL94-V0 enclosure",
            ha="center", va="top", fontsize=10.6, color="#666666",
            fontweight="bold")

    # ── wall inlet ────────────────────────────────────────────────────
    dev(ax, 0.40, 2.75, 2.55, 4.70, "Wall socket",
        sub="on a 30 mA RCD", fc="#eef4fb", title_size=11.5)
    pin(ax, 2.95, y_l, "L", "right", LIVE)
    pin(ax, 2.95, y_n, "N", "right", NEUT)
    pin(ax, 2.95, y_pe, "PE", "right", EARTH)
    ax.text(1.67, 5.95,
            "6 A moulded plug\n+ 1.0 mm² flex\n\nDo NOT wire\nyour own plug top.",
            ha="center", va="top", fontsize=9.2, zorder=4)

    # ── fuse ──────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((3.80, y_l - 0.34), 1.75, 0.68,
                                boxstyle="round,pad=0.04", fc="#fff4d6",
                                ec=EDGE, lw=1.8, zorder=3))
    ax.text(4.67, y_l, "5 A ceramic", ha="center", va="center", fontsize=9.4,
            zorder=4, fontweight="bold")
    ax.text(4.67, y_l - 0.50, "fuse — never 2 A, never 16 A", ha="center",
            va="top", fontsize=8.5, color="#555555", zorder=4)

    # ── PZEM ──────────────────────────────────────────────────────────
    dev(ax, 6.25, 5.90, 3.10, 1.85, "PZEM-004T v3.0", fc="#f3eefb",
        title_size=11.5)
    pin(ax, 6.25, y_l, "L in", "left", LIVE)
    pin(ax, 9.35, y_l, "L out", "right", LIVE)
    pin(ax, 7.80, 5.90, "N", "left", NEUT)
    ax.text(7.80, 6.42, "load current flows\nTHROUGH its terminals",
            ha="center", va="center", fontsize=8.6, color="#555555", zorder=4)

    # ── relay ─────────────────────────────────────────────────────────
    dev(ax, 10.40, 5.90, 2.85, 1.85, "Relay ch.1", fc="#fbf3ee",
        title_size=11.5)
    pin(ax, 10.40, y_l, "COM", "left", LIVE)
    pin(ax, 13.25, y_l, "NO", "right", LIVE)
    ax.text(11.82, 6.42, "NC unused", ha="center", va="center", fontsize=8.6,
            color="#555555", zorder=4)

    # ── socket + loads ────────────────────────────────────────────────
    dev(ax, 14.30, 4.15, 3.25, 3.35, "IS 1293 6 A socket",
        sub="then a 3-pin\nEARTHED multi-plug", fc="#eef4fb", title_size=11.5)
    pin(ax, 14.30, y_l, "L", "left", LIVE)
    pin(ax, 14.30, y_n, "N", "left", NEUT)
    pin(ax, 14.30, y_pe, "PE", "left", EARTH)
    dev(ax, 14.30, 1.30, 3.25, 1.55, "The loads — all at once",
        sub="laptop brick · phone charger\n100 W lamp", fc="#f0f0f0",
        title_size=10.5)
    wire(ax, [(15.92, 4.15), (15.92, 2.85)], "#999999", lw=1.9, ls=":")

    # ── LIVE ──────────────────────────────────────────────────────────
    wire(ax, [(2.95, y_l), (3.80, y_l)], LIVE)
    wire(ax, [(5.55, y_l), (6.25, y_l)], LIVE)
    wire(ax, [(9.35, y_l), (10.40, y_l)], LIVE)
    wire(ax, [(13.25, y_l), (14.30, y_l)], LIVE)
    wlabel(ax, 9.87, y_l + 0.15, "L out -> COM", LIVE)
    wlabel(ax, 13.77, y_l + 0.15, "NO -> L", LIVE)

    # ── NEUTRAL ───────────────────────────────────────────────────────
    wire(ax, [(2.95, y_n), (14.30, y_n)], NEUT)
    wlabel(ax, 11.60, y_n + 0.15, "N  —  straight through, never switched", NEUT)
    junction(ax, 7.80, y_n, NEUT)
    wire(ax, [(7.80, y_n), (7.80, 5.90)], NEUT, lw=2.3)
    wlabel(ax, 7.80, 5.24, "PZEM taps N to sense voltage", NEUT, size=8.8)

    # ── EARTH ─────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((6.45, y_pe - 0.35), 2.70, 0.70,
                                boxstyle="round,pad=0.04", fc="#e8f6ea",
                                ec=EARTH, lw=2.0, zorder=3))
    ax.text(7.80, y_pe, "PE earth block", ha="center", va="center",
            fontsize=9.6, zorder=4, fontweight="bold")
    wire(ax, [(2.95, y_pe), (6.45, y_pe)], EARTH)
    wire(ax, [(9.15, y_pe), (14.30, y_pe)], EARTH)
    wlabel(ax, 4.70, y_pe + 0.15, "PE", EARTH)
    wlabel(ax, 11.70, y_pe + 0.15, "PE to socket  —  UNSWITCHED", EARTH)
    wire(ax, [(7.80, y_pe - 0.35), (7.80, 2.60)], EARTH, lw=2.3)
    wlabel(ax, 7.80, 2.40, "bonded to the enclosure body", EARTH, size=8.8)

    # ── control arrow ─────────────────────────────────────────────────
    ax.annotate("", xy=(11.82, 5.90), xytext=(11.82, 5.05),
                arrowprops=dict(arrowstyle="-|>", color=SIG, lw=2.6))
    ax.text(12.05, 5.42, "GPIO18 from Diagram 1\nopens / closes this",
            fontsize=9.2, color=SIG, fontweight="bold", va="center", zorder=5,
            bbox=dict(fc="white", ec="none", alpha=0.94, pad=1.8))

    # ── two explainer panels ──────────────────────────────────────────
    ax.add_patch(Rectangle((0.40, 0.40), 6.20, 1.90, fc="#f7f7f9", ec=EDGE,
                           lw=1.5, zorder=3))
    ax.text(0.62, 2.12, "WHY the PZEM comes BEFORE the relay",
            fontsize=10.4, fontweight="bold", va="top", zorder=4)
    ax.text(0.62, 1.74,
            "PZEM upstream, relay trips -> 230 V, 0.000 A, 0 W\n"
            "    proves the load really is dead.\n"
            "PZEM downstream, relay trips -> 0 V, 0 A, 0 W\n"
            "    identical to a dead sensor. You cannot tell.",
            fontsize=8.9, va="top", zorder=4, family="monospace")

    ax.add_patch(Rectangle((6.95, 0.40), 6.35, 1.90, fc="#f7f7f9", ec=EDGE,
                           lw=1.5, zorder=3))
    ax.text(7.17, 2.12, "WHAT PROTECTS WHAT — do not mix these up",
            fontsize=10.4, fontweight="bold", va="top", zorder=4)
    ax.text(7.17, 1.74,
            "relay at 312 W  ->  the working protection\n"
            "5 A fuse        ->  protects the WIRING only\n"
            "30 mA RCD       ->  protects YOU\n"
            "1.0 mm2 wire    ->  good for 13 A, far above the fuse",
            fontsize=8.9, va="top", zorder=4, family="monospace")

    legend(ax, 14.35, 0.95, [(LIVE, "L   live (brown)"),
                             (NEUT, "N   neutral (blue)"),
                             (EARTH, "PE  earth (green/yellow)")], size=9.2)

    p = os.path.join(OUT_DIR, "wiring_2_mains.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


if __name__ == "__main__":
    for path in (diagram_lowvoltage(), diagram_mains()):
        print("wrote", path)
