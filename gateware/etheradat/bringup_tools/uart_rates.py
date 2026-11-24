#!/usr/bin/env python3
"""Show per-second rates of the EtherADAT UART counters.

`parse_eduart.py` prints the raw 16-bit counters of one dump. Reading a rate
by eye from those is error-prone because the counters wrap every 65536, so
this tool parses several dumps, takes the deltas between consecutive 250 ms
dumps and reports the median rate. That avoids the aliasing a long-window
delta suffers once a counter wraps more than once.

Usage (from gateware/, with the project venv):
    sudo stty -F /dev/ttyUSB0 115200 cs8 -cstopb -parenb raw -echo
    timeout 3 sudo cat /dev/ttyUSB0 | \
        venv/bin/python etheradat/bringup_tools/uart_rates.py
"""
import os
import sys

GATEWARE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GATEWARE not in sys.path:
    sys.path.insert(0, GATEWARE)

from etheradat.bringup_tools.parse_eduart import parse, NAMES, NV  # noqa: E402

DUMP_PERIOD = 0.25          # the top dumps every 250 ms
SKIP = set(range(24, 56))   # byte-capture tap u16s (not scalar counters)
# level / length values whose "rate" is meaningless
SKIP |= {10, 11, 12, 13, 15, 17, 19, 20, 21, 22, 23}
SKIP |= {71, 72, 73}        # REF_CLK monitor: prev/last event + marker


def main():
    frames = parse(sys.stdin.buffer.read())
    print(f"dumps: {len(frames)}")
    if len(frames) < 2:
        return
    a, b = frames[0], frames[-1]
    print(f"elapsed ~{DUMP_PERIOD * (len(frames) - 1):.2f}s")
    print(f"{'name':12s} {'first':>8s} {'last':>8s} {'rate/s':>10s}")
    for k in range(NV):
        if k in SKIP:
            continue
        # per-dump rates; median rejects a torn sample at the edges
        rates = [((frames[i + 1][k] - frames[i][k]) & 0xFFFF) / DUMP_PERIOD
                 for i in range(len(frames) - 1)]
        rates.sort()
        rate = rates[len(rates) // 2]
        name = NAMES[k] if k < len(NAMES) else f"v{k}"
        print(f"{name:12s} {a[k]:8d} {b[k]:8d} {rate:10.0f}")


if __name__ == "__main__":
    main()
