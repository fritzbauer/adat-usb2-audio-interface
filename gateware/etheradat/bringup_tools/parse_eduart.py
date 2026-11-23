#!/usr/bin/env python3
"""Decode the EtherADAT bring-up UART dump (115200, /dev/ttyUSB0).

Frame: AA 01 <56 x u16 LE> 5A, one dump every 250 ms. The taps
(sink_cap/mac_cap/pin_cap/sink_pay) are only present in the bridge top's
debug build; if the value count differs, adjust NV/TOTAL below (the byte
count is 2 + 2*NV + 1).

Usage:
    sudo stty -F /dev/ttyUSB0 115200 cs8 -cstopb -parenb raw -echo
    timeout 3 sudo cat /dev/ttyUSB0 | venv/bin/python etheradat/bringup_tools/parse_eduart.py
"""
import sys

NAMES = ["tx_frames", "rx_frames", "hello_tx", "hello_rx", "build", "feed",
         "rx_err", "tx_drop", "filtered", "underflow", "status0", "status1",
         "linked", "hello_due", "mac_sfn", "mac_sfs", "mac_pfn", "mac_pfs",
         "sink_len", "sink_count", "mac_len", "mac_count",
         "pin_len", "pin_count"]
CAP_SINK = 24          # first u16 of sink_cap (16 bytes -> 8 u16)
CAP_MAC = 32
CAP_PIN = 40
CAP_PAY = 48
NV = 56                # 24 named + 8*4 tap u16
TOTAL = 2 + 2 * NV + 1


def parse(buf):
    out = []
    i = 0
    while True:
        j = buf.find(b"\xaa\x01", i)
        if j < 0 or j + TOTAL > len(buf):
            break
        if buf[j + TOTAL - 1] != 0x5A:
            i = j + 1
            continue
        vals = [buf[j + 2 + 2 * k] | (buf[j + 3 + 2 * k] << 8)
                for k in range(NV)]
        out.append(vals)
        i = j + TOTAL
    return out


def cap_bytes(vals, base):
    b = bytearray()
    for k in range(8):
        w = vals[base + k]
        b += bytes([w & 0xFF, (w >> 8) & 0xFF])
    return bytes(b)


if __name__ == "__main__":
    fr = parse(sys.stdin.buffer.read())
    print("frames:", len(fr))
    for v in fr[-1:]:
        for n, val in zip(NAMES, v):
            print(f"  {n:10s} {val:8d}  0x{val:04x}")
        s0 = v[10]
        print("  status0:", " ".join(
            f"{n}={(s0 >> b) & 1}" for b, n in enumerate(
                ["link", "pc_linked", "free_run", "sync0", "sync1", "sync2",
                 "phy_found"])))
        print("  SINK:", cap_bytes(v, CAP_SINK).hex(' '))
        print("  MAC :", cap_bytes(v, CAP_MAC).hex(' '))
        print("  PIN :", cap_bytes(v, CAP_PIN).hex(' '))
        print("  PAY :", cap_bytes(v, CAP_PAY).hex(' '))
