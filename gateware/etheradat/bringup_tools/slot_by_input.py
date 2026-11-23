#!/usr/bin/env python3
"""Verify the TX (device->PC) payload slots against EMPTY_USER_FRAME.

With no ADAT input synced (free_run), all 45 slots must be the exact
silence frame. Decoding a rotated copy makes the PC driver reject every
frame (decode_frame: frame[0] != 0 -> Err(Sync)); this tool detects the
old "payload read one byte off" regression.

Needs root (AF_PACKET). Run from gateware/ with the project venv:
  sudo venv/bin/python etheradat/bringup_tools/slot_by_input.py
"""
import os
import socket
import sys
import time

GATEWARE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, GATEWARE)
from etheradat.reference import EMPTY_USER_FRAME  # noqa: E402

IF = os.environ.get("ADATNET_IF", "enxf8e43b907775")
FPGA = bytes.fromhex("02ad6e657401")
ROT = EMPTY_USER_FRAME[1:] + EMPTY_USER_FRAME[:1]

s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3))
s.bind((IF, 0))
s.settimeout(0.3)
t = time.time()
fr = None
while time.time() - t < 3:
    try:
        d, a = s.recvfrom(65535)
    except socket.timeout:
        continue
    if a[2] == socket.PACKET_OUTGOING:
        continue
    if len(d) == 1462 and d[12:14] == b"\x88\xb5" and d[6:12] == FPGA and d[15] == 0:
        fr = d
        break
if fr is None:
    print("no TX frame captured")
    sys.exit(1)
p = fr[22:]
n = len(p) // 32
per = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0]}   # exact, rotated, other
for k in range(n):
    sl = p[k * 32:(k + 1) * 32]
    inp = k % 3
    if sl == EMPTY_USER_FRAME:
        per[inp][0] += 1
    elif sl == ROT:
        per[inp][1] += 1
    else:
        per[inp][2] += 1
tot = sum(1 for k in range(n) if p[k*32:(k+1)*32] == EMPTY_USER_FRAME)
print(f"frame {len(fr)}B, slots {n}, exact-EMPTY {tot}/{n}")
for i in range(3):
    e, r, o = per[i]
    print(f"  input{i}: exact={e} rotated={r} other={o}")
