#!/usr/bin/env python3
"""Decode the PC's own playback frames (PC -> FPGA, 0x88B5 dir=1) to confirm
the kernel driver is encoding the test tone correctly.

RX payload layout (§4.2): 30 slots (s*2+o); o=0 -> OUT1, o=1 -> OUT2.
Expected with `aplay tone16.wav`: out0 ch0-7 = 1000 Hz, out1 ch0-7 = 2000 Hz.

Needs root (AF_PACKET). Run from gateware/ with the project venv:
  sudo venv/bin/python etheradat/bringup_tools/loop_sniff_tx.py [wav]
"""
import os
import socket
import struct
import subprocess
import sys
import time

import numpy as np

GATEWARE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, GATEWARE)
from etheradat.reference import adat_decode_frame  # noqa: E402

IF = os.environ.get("ADATNET_IF", "enxf8e43b907775")
PC = bytes.fromhex("f8e43b907775")
FS = 48000

TONE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/edtools/tone16.wav"
aplay = subprocess.Popen(["aplay", "-D", "hw:1,0", TONE],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.5)
s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3))
s.bind((IF, 0))
s.settimeout(0.2)
frames = {}
t = time.time()
while time.time() - t < 2.0 and len(frames) < 400:
    try:
        d, a = s.recvfrom(65535)
    except socket.timeout:
        continue
    if a[2] != socket.PACKET_OUTGOING:
        continue
    if len(d) == 982 and d[12:14] == b"\x88\xb5" and d[6:12] == PC and (d[15] & 0x08):
        frames[struct.unpack_from("<I", d, 16)[0]] = d[22:]
aplay.terminate()
order = sorted(frames)
print("PC playback frames:", len(order))
if len(order) >= 10:
    for o in (0, 1):
        for c in range(8):
            chans = []
            for cntr in order:
                p = frames[cntr]
                for smp in range(15):
                    slot = p[(smp * 2 + o) * 32:(smp * 2 + o) * 32 + 32]
                    samps, _ = adat_decode_frame(slot)
                    chans.append(float(samps[c]))
            a = np.array(chans); a = a - a.mean()
            rms = float(np.sqrt(np.mean(a ** 2)))
            seg = a[:min(len(a), 8192)] * np.hanning(min(len(a), 8192))
            spec = np.abs(np.fft.rfft(seg)); fr = np.fft.rfftfreq(len(seg), 1 / FS)
            k = int(np.argmax(spec[1:]) + 1) if len(spec) > 1 else 0
            if c == 0:
                print(f"  out{o} ch0: rms={rms:11.0f} peak={fr[k]:6.0f} Hz")
