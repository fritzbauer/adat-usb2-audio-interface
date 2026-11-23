#!/usr/bin/env python3
"""ADAT-INPUT debug: play an Ethernet->ADAT tone and decode what the FPGA
puts back into its TX frames for each of the 3 ADAT inputs.

This is the primary tool for the "ADAT IN -> PC is silent/garbage" bug:
it bypasses the ALSA capture path and reads the FPGA's own Ethernet TX
frames (passive AF_PACKET sniff, no arecord, so it does not wedge the
link), decoding slots (s*3+i) for input i with etheradat.reference.

Expected with OUT 9-16 -> IN 1-8 wiring and `aplay tone_out2.wav`:
  input0 (IN1) ch0..7  ~= 2000 Hz, rms ~ 1e8  (if the RX decode is correct)

Observed 23.11.2026: input0 mostly EMPTY (silence), occasional garbage
(peaks 3200/22400 Hz, tiny rms) despite sync0=1 -> ADAT RX decode broken.

Needs root (AF_PACKET). Run from gateware/ with the project venv:
  sudo venv/bin/python etheradat/bringup_tools/loop_sniff_all.py [wav]
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
FPGA = bytes.fromhex("02ad6e657401")
FS = 48000

TONE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/edtools/tone_out2.wav"
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
    if a[2] == socket.PACKET_OUTGOING:
        continue
    if len(d) == 1462 and d[12:14] == b"\x88\xb5" and d[6:12] == FPGA and d[15] == 0:
        frames[struct.unpack_from("<I", d, 16)[0]] = d[22:]
aplay.terminate()
order = sorted(frames)
print("collected FPGA TX frames:", len(order))
if len(order) < 10:
    sys.exit(0)
for inp in range(3):
    chans = [[] for _ in range(8)]
    for cntr in order:
        p = frames[cntr]
        for smp in range(15):
            slot = p[(smp * 3 + inp) * 32:(smp * 3 + inp) * 32 + 32]
            samps, _ = adat_decode_frame(slot)
            for c in range(8):
                chans[c].append(float(samps[c]))
    for c in range(8):
        a = np.array(chans[c])
        a = a - a.mean()
        rms = float(np.sqrt(np.mean(a ** 2)))
        seg = a[:min(len(a), 8192)] * np.hanning(min(len(a), 8192))
        spec = np.abs(np.fft.rfft(seg))
        fr = np.fft.rfftfreq(len(seg), 1 / FS)
        k = int(np.argmax(spec[1:]) + 1) if len(spec) > 1 else 0
        if rms <= 1000:
            continue
        print(f"  input{inp} (IN{inp+1}) ch{c}: rms={rms:11.0f} peak={fr[k]:6.0f} Hz")
