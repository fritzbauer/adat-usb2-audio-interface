#!/usr/bin/env python3
"""Full-duplex loop test: play tone16 while capturing both the AdatNet
capture PCM and the Audient iD14 ADAT input.

- OUT1 (playback ch0-7, 1 kHz) -> Audient ADAT input (Audient capture
  ch2-9 should show ~1000 Hz)  -- this path was VERIFIED working.
- OUT2 (playback ch8-15, 2 kHz) -> (your routing; e.g. IN 1-8) -> AdatNet
  capture (see loop_sniff_all.py for the per-input decode).

Requires the NIC at full-duplex + enough USB bandwidth; on a half-duplex
or 12 Mbit-USB link this test wedges the FPGA TX (reflash to recover).

Run from gateware/ with the project venv:
  venv/bin/python etheradat/bringup_tools/gen_tones.py
  venv/bin/python etheradat/bringup_tools/loop_full.py
"""
import os
import subprocess
import time

import numpy as np
import wave

FS = 48000
wav = "/tmp/edtools/tone16.wav"
cap_ad = "/tmp/edtools/loop_adatnet.wav"
cap_au = "/tmp/edtools/loop_audient.wav"
ADATNET = os.environ.get("ADATNET_PCM", "hw:1,0")
AUDIENT = os.environ.get("AUDIENT_PCM", "hw:0,0")

for f in (cap_ad, cap_au):
    try:
        os.remove(f)
    except OSError:
        pass
r_ad = subprocess.Popen(["arecord", "-D", ADATNET, "-f", "S32_LE", "-r", str(FS),
                         "-c", "24", "-d", "4", cap_ad], stderr=subprocess.DEVNULL)
r_au = subprocess.Popen(["arecord", "-D", AUDIENT, "-f", "S32_LE", "-r", str(FS),
                         "-c", "10", "-d", "4", cap_au], stderr=subprocess.DEVNULL)
time.sleep(0.6)
p = subprocess.run(["aplay", "-D", ADATNET, wav], capture_output=True, text=True)
print("aplay rc", p.returncode)
r_ad.wait()
r_au.wait()


def analyze(path, label, chans):
    try:
        with wave.open(path, "rb") as w:
            nc = w.getnchannels()
            raw = w.readframes(w.getnframes())
    except Exception as e:
        print(f"{label}: {e}")
        return
    a = np.frombuffer(raw, dtype=np.int32).reshape(-1, nc).astype(np.float64)
    print(f"\n{label}: {nc}ch {len(a)} frames")
    for ch in chans:
        if ch >= nc:
            continue
        s = a[:, ch] - a[:, ch].mean()
        rms = float(np.sqrt(np.mean(s ** 2)))
        seg = s[:min(len(s), 32768)] * np.hanning(min(len(s), 32768))
        spec = np.abs(np.fft.rfft(seg))
        fr = np.fft.rfftfreq(len(seg), 1 / FS)
        k = int(np.argmax(spec[1:]) + 1) if len(spec) > 1 else 0
        print(f"  ch{ch:2d}: rms={rms:11.0f} peak={fr[k]:6.0f} Hz")


analyze(cap_ad, "adatnet capture (ch8-15 = IN2/OUT2 loop, expect 2kHz)", range(24))
analyze(cap_au, "Audient ADAT (ch2-9 = OUT1, expect 1kHz)", range(10))
