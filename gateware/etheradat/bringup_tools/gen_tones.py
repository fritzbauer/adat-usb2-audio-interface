#!/usr/bin/env python3
"""Generate the multichannel 48 kHz S32_LE test tones used by the loop tests.

Writes into /tmp/edtools (recreated on each run; /tmp is wiped by a reboot,
so regenerate the tones after every reboot — the .wav files are NOT stored
in the repo):
    tone16.wav      16ch: ch0-7 = 1 kHz (OUT1), ch8-15 = 2 kHz (OUT2)
    tone_out1.wav   16ch: ch0-7 = 1 kHz, rest silent
    tone_out2.wav   16ch: ch8-15 = 2 kHz, rest silent

Run with the gateware venv (numpy): venv/bin/python gen_tones.py
"""
import os
import numpy as np
import wave

FS = 48000
DUR = 2.5
AMP = 1 << 28
OUT = "/tmp/edtools"


def write(path, ch_freq):
    n = int(FS * DUR)
    t = np.arange(n) / FS
    x = np.zeros((n, 16), dtype=np.int32)
    for c, f in ch_freq.items():
        x[:, c] = (AMP * np.sin(2 * np.pi * f * t)).astype(np.int32)
    with wave.open(path, "wb") as w:
        w.setnchannels(16)
        w.setsampwidth(4)
        w.setframerate(FS)
        w.writeframes(x.tobytes())
    print("wrote", path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    write(f"{OUT}/tone16.wav",
          {**{c: 1000 for c in range(8)}, **{c: 2000 for c in range(8, 16)}})
    write(f"{OUT}/tone_out1.wav", {c: 1000 for c in range(0, 8)})
    write(f"{OUT}/tone_out2.wav", {c: 2000 for c in range(8, 16)})
