#!/usr/bin/env python3
"""Decode the old UART-probe format [0xAA][marker][len][len bytes]."""
import os
import select
import sys
import time

dev = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
fd = os.open(dev, os.O_RDONLY)
os.set_blocking(fd, False)
buf = bytearray()
end = time.monotonic() + seconds
while time.monotonic() < end:
    r, _, _ = select.select([fd], [], [], 0.1)
    if r:
        buf += os.read(fd, 4096)
os.close(fd)
frames = []
i = 0
while i < len(buf) - 3:
    if buf[i] != 0xAA:
        i += 1
        continue
    marker, length = buf[i+1], buf[i+2]
    if length == 0 or length > 128 or i + 3 + length > len(buf):
        i += 1
        continue
    frames.append((marker, bytes(buf[i+3:i+3+length])))
    i += 3 + length
print(f"{len(frames)} frames")
for marker, data in frames[:12]:
    print(f"marker={marker:#04x} len={len(data)} head={data[:40].hex(' ')}")
