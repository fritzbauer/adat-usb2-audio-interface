#!/usr/bin/env python3
"""Read the ILA capture from the ADATface diagnostic endpoint (EP 4 IN /
0x84, interface 4) and decode it using ila.P (pickled ILACoreParameters).

Usage: gateware/venv/bin/python tools/audio_test/ila_read.py <out_prefix>

Start this BEFORE the audio stream that triggers the ILA. The ILA emits
one sample buffer (sample_depth * bytes_per_sample bytes) per trigger;
this tool keeps reading until the timeout expires.
"""
import sys
import os
import time
import struct

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "amlib"))

import usb.core
import usb.util

GATEWARE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, GATEWARE)

from amlib.debug.ila import ILACoreParameters


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "ila_dump"
    timeout_s = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0

    ila = ILACoreParameters.unpickle(os.path.join(GATEWARE, "ila.P"))
    bps = ila.bytes_per_sample
    depth = ila.sample_depth
    bufsize = depth * bps
    print(f"ILA: {len(ila.signals)} signals, depth={depth}, "
          f"bits/sample={ila.bits_per_sample}, bytes/sample={bps}")

    dev = usb.core.find(idVendor=0x1209, idProduct=0xada1)
    if dev is None:
        print("ADATface not found"); return 1
    cfg = dev.get_active_configuration()
    itf = usb.util.find_descriptor(cfg, custom_match=lambda i: i.bInterfaceNumber == 4)
    if itf is None:
        print("no interface 4 (USE_ILA not enabled in this gateware?)"); return 1
    if dev.is_kernel_driver_active(4):
        dev.detach_kernel_driver(4)
    usb.util.claim_interface(dev, 4)
    ep = usb.util.find_descriptor(itf, custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN)
    print(f"reading from EP {hex(ep.bEndpointAddress)}, expecting {bufsize} bytes per capture")

    raw = b""
    captures = []
    t_end = time.time() + timeout_s
    while time.time() < t_end:
        want = min(4096, bufsize - len(raw)) if len(raw) < bufsize else 4096
        try:
            data = ep.read(want, timeout=1000)
        except usb.core.USBError as e:
            if e.errno == 110:  # timeout -> no data yet
                if len(raw) > bufsize:
                    captures.append(raw)
                    raw = b""
                continue
            elif e.errno == 75:  # overflow -> data lost, retry smaller
                print("  EOVERFLOW, retrying")
                continue
            raise
        raw += bytes(data)
        if len(raw) >= bufsize:
            captures.append(raw)
            raw = b""
            print(f"  capture #{len(captures)} complete ({bufsize} bytes)")

    usb.util.release_interface(dev, 4)

    if not captures:
        print("NO CAPTURES (no trigger occurred)")
        return 1

    for i, cap in enumerate(captures):
        path = f"{prefix}_{i}.bin"
        with open(path, "wb") as f:
            f.write(cap)
        print(f"saved {path} ({len(cap)} bytes)")

    # decode the first capture
    cap = captures[0]
    n = len(cap) // bps
    print(f"\ndecoding {n} samples from capture 0")
    signals = ila.signals
    offsets = []
    pos = 0
    for s in signals:
        offsets.append((s.name, pos, len(s)))
        pos += len(s)
    rows = []
    for i in range(n):
        word = int.from_bytes(cap[i*bps:(i+1)*bps], "little")
        vals = {name: (word >> off) & ((1 << w) - 1) for name, off, w in offsets}
        rows.append(vals)
    with open(f"{prefix}_decoded.txt", "w") as f:
        header = ["idx"] + [name for name, _, _ in offsets]
        f.write("\t".join(header) + "\n")
        for i, r in enumerate(rows):
            f.write("\t".join([str(i)] + [str(r[name]) for name, _, _ in offsets]) + "\n")
    print(f"wrote {prefix}_decoded.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
