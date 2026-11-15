#!/usr/bin/env python3
"""Simulation testbench for the patched NRZIDecoder.

Generates the exact NRZI bitstream the ADATTransmitter would send for a
ramp signal (same encoding: per channel 0b1dddd1dddd1dddd1dddd1dddd1dddd,
sync pad 0b1000000000010000, NRZI: 0 = toggle, 1 = hold) at 12.288 Mbit/s,
oversampled 8x in the fast domain (98.304 MHz). Checks that the decoder
locks (running), emits one data_out_en pulse per bit, and decodes the
expected bits.

Run: gateware/venv/bin/python tools/audio_test/sim_nrzidecoder.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "venv", "lib", "python3.13", "site-packages"))

from amaranth import *
from amaranth.sim import Simulator

from adat.nrzidecoder import NRZIDecoder

CLK_FREQ = 98_304_000
BIT_CYCLES = CLK_FREQ // 12_288_000  # 8


def tx_frame_bits(n: int, nch: int = 8) -> list:
    """The bits the transmitter sends for frame n (before NRZI)."""
    bits = []
    for ch in range(nch):
        sample = ((n & 0x3FFF) << 1) | (ch >> 4)  # 24-bit, upper-24 container
        sample &= 0xFFFFFF
        for nib in range(6):
            bits.append(1)  # sync bit
            nib_val = (sample >> (20 - 4 * nib)) & 0xF
            for b in range(3, -1, -1):
                bits.append((nib_val >> b) & 1)
    # sync pad: 0b1000000000010000 (MSB first), user bits 0
    pad = 0b1000000000010000
    for b in range(15, -1, -1):
        bits.append((pad >> b) & 1)
    return bits


def nrzi_encode(bits: list) -> list:
    """INVERTED NRZI as used by this design (see ADATTransmitter's
    NRZIEncoder: data 1 toggles the line, data 0 holds it)."""
    levels = []
    level = 1
    for b in bits:
        if b == 1:
            level ^= 1
        levels.append(level)
    return levels


def main():
    dut = NRZIDecoder(CLK_FREQ)

    # build ~30 frames of the test signal
    all_bits = []
    for n in range(1, 31):
        all_bits += tx_frame_bits(n)
    levels = nrzi_encode(all_bits)

    sim = Simulator(dut)
    sim.add_clock(1 / CLK_FREQ, domain="sync")

    idx = 0
    edge_count = 0
    outen_count = 0
    outen_positions = []
    prev_level = levels[0]
    prev_data = None
    decoded_bits = []
    locked_at = None

    def process():
        # NOTE: amaranth 0.5's pysim: the drive sequence must be
        # precomputed (rebinding the level variable in the drive loop
        # breaks the assignments) and each value needs eq + one bare
        # yield; without the bare yield the assignments stop applying.
        oversampled = []
        for lvl in levels:
            oversampled.extend([lvl] * BIT_CYCLES)
        oversampled.extend([0] * (BIT_CYCLES * 4))
        for v in oversampled:
            yield dut.nrzi_in.eq(v)
            yield

    def monitor():
        nonlocal idx, edge_count, outen_count, locked_at
        cycles = 0
        while cycles < BIT_CYCLES * len(levels) + BIT_CYCLES * 8:
            running = yield dut.running
            outen = yield dut.data_out_en
            data = yield dut.data_out
            if running and locked_at is None:
                locked_at = cycles
                print(f"LOCKED at cycle {cycles} (bit {cycles // BIT_CYCLES})")
            if outen:
                outen_count += 1
                outen_positions.append(cycles)
                decoded_bits.append(data)
            cycles += 1
            yield

    sim.add_sync_process(monitor, domain="sync")
    sim.add_sync_process(process, domain="sync")
    sim.run()

    total_bits = len(levels)
    print(f"bits sent: {total_bits}, data_out_en pulses: {outen_count}")
    if locked_at is not None:
        # With the eq+bare-yield drive, one monitor cycle == one
        # oversampled value; the lock at monitor cycle c == input bit c/8.
        locked_bit = locked_at // BIT_CYCLES
        print(f"locked at bit {locked_bit} of {total_bits}")
        # after lock, the decoder should emit ~1 pulse per bit period
        tail = [p for p in outen_positions if p > locked_at + BIT_CYCLES]
        gaps = [b - a for a, b in zip(tail, tail[1:])]
        import collections
        print("pulse gap histogram (oversampled values):", dict(collections.Counter(gaps).most_common(6)))
        # each pulse at oversampled position p decodes input bit p // 8
        tail = [t for t in tail if t // BIT_CYCLES < len(all_bits)]
        mism = sum(1 for p, d in zip(tail, decoded_bits[-len(tail):])
                   if d != all_bits[p // BIT_CYCLES])
        print(f"decoded vs expected mismatches (post-lock): {mism}/{len(tail)}")
    else:
        print("NEVER LOCKED")


if __name__ == "__main__":
    main()
