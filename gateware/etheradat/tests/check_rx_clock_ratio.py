#!/usr/bin/env python3
"""Standalone diagnostic: does the ADAT receiver decode at a given fast-clock
ratio? (Not part of run_tests -- it deliberately simulates a broken ratio.)

The shared adat-core ADATReceiver/NRZIDecoder hardcodes `bit_time = 8` fast
cycles per ADAT bit, so the "fast" clock MUST be exactly 8x the 12.288 MHz
ADAT bit clock (98.304 MHz). This replays the exact
ADATTransmitter -> NRZI wire -> ADATReceiver -> InputFrameCollector chain at
8x (correct) and 9x (the old EP4CGX150 fast_multiplier=9 bug: receiver
"syncs" but the ring stays EMPTY). It is the reproduction used to find the
2026-11-23 ADAT-input bug.

    venv/bin/python etheradat/tests/check_rx_clock_ratio.py

Expected: slot == B (exact frame) at 8x; EMPTY at 9x. The regression guard
for the platform setting itself is tests/test_platform_clocks.py.
"""
import os
import random
import sys

from amaranth import Elaboratable, Module, DomainRenamer
from amaranth.sim import Simulator, Tick

GATEWARE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GATEWARE not in sys.path:
    sys.path.insert(0, GATEWARE)

from adat import ADATTransmitter, ADATReceiver  # noqa: E402
from etheradat import reference as ref  # noqa: E402
from etheradat.adat_frame_codec import InputFrameCollector  # noqa: E402

ADAT_CLK = 12_288_000
SYNC_CLK = 61_440_000


class Chain(Elaboratable):
    """Transmitter -> NRZI wire -> receiver -> collector, for one ratio."""

    def __init__(self):
        self.tx = ADATTransmitter(fifo_depth=8)
        self.rx = ADATReceiver(98_304_000)   # clk_freq is ignored by the decoder
        self.collector = InputFrameCollector(domain="fast")
        self.ring_read = self.collector.ring.read_port(domain="fast")

    def elaborate(self, platform):
        m = Module()
        m.submodules.tx = self.tx
        m.submodules.rx = DomainRenamer("fast")(self.rx)
        m.submodules.collector = self.collector
        m.submodules.ring_read = self.ring_read
        m.d.comb += [
            self.collector.sample_in.eq(self.rx.sample_out),
            self.collector.addr_in.eq(self.rx.addr_out),
            self.collector.enable_in.eq(self.rx.output_enable),
            self.collector.user_in.eq(self.rx.user_data_out),
            self.collector.synced_in.eq(self.rx.synced_out),
            self.rx.adat_in.eq(self.tx.adat_out),
        ]
        return m


def run(fast_clk, tag):
    rng = random.Random(0xC0FFEE)
    sa = [rng.getrandbits(24) for _ in range(8)]
    sb = [rng.getrandbits(24) for _ in range(8)]
    ua, ub = 0b1001, 0b0110
    exp_a = ref.adat_encode_frame(sa, ua)
    exp_b = ref.adat_encode_frame(sb, ub)
    dut = Chain()

    def commit(samples, user):
        for _ in range(10):
            yield Tick("sync")
        for ch in range(8):
            yield dut.tx.sample_in.eq(samples[ch])
            yield dut.tx.addr_in.eq(ch)
            yield dut.tx.valid_in.eq(1)
            yield dut.tx.last_in.eq(ch == 7)
            yield dut.tx.user_data_in.eq(user)
            while not (yield dut.tx.ready_out):
                yield Tick("sync")
            yield Tick("sync")
        yield dut.tx.valid_in.eq(0)
        yield dut.tx.last_in.eq(0)

    def driver():
        yield from commit(sa, ua)
        yield from commit(sb, ub)       # two frames; a repeated one hides bugs
        for _ in range(4000):
            yield Tick("sync")

    words = []

    def reader():
        for _ in range(120_000):
            yield Tick("fast")
        for slot in range(3):
            for w in range(8):
                yield dut.ring_read.addr.eq(slot * 8 + w)
                yield Tick("fast")
                yield Tick("fast")
                words.append((yield dut.ring_read.data))

    sim = Simulator(dut)
    sim.add_clock(1 / SYNC_CLK)
    sim.add_clock(1 / ADAT_CLK, domain="adat")
    sim.add_clock(1 / fast_clk, domain="fast")
    sim.add_sync_process(driver, domain="sync")
    sim.add_sync_process(reader, domain="fast")
    sim.run()

    print(f"=== {tag}: fast={fast_clk / 1e6:.3f} MHz, "
          f"ratio={fast_clk / ADAT_CLK:.1f}x ===")
    for slot in range(3):
        frame = b"".join(w.to_bytes(4, "big")
                         for w in words[slot * 8:slot * 8 + 8])
        verdict = ("B" if frame == exp_b else
                   "A" if frame == exp_a else
                   "EMPTY" if frame == ref.EMPTY_USER_FRAME else "GARBAGE")
        print(f"  slot {slot}: {verdict}  {frame.hex()}")
    sys.stdout.flush()


if __name__ == "__main__":
    run(98_304_000, "8x (correct: fast_multiplier=8)")
    run(110_592_000, "9x (old Cyclone IV bug: fast_multiplier=9)")
