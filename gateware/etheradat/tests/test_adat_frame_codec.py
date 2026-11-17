"""Tests for the canonical ADAT frame codec, cross-checked against the
project's proven ADAT wire path (ADATTransmitter -> NRZI wire ->
ADATReceiver) and the PC-side reference model.

Run:  python -m etheradat.tests.test_adat_frame_codec
"""

import os
import random
import unittest

from amaranth import *
from amaranth.sim import Simulator, Tick

from adat import ADATTransmitter, ADATReceiver

from etheradat import reference as ref
from etheradat.adat_frame_codec import (InputFrameCollector,
                                        FrameDeserializer)

FAST_CLK = 98_304_000
SYNC_CLK = 61_440_000
ADAT_CLK = 12_288_000


class WireCollectorChain(Elaboratable):
    """ADATTransmitter (sync/adat domains) -> NRZI wire ->
    ADATReceiver + InputFrameCollector (fast domain).

    Exposes the transmitter's commit interface and a test read port on
    the collector's ring.
    """

    def __init__(self):
        self.tx = ADATTransmitter(fifo_depth=8)
        self.rx = ADATReceiver(FAST_CLK)
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


class TestFrameCodecAgainstWire(unittest.TestCase):
    def run_sim(self, dut, processes):
        """processes: list of (process_function, domain) — the functions
        are zero-argument callables returning generators."""
        sim = Simulator(dut)
        sim.add_clock(1 / SYNC_CLK)
        sim.add_clock(1 / ADAT_CLK, domain="adat")
        sim.add_clock(1 / FAST_CLK, domain="fast")
        for proc, domain in processes:
            sim.add_sync_process(proc, domain=domain)
        if os.getenv("GENERATE_VCDS"):
            with sim.write_vcd("etheradat_codec.vcd"):
                sim.run()
        else:
            sim.run()

    @staticmethod
    def commit_frame(dut, samples, user):
        """Commit one frame's 8 samples into the transmitter."""
        for _ in range(10):          # settle: let the FSMs leave reset
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

    @staticmethod
    def read_ring_slots(dut, out, n_slots, wait_ticks=120_000):
        """Wait for the receiver to sync, then read n_slots ring slots."""
        for _ in range(wait_ticks):
            yield Tick("fast")
        for slot in range(n_slots):
            for w in range(8):
                yield dut.ring_read.addr.eq(slot * 8 + w)
                yield Tick("fast")
                yield Tick("fast")
                out.append((yield dut.ring_read.data))

    def test_wire_to_ring_produces_reference_bytes(self):
        """Samples fed into the real transmitter must come back as the
        canonical 32-byte frames the PC expects. Two DIFFERENT frames
        are committed: the transmitter repeats the last one when the
        FIFO runs empty, so a single frame could hide a collector that
        latches a stale channel 7 (live[7] still holds the previous
        frame's ch7 at the frame-end edge)."""
        rng = random.Random(0xC0FFEE)
        samples_a = [rng.getrandbits(24) for _ in range(8)]
        samples_b = [rng.getrandbits(24) for _ in range(8)]
        user_a, user_b = 0b1001, 0b0110     # order-sensitive
        expected_b = ref.adat_encode_frame(samples_b, user_b)

        dut = WireCollectorChain()

        def driver():
            yield from self.commit_frame(dut, samples_a, user_a)
            yield from self.commit_frame(dut, samples_b, user_b)
            for _ in range(4_000):
                yield Tick("sync")

        words = []

        def reader():
            yield from self.read_ring_slots(dut, words, 3)

        self.run_sim(dut, [(driver, "sync"), (reader, "fast")])

        # slot 0 is the first frame after sync (may still carry partial
        # data from the acquisition); the settled slots must be the
        # second frame, exactly
        for slot in (1, 2):
            frame = b"".join(w.to_bytes(4, "big")
                             for w in words[slot * 8:(slot + 1) * 8])
            self.assertEqual(frame, expected_b,
                             f"slot {slot}: wire path produced different "
                             "canonical bytes")

    def test_wire_silence_is_empty_user_frame(self):
        """Zero samples + zero user bits must produce the exact
        EMPTY_USER_FRAME constant of the PC reference."""
        expected = ref.EMPTY_USER_FRAME
        dut = WireCollectorChain()

        def driver():
            yield from self.commit_frame(dut, [0] * 8, 0)
            for _ in range(4_000):
                yield Tick("sync")

        words = []

        def reader():
            yield from self.read_ring_slots(dut, words, 1)

        self.run_sim(dut, [(driver, "sync"), (reader, "fast")])

        frame = b"".join(w.to_bytes(4, "big") for w in words)
        self.assertEqual(frame, expected)


class TestFrameDeserializer(unittest.TestCase):
    def test_reference_frames_decode_to_samples(self):
        rng = random.Random(0x1234)
        for _ in range(5):
            samples = [rng.getrandbits(24) for _ in range(8)]
            user = rng.getrandbits(4)
            frame = ref.adat_encode_frame(samples, user)

            mem = Memory(width=32, depth=8,
                         init=[int.from_bytes(frame[4*i:4*i+4], "big")
                               for i in range(8)])
            dut = FrameDeserializer(mem, domain="sync")

            captured = []

            def proc():
                for _ in range(5):
                    yield Tick("sync")
                yield dut.read_base_in.eq(0)
                yield dut.read_strobe_in.eq(1)
                for _ in range(60):
                    yield Tick("sync")
                    if (yield dut.samples_valid_out):
                        break
                yield Tick("sync")
                for ch in range(8):
                    captured.append((yield dut.sample_out[ch]))
                captured.append((yield dut.user_out))

            sim = Simulator(dut)
            sim.add_clock(1 / SYNC_CLK)
            sim.add_sync_process(proc)
            if os.getenv("GENERATE_VCDS"):
                with sim.write_vcd("etheradat_deser.vcd"):
                    sim.run()
            else:
                sim.run()

            got_samples = captured[:8]
            got_user = captured[8]
            self.assertEqual(got_samples, samples)
            self.assertEqual(got_user, user)


if __name__ == "__main__":
    unittest.main(verbosity=2)
