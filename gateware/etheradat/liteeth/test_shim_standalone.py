#!/usr/bin/env python3
"""Standalone tests of the RMII shim (without the LiteEth MAC core):
TX byte stream -> dibits (LSB first, TX_EN framing) and RMII dibits ->
byte stream with frame delimitation.

TX contract note: the shim serializes continuously while beats are
offered; a mid-frame gap in the sink stream would drop TX_EN mid-frame.
The LiteEth MAC core's store-and-forward packet FIFO guarantees
gapless beats (that is exactly what it exists for), so the shim never
sees gaps in the real system — and this test drives it gaplessly.

Run with the scratch venv:
    /tmp/etheradat-scratch/liteenv/bin/python test_shim_standalone.py
"""

import os
import sys
import unittest

from migen import *
from litex.gen import LiteXModule, ClockDomainsRenamer
from litex.gen.sim import Simulator
from litex.soc.interconnect import stream

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmii_shim import RMIITXShim, RMIIRXShim

from liteeth.common import eth_phy_description


class TXOnly(LiteXModule):
    def __init__(self):
        self.tx_en = Signal()
        self.txd = Signal(2)
        self.submodules.shim = RMIITXShim(self.tx_en, self.txd)


class RXOnly(LiteXModule):
    def __init__(self):
        self.crs_dv = Signal()
        self.rxd = Signal(2)
        self.submodules.shim = ClockDomainsRenamer("eth_rx")(
            RMIIRXShim(self.crs_dv, self.rxd))


class TestTXShim(unittest.TestCase):
    def test_lsb_first_dibits_and_tx_en(self):
        """Bytes leave LSB-first as dibits; TX_EN covers exactly the
        presented bytes."""
        dut = TXOnly()
        payload = [0x55, 0xD5, 0x00, 0xFF, 0xA7]
        got = []

        def drive():
            # gapless handshake: each iteration issues the beat (it
            # lands at the next edge), then waits for a cycle in which
            # the sink accepts; the next beat lands exactly at that
            # accepting cycle's end
            for i, byte in enumerate(payload):
                yield dut.shim.sink.payload.data.eq(byte)
                yield dut.shim.sink.valid.eq(1)
                yield dut.shim.sink.last.eq(i == len(payload) - 1)
                yield
                while not (yield dut.shim.sink.ready):
                    yield
            yield dut.shim.sink.valid.eq(0)
            yield dut.shim.sink.last.eq(0)
            for _ in range(200):
                yield

        def capture():
            for _ in range(600):
                yield
                en = (yield dut.tx_en)
                txd = (yield dut.txd)
                got.append((en, txd))

        sim = Simulator(dut, {"sys": [drive(), capture()]},
                        clocks={"sys": 10})
        with sim:
            sim.run()

        dibits = [t for en, t in got if en]
        self.assertEqual(len(dibits), 5 * 4,
                         f"TX_EN framing wrong: {len(dibits)} dibit cycles")
        for byte_idx, byte in enumerate(payload):
            for pair in range(4):
                want = (byte >> (2 * pair)) & 0x3
                got_d = dibits[byte_idx * 4 + pair]
                self.assertEqual(got_d, want,
                                 f"byte {byte_idx} dibit {pair}")


class TestRXShim(unittest.TestCase):
    def test_bytes_from_dibits_with_framing(self):
        """Dibits reassemble to bytes (LSB first); last on CRS_DV loss;
        preamble passed through."""
        dut = RXOnly()
        frame = [0x55, 0x55, 0xD5, 0x02, 0x11, 0xF0]   # incl. preamble

        got = []

        def drive():
            for byte in frame:
                for pair in range(4):
                    yield dut.crs_dv.eq(1)
                    yield dut.rxd.eq((byte >> (2 * pair)) & 0x3)
                    yield
            yield dut.crs_dv.eq(0)
            yield dut.rxd.eq(0)
            yield
            yield

        def collect():
            yield dut.shim.source.ready.eq(1)   # consume every beat
            for _ in range(400):
                yield
                if (yield dut.shim.source.valid):
                    data = (yield dut.shim.source.payload.data)
                    last = (yield dut.shim.source.last)
                    got.append((data, last))

        sim = Simulator(dut, {"eth_rx": [drive(), collect()]},
                        clocks={"sys": 10, "eth_rx": 10})
        with sim:
            sim.run()

        self.assertEqual([d for d, _ in got], frame,
                         "reassembled bytes differ")
        self.assertTrue(got[-1][1], "last not flagged on the final byte")
        self.assertFalse(any(l for _, l in got[:-1]),
                         "last flagged mid-frame")


if __name__ == "__main__":
    unittest.main(verbosity=2)
