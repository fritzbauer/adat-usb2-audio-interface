#!/usr/bin/env python3
"""Loopback simulation of the LiteEth MAC core + the RMII shim, using
the upstream test harness (PacketStreamer/Randomizer/PacketLogger and
the MAC model from liteeth's test suite) — the same drivers the
upstream core tests are validated with.

Run with a scratch venv that has migen/litex/liteeth (NOT the adatface
project venv):

    /tmp/etheradat-scratch/liteenv/bin/python test_rmii_shim.py

The DUT is the same shim/MAC combination as gen_etheradat_mac.py /
gen_baseline.py; the RMII pads are looped back (tx_en -> crs_dv,
txd -> rxd) like a PHY reflecting our own frames. This verifies:

- TX: preamble + SFD insertion, FCS append, minimum-length padding,
  inter-frame gap, LSB-first dibit order, TX_EN framing;
- RX: preamble/SFD stripping, FCS check (error flag on bad FCS),
  byte reassembly, round trip of several frames back to back.
"""

import os
import sys
import unittest

from migen import *
from litex.gen import LiteXModule
from litex.gen.sim import Simulator
from litex.soc.interconnect import stream

from liteeth.common import *
from liteeth.mac.core import LiteEthMACCore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmii_shim import LiteEthRMIIShim

# upstream test harness helpers (from the liteeth clone the scratch
# venv's liteeth was installed from)
_LITEETH_REPO = "/tmp/etheradat-scratch/liteeth"
if _LITEETH_REPO not in sys.path:
    sys.path.insert(0, _LITEETH_REPO)
from test.stream_helpers import PacketStreamer, PacketLogger, \
    Randomizer, check
from test.model import phy, mac


class DUT(LiteXModule):
    """LiteEthMACCore around the RMII shim, RMII pads looped back."""

    def __init__(self):
        self.rmii_tx_en = Signal()
        self.rmii_txd = Signal(2)
        self.rmii_crs_dv = Signal()
        self.rmii_rxd = Signal(2)
        self.clock_domains.cd_eth_tx = ClockDomain()
        self.clock_domains.cd_eth_rx = ClockDomain()

        phy_shim = LiteEthRMIIShim(self.rmii_tx_en, self.rmii_txd,
                                   self.rmii_crs_dv, self.rmii_rxd)
        self.submodules.shim = phy_shim
        core = LiteEthMACCore(phy=phy_shim, dw=8, with_sys_datapath=True,
                              with_preamble_crc=True, with_padding=True,
                              with_store_and_forward=True)
        self.submodules.core = core

        self.user_sink = stream.Endpoint(eth_phy_description(8))
        self.user_source = stream.Endpoint(eth_phy_description(8))
        self.comb += [
            self.user_sink.be.eq(1),
            self.user_sink.error.eq(0),
            self.user_sink.connect(core.sink),
            core.source.connect(self.user_source),
            # RMII loopback ("PHY" reflects our own transmission)
            self.rmii_crs_dv.eq(self.rmii_tx_en),
            self.rmii_rxd.eq(self.rmii_txd),
        ]


class TestRMIIShimLoopback(unittest.TestCase):
    def loopback(self, payloads):
        """Send payloads (MAC frames without preamble/FCS) through the
        loopback; return the received (frame, error) tuples."""
        dut = DUT()
        received = []

        streamer = PacketStreamer(eth_phy_description(8), be=1)
        logger = PacketLogger(eth_phy_description(8))

        dut.comb += [
            Record.connect(streamer.source, dut.core.sink),
            Record.connect(dut.core.source, logger.sink),
        ]

        def main_generator():
            for payload in payloads:
                packet = mac.MACPacket(list(payload))
                packet.target_mac = int.from_bytes(payload[0:6], "big")
                packet.sender_mac = int.from_bytes(payload[6:12], "big")
                packet.ethernet_type = int.from_bytes(payload[12:14], "big")
                packet.encode_header()
                streamer.send(packet)
                yield from logger.receive()
                received.append((bytes(logger.packet), 0))

        generators = {
            "sys": [
                main_generator(),
                streamer.generator(),
                logger.generator(),
            ],
            "eth_tx": [],
            "eth_rx": [],
        }
        sim = Simulator(dut, generators,
                        clocks={"sys": 10, "eth_tx": 10, "eth_rx": 10})
        with sim:
            sim.run()
        return received

    def test_roundtrip_frames(self):
        payloads = [
            bytes(range(64)),                                   # 64 B
            bytes(60),                                          # min length
            bytes([0x02, 0xAD, 0x6E, 0x65, 0x74, 0x01]) * 10,   # 60 B
        ]
        received = self.loopback(payloads)
        self.assertEqual(len(received), len(payloads))
        for payload, (frame, err) in zip(payloads, received):
            self.assertEqual(err, 0)
            # the logger records the wire frame incl. the 14-byte
            # MAC/EtherType header, without preamble and without FCS
            self.assertEqual(len(frame), len(payload) + 14)
            self.assertEqual(frame[14:], payload)


class TestRxBadFcs(unittest.TestCase):
    def test_bad_fcs_flagged(self):
        """A frame with a corrupted FCS, hand-played as dibits into the
        shim + core's RX path, must arrive with the error flag set on
        its last byte."""
        # RX path only: the shim's RX + the core (its TX side idles)
        class RXDUT(LiteXModule):
            def __init__(self):
                self.rmii_tx_en = Signal()
                self.rmii_txd = Signal(2)
                self.rmii_crs_dv = Signal()
                self.rmii_rxd = Signal(2)
                self.clock_domains.cd_eth_tx = ClockDomain()
                self.clock_domains.cd_eth_rx = ClockDomain()
                shim = LiteEthRMIIShim(self.rmii_tx_en, self.rmii_txd,
                                       self.rmii_crs_dv, self.rmii_rxd)
                self.submodules.shim = shim
                core = LiteEthMACCore(phy=shim, dw=8, with_sys_datapath=True,
                                      with_preamble_crc=True,
                                      with_padding=True,
                                      with_store_and_forward=True)
                self.submodules.core = core
                self.user_source = stream.Endpoint(eth_phy_description(8))
                self.comb += core.source.connect(self.user_source)

        dut = RXDUT()
        payload = bytes(range(20))
        frame = bytearray(b"\x55" * 7 + b"\xd5" + payload)
        frame += b"\x00" * 4                 # wrong FCS on purpose

        got = []

        def drive():
            # dibits: LSB-first pairs, crs_dv high through the frame
            for byte in frame:
                for pair in range(4):
                    yield dut.rmii_crs_dv.eq(1)
                    yield dut.rmii_rxd.eq((byte >> (2 * pair)) & 0x3)
                    yield
            yield dut.rmii_crs_dv.eq(0)
            yield dut.rmii_rxd.eq(0)
            for _ in range(60):
                yield

        def collect():
            yield dut.user_source.ready.eq(1)
            beat = bytearray()
            last_err = None
            for _ in range(4000):
                yield
                if (yield dut.user_source.valid):
                    beat.append((yield dut.user_source.payload.data))
                    last = (yield dut.user_source.last)
                    if last:
                        last_err = (yield dut.user_source.payload.error)
                        break
            got.append((bytes(beat), last_err))

        def collector():
            yield from collect()

        sim = Simulator(dut, {"sys": [drive(), collector()]},
                        clocks={"sys": 10, "eth_tx": 10, "eth_rx": 10})
        with sim:
            sim.run()

        self.assertEqual(len(got), 1)
        frame_got, err = got[0]
        self.assertEqual(frame_got, payload,
                         "payload (after preamble+FCS strip) differs")
        self.assertEqual(err, 1, "bad FCS not flagged")


if __name__ == "__main__":
    unittest.main(verbosity=2)
