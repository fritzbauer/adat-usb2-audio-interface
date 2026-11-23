#!/usr/bin/env python3
"""Generate liteeth_etheradat_mac.v — the LiteEth MAC for the EtherADAT
bridge (3 ADAT inputs / 2 ADAT outputs over raw Ethernet).

This is the MAC-only sibling of gen_baseline.py: the *same* proven RMII
shim and clock-domain-crossing fix, but the MAC core's user streams are
exposed instead of a UDP/IP stack.  The EtherADAT bridge (Amaranth)
speaks the LiteEth MAC user-stream contract:

    TX sink:   frame bytes WITHOUT preamble/SFD and WITHOUT FCS, `last`
               on the last payload byte. The core pads to the Ethernet
               minimum, appends FCS, inserts the preamble and IFG.
    RX source: frame bytes WITHOUT preamble/SFD/FCS; `last` on the final
               payload byte; `error` set on the last byte of bad frames.

Hardware fixes carried over from gen_baseline.py (verified 21./22.11.2026,
see ETHERADAT_STATUS_20261122.md):
- RMII RX shim scans for the SFD (CRS_DV asserts late on the LAN8720)
  and re-arms after each frame; it synthesizes the 8 preamble bytes for
  the MAC preamble checker.
- LiteEthMACCDC's AsyncFIFO lost `last` on this Cyclone IV build.  sys
  and eth_rx/eth_tx share one physical REF_CLK pin here, so the CDC is
  a plain register slice (SameClockRXCDC).

Needs migen/litex/liteeth — run with the scratch venv:

    /tmp/etheradat-scratch/liteenv/bin/python gen_etheradat_mac.py
"""

import os
import sys

from migen import *
from migen.fhdl.verilog import convert

from litex.gen import LiteXModule
from litex.soc.interconnect import stream

from liteeth.common import eth_phy_description
from liteeth.mac.core import LiteEthMACCore
import liteeth.mac.core as _liteeth_mac_core

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmii_shim import LiteEthRMIIShim


_StockCDC = _liteeth_mac_core.LiteEthMACCDC


class SameClockRXCDC(LiteXModule):
    """sys/eth_rx/eth_tx are one physical clock: a register slice.

    Identical to the class in gen_baseline.py (kept in sync on purpose;
    both generators must carry the same hardware fix).
    """

    def __init__(self, layout, cd_from, cd_to, depth, buffered):
        if (cd_from, cd_to) not in (("eth_rx", "sys"), ("sys", "eth_tx")):
            stock = _StockCDC(layout, cd_from, cd_to, depth, buffered)
            self.submodules.stock = stock
            self.sink = stock.sink
            self.source = stock.source
            self.ext_rst_to = stock.ext_rst_to
            return
        self.sink = stream.Endpoint(layout)
        self.source = stream.Endpoint(layout)
        self.ext_rst_to = Signal()
        buf = stream.Buffer(layout)
        self.submodules.buf = ClockDomainsRenamer("eth_rx")(buf)
        self.comb += [
            self.sink.connect(buf.sink),
            buf.source.connect(self.source),
        ]


_liteeth_mac_core.LiteEthMACCDC = SameClockRXCDC


class EtherADATMAC(LiteXModule):
    """LiteEth MAC core + RMII shim, single-clock (50 MHz) wrapper."""

    def __init__(self):
        # top-level RMII pads
        self.rmii_tx_en = Signal()
        self.rmii_txd = Signal(2)
        self.rmii_crs_dv = Signal()
        self.rmii_rxd = Signal(2)

        # extra clock domains used by the core (all on clk50)
        self.clock_domains.cd_eth_tx = ClockDomain()
        self.clock_domains.cd_eth_rx = ClockDomain()

        # user-side streams (sys domain)
        self.user_sink = stream.Endpoint(eth_phy_description(8))
        self.user_source = stream.Endpoint(eth_phy_description(8))

        phy = LiteEthRMIIShim(tx_en=self.rmii_tx_en, txd=self.rmii_txd,
                              crs_dv=self.rmii_crs_dv, rxd=self.rmii_rxd)
        self.submodules.shim = phy

        core = LiteEthMACCore(phy=phy, dw=8,
                              with_sys_datapath=True,
                              with_preamble_crc=True,
                              with_padding=True,
                              with_store_and_forward=True)
        self.submodules.core = core

        self.comb += [
            self.user_sink.be.eq(1),
            self.user_sink.error.eq(0),
            self.user_sink.connect(core.sink),
            core.source.connect(self.user_source),
        ]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    dut = EtherADATMAC()

    ios = {
        dut.rmii_tx_en, dut.rmii_txd, dut.rmii_crs_dv, dut.rmii_rxd,
        dut.user_sink.payload.data, dut.user_sink.valid,
        dut.user_sink.ready, dut.user_sink.last,
        dut.user_source.payload.data, dut.user_source.valid,
        dut.user_source.ready, dut.user_source.last,
        dut.user_source.payload.error,
        dut.cd_eth_tx.clk, dut.cd_eth_tx.rst,
        dut.cd_eth_rx.clk, dut.cd_eth_rx.rst,
    }

    out = os.path.join(here, "liteeth_etheradat_mac.v")
    result = convert(dut, ios=ios, name="liteeth_etheradat_mac")
    result.write(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
