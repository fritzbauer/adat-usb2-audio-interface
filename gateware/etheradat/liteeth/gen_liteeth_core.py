#!/usr/bin/env python3
"""Generate liteeth_rmii_mac.v — the RMII MAC for the EtherADAT bridge.

Wraps LiteEth's LiteEthMACCore (preamble insertion/checking, CRC-32
insert/check, minimum-length padding, inter-frame gap — the
battle-tested parts) around a minimal fixed-100-Mbit/s RMII shim
(rmii_shim.py). The result is a plain-synchronous-logic Verilog module
with a single 50 MHz clock:

    clk50        : RMII reference clock (drives sys/eth_tx/eth_rx)
    rst          : synchronous, active high (all three domains)
    rmii side    : rmii_tx_en, rmii_txd[1:0], rmii_crs_dv, rmii_rxd[1:0]
    TX user side : tx_data[7:0], tx_valid, tx_ready, tx_last
                   (frame bytes WITHOUT FCS; be=1/error=0 tied inside;
                   the core pads to 60 bytes, appends FCS, prepends
                   preamble, enforces IFG)
    RX user side : rx_data[7:0], rx_valid, rx_ready, rx_last, rx_error
                   (frames WITHOUT preamble/SFD and WITHOUT FCS; error
                   flagged on the last byte of bad-FCS frames)

This script needs migen, litex and liteeth — run it with a scratch
venv (NOT the adatface project venv), e.g.:

    python3 -m venv /tmp/etheradat-scratch/liteenv
    /tmp/etheradat-scratch/liteenv/bin/pip install migen liteeth
    /tmp/etheradat-scratch/liteenv/bin/python gen_liteeth_core.py

and check the resulting liteeth_rmii_mac.v into the repository (the
adatface build itself does not need migen/litex).
"""

import os
import sys

from migen import *
from migen.fhdl.verilog import convert

from litex.gen import LiteXModule
from litex.soc.interconnect import stream

from liteeth.common import eth_phy_description
from liteeth.mac.core import LiteEthMACCore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmii_shim import LiteEthRMIIShim


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

        # tie the unused user-side fields and wire the user streams to
        # the core (the core wires core.sink -> ... -> phy.sink and
        # phy.source -> ... -> core.source itself via its pipeline)
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
        # RMII pads
        dut.rmii_tx_en, dut.rmii_txd, dut.rmii_crs_dv, dut.rmii_rxd,
        # TX user side (design -> MAC core)
        dut.user_sink.payload.data, dut.user_sink.valid,
        dut.user_sink.ready, dut.user_sink.last,
        # RX user side (MAC core -> design)
        dut.user_source.payload.data, dut.user_source.valid,
        dut.user_source.ready, dut.user_source.last,
        dut.user_source.payload.error,
        # the extra clock domains used by the core (all wired to clk50)
        dut.cd_eth_tx.clk, dut.cd_eth_tx.rst,
        dut.cd_eth_rx.clk, dut.cd_eth_rx.rst,
    }

    out = os.path.join(here, "liteeth_rmii_mac.v")
    result = convert(dut, ios=ios, name="liteeth_rmii_mac")
    result.write(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
