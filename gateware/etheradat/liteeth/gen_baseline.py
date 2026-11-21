#!/usr/bin/env python3
"""Generate liteeth_baseline.v — a bring-up baseline for the EtherADAT
RMII wiring: STOCK LiteEth IP/UDP stack (MAC core + ARP + IP + ICMP +
UDP) on top of the checked-in RMII shim, with a UDP echo on port 1234.

Purpose: an independent test that does NOT contain any EtherADAT RTL.
If the PC can exchange UDP datagrams with this design, PHY, RMII wiring
and the LAN8720 module are proven good, and any remaining problem is in
the EtherADAT bridge. (Wireshark will show ARP/ping/UDP frames; ping
192.168.42.2 works because ICMP is enabled.)

Layout (single 50 MHz clock):

    clk50          : RMII reference clock. The LAN8720 module is
                     strapped REF_CLK-OUT and DRIVES the RMII REF_CLK
                     line, so REF_CLK is an FPGA input. The wrapper
                     (etheradat_baseline.py) feeds that 50 MHz to
                     sys_clk, eth_tx_clk and eth_rx_clk.
    rst            : synchronous active high, all domains
    rmii pads      : txd[1:0], tx_en, rxd[1:0], crs_dv
    dbg            : crs_dv, rx_udp_valid, rx_mac_valid, tx_mac_valid

Needs migen/litex/liteeth (scratch venv, NOT the adatface venv):

    python3 -m venv /tmp/etheradat-scratch/liteenv
    /tmp/etheradat-scratch/liteenv/bin/pip install migen litex liteeth
    /tmp/etheradat-scratch/liteenv/bin/python gen_baseline.py

Pinned commits (21.11.2026, all git master): migen @ e19524c963a8,
litex @ 6438a25a060b, liteeth @ a54c730c323f. Do NOT use PyPI's
migen 0.10.0 / litex+liteeth 2024.12 releases: liteeth 2024.12 still
has last_be instead of native `be` payload fields and breaks the shim.
For simulations, migen additionally needs the local patch
gateware/patches/migen-e19524c-sim-write-only-port.patch (its
simulator crashes on write-only memory ports of its own SyncFIFO).

The amaranth-side wrapper is etheradat/etheradat_baseline.py.
"""

import os
import sys

from migen import *
from migen.fhdl.verilog import convert

from litex.gen import LiteXModule
from litex.soc.interconnect import stream

from liteeth.common import eth_phy_description, eth_udp_user_description
from liteeth.core import LiteEthUDPIPCore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmii_shim import LiteEthRMIIShim

UDP_ECHO_PORT = 1234


class BaselinePhy(LiteXModule):
    """RMII shim exposed as a liteeth 'phy' (sink = TX bytes in, source =
    RX bytes out). NO MAC core here: LiteEthIPCore instantiates its own
    LiteEthMAC (with preamble/CRC/padding) around this phy — a second
    core here would strip the preamble/FCS before the stack's core
    expects to see them (found via sim 18.11.2026)."""

    dw = 8

    def __init__(self, rmii_tx_en, rmii_txd, rmii_crs_dv, rmii_rxd):
        # The eth_tx/eth_rx clock domains are defined ONCE, by Baseline.
        # LiteEthMAC's CDC expects those domain names; do NOT re-declare
        # them here (migen merges same-named domains, but a single owner
        # keeps the generated Verilog unambiguous).
        self.sink = stream.Endpoint(eth_phy_description(8))
        self.source = stream.Endpoint(eth_phy_description(8))

        phy = LiteEthRMIIShim(tx_en=rmii_tx_en, txd=rmii_txd,
                              crs_dv=rmii_crs_dv, rxd=rmii_rxd)
        self.submodules.shim = phy

        # be/error need no extra wiring: the MAC core drives be/error on
        # its tx path itself, and the RX shim's RMIIRXShim already sets
        # source.be = 1 (error stays 0).
        self.comb += [
            self.sink.connect(phy.sink),
            phy.source.connect(self.source),
        ]


class Baseline(LiteXModule):
    """Stock LiteEth IP/UDP stack + UDP echo, RMII pads on top."""

    def __init__(self, mac_address=0x02AD6E657401, ip_address="192.168.42.2"):
        # RMII pads. No MDC/MDIO: the LAN8720 runs from its power-up
        # straps (RMII mode, REF_CLK-OUT), nothing here configures it.
        self.rmii_tx_en = Signal()
        self.rmii_txd = Signal(2)
        self.rmii_crs_dv = Signal()
        self.rmii_rxd = Signal(2)

        # debug outputs for LEDs
        self.dbg_crs_dv = Signal()
        self.dbg_rx_udp = Signal()      # UDP payload byte arriving
        self.dbg_rx_mac = Signal()      # MAC frame byte arriving
        self.dbg_tx_mac = Signal()      # MAC frame byte leaving

        # clock domains (all on clk50)
        self.clock_domains.cd_sys = ClockDomain()
        self.clock_domains.cd_eth_tx = ClockDomain()
        self.clock_domains.cd_eth_rx = ClockDomain()

        phy = BaselinePhy(self.rmii_tx_en, self.rmii_txd,
                          self.rmii_crs_dv, self.rmii_rxd)
        self.submodules.phy = phy

        core = LiteEthUDPIPCore(phy=phy,
                                mac_address=mac_address,
                                ip_address=ip_address,
                                clk_freq=50_000_000,
                                with_icmp=True,
                                with_sys_datapath=True)
        self.submodules.core = core

        # UDP echo: everything sent to UDP port 1234 comes back.
        # The loop NEEDS a packet buffer: with dw=8 in the sys domain the
        # UDP crossbar installs no per-port FIFOs ("slower" is False), so
        # an unbuffered loop blocks the serial MAC crossbar on the first
        # datagram until its echo can be sent — and that echo waits for
        # an ARP reply that can never pass the blocked crossbar. The
        # FIFO takes the whole payload off the shared path instead
        # (same pattern as LiteEth's own UDP frontends, which always
        # buffer the RX side).
        port = core.udp.crossbar.get_port(udp_port=UDP_ECHO_PORT, dw=8)
        self.echo_buffer = stream.SyncFIFO(
            eth_udp_user_description(8), depth=256, buffered=True)
        self.comb += [
            port.source.connect(self.echo_buffer.sink),
            # reply: source port = 1234, destination port = the sender's
            # port (copied straight through, the reply would carry the
            # peer's source port as its own and no UDP stack would
            # accept it as a reply).
            self.echo_buffer.source.connect(port.sink,
                                            omit={"src_port", "dst_port"}),
            port.sink.src_port.eq(UDP_ECHO_PORT),
            port.sink.dst_port.eq(self.echo_buffer.source.src_port),
            self.dbg_crs_dv.eq(self.rmii_crs_dv),
            self.dbg_rx_udp.eq(port.sink.valid),
            self.dbg_rx_mac.eq(phy.source.valid),
            self.dbg_tx_mac.eq(phy.sink.valid),
        ]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    dut = Baseline()

    ios = {
        dut.rmii_tx_en, dut.rmii_txd, dut.rmii_crs_dv, dut.rmii_rxd,
        dut.dbg_crs_dv, dut.dbg_rx_udp, dut.dbg_rx_mac, dut.dbg_tx_mac,
        dut.cd_sys.clk, dut.cd_sys.rst,
        dut.cd_eth_tx.clk, dut.cd_eth_tx.rst,
        dut.cd_eth_rx.clk, dut.cd_eth_rx.rst,
    }

    out = os.path.join(here, "liteeth_baseline.v")
    result = convert(dut, ios=ios, name="liteeth_baseline")
    result.write(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
