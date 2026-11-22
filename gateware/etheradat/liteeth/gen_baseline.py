#!/usr/bin/env python3
"""Generate liteeth_baseline.v — stock LiteEth IP/UDP stack over the
EtherADAT RMII shim, UDP echo on port 1234 (bring-up design).

Goal: ping 192.168.42.2 / UDP echo port 1234 from 192.168.42.1 over the
LAN8720 module. Once this works, PHY/RMII/wiring are proven good and any
remaining EtherADAT problem is in the bridge itself.

Hardware fixes found 21./22.11.2026 (all verified with UART byte dumps):

* RX sampling on the plain REF_CLK edge; RMIIRXShim scans for the SFD
  because the LAN8720 asserts CRS_DV 1-2 dibits late, and re-arms its
  scanner after every frame (otherwise the next frame's preamble is
  emitted as payload).
* `LiteEthMACCDC`'s AsyncFIFO lost `last` on this Cyclone IV build.
  eth_rx and sys share one REF_CLK pin, so SameClockRXCDC uses a
  register slice; eth_tx (inverted clock) keeps the stock CDC.
* LiteX's Depacketizer shifted the MAC/ARP header by one byte because
  `sr_shift` is registered (registered value used in the same edge).
  Local FixedDepacketizer makes it combinational; the MAC stage emits
  one leading dummy byte and ARP skips it (crossbar beat alignment).
* The UDP echo needs a SyncFIFO (dw=8, sys domain: no per-port FIFOs).
* TX: a 4-dibit shift-register shim with gapless byte handover; the
  baseline wrapper connects o_rmii_tx_en/o_rmii_txd to the pads.

Versions: migen e19524c, litex 6438a25, liteeth a54c730 (git master).
Scratch venv: /tmp/etheradat-scratch/liteenv. For sims apply
gateware/patches/migen-e19524c-sim-write-only-port.patch.
"""

import os
import sys

from migen import *
from migen.fhdl.verilog import convert

from litex.gen import LiteXModule
from litex.soc.interconnect import stream

from liteeth.common import (eth_phy_description, eth_udp_user_description,
                            eth_mac_description, eth_arp_description,
                            eth_ipv4_description, eth_ipv4_user_description,
                            eth_icmp_description, eth_udp_description,
                            mac_header, arp_header, ipv4_header, icmp_header,
                            udp_header)
from liteeth.core import LiteEthUDPIPCore
from liteeth.mac.common import LiteEthMACDepacketizer
from liteeth.core.arp import LiteEthARPDepacketizer
from liteeth.core.ip import LiteEthIPV4Depacketizer
from liteeth.core.icmp import LiteEthICMPDepacketizer
from liteeth.core.udp import LiteEthUDPDepacketizer
import liteeth.mac.core as _liteeth_mac_core

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmii_shim import LiteEthRMIIShim
from gen_uartprobe import UARTTX
from fixed_depacketizer import FixedDepacketizer

UDP_ECHO_PORT = 1234


class EventCounter(LiteXModule):
    """8-bit saturating debug counter."""

    def __init__(self, event):
        self.value = Signal(8)
        self.sync += If(event,
                        If(self.value != 0xFF, self.value.eq(self.value + 1)))


_StockCDC = _liteeth_mac_core.LiteEthMACCDC


class SameClockRXCDC(LiteXModule):
    """eth_rx and sys are the same physical clock: no gray-pointer FIFO."""

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

LiteEthMACDepacketizer.__init__ = (
    lambda self, dw=8: FixedDepacketizer.__init__(
        self, eth_phy_description(dw), eth_mac_description(dw), mac_header))
LiteEthARPDepacketizer.__init__ = (
    lambda self, dw=8: FixedDepacketizer.__init__(
        self, eth_mac_description(dw), eth_arp_description(dw), arp_header))
LiteEthUDPDepacketizer.__init__ = (
    lambda self, dw=8: FixedDepacketizer.__init__(
        self, eth_ipv4_user_description(dw), eth_udp_description(dw),
        udp_header))
LiteEthICMPDepacketizer.__init__ = (
    lambda self, dw=8: FixedDepacketizer.__init__(
        self, eth_ipv4_user_description(dw), eth_icmp_description(dw),
        icmp_header))
LiteEthIPV4Depacketizer.__init__ = (
    lambda self, dw=8: FixedDepacketizer.__init__(
        self, eth_mac_description(dw), eth_ipv4_description(dw), ipv4_header))


class BaselinePhy(LiteXModule):
    dw = 8
    with_preamble_crc = True   # stock TX preamble + FCS, RX checking

    def __init__(self, rmii_tx_en, rmii_txd, rmii_crs_dv, rmii_rxd):
        self.sink = stream.Endpoint(eth_phy_description(8))
        self.source = stream.Endpoint(eth_phy_description(8))
        phy = LiteEthRMIIShim(tx_en=rmii_tx_en, txd=rmii_txd,
                              crs_dv=rmii_crs_dv, rxd=rmii_rxd)
        self.submodules.shim = phy
        self.comb += [
            self.sink.connect(phy.sink),
            phy.source.connect(self.source),
        ]


class Baseline(LiteXModule):
    def __init__(self, mac_address=0x02AD6E657401, ip_address="192.168.42.2"):
        self.rmii_tx_en = Signal()
        self.rmii_txd = Signal(2)
        self.rmii_crs_dv = Signal()
        self.rmii_rxd = Signal(2)

        # debug outputs (LEDs + small UART dump)
        for name in ("crs_dv", "rx_udp", "rx_mac", "tx_mac", "pre_err",
                     "crc_err", "arp_rx", "arp_req", "arp_rep", "mac_in",
                     "mac_out", "led_shim_last", "led_cdc_last",
                     "led_mac_last", "led_arp_req", "led_arp_rep",
                     "led_tx_last", "led_rx_err"):
            setattr(self, "dbg_" + name, Signal(name="dbg_" + name))

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

        port = core.udp.crossbar.get_port(udp_port=UDP_ECHO_PORT, dw=8)
        self.echo_buffer = stream.SyncFIFO(
            eth_udp_user_description(8), depth=256, buffered=True)
        self.comb += [
            port.source.connect(self.echo_buffer.sink),
            self.echo_buffer.source.connect(port.sink,
                                            omit={"src_port", "dst_port"}),
            port.sink.src_port.eq(UDP_ECHO_PORT),
            port.sink.dst_port.eq(self.echo_buffer.source.src_port),
        ]

        arp = core.arp
        dep = core.mac.depacketizer
        rx_pipe = core.mac.core.rx_datapath.pipeline
        cdc = rx_pipe[1]
        shim_last = phy.source.valid & phy.source.last
        cdc_last = cdc.source.valid & cdc.source.last
        dep_last = dep.sink.valid & dep.sink.last
        arp_req = arp.table.sink.valid & arp.table.sink.payload.request
        arp_rep = arp.table.source.valid
        tx_last = phy.sink.valid & phy.sink.last

        def stretch(name, event, cd="sys"):
            sig = Signal(name=name)
            hold = Signal(8)
            dst = self.sync.eth_rx if cd == "eth_rx" else self.sync
            dst += If(event, hold.eq(128)).Elif(hold != 0,
                                                hold.eq(hold - 1))
            self.comb += sig.eq(hold != 0)
            return sig

        self.comb += [
            self.dbg_crs_dv.eq(self.rmii_crs_dv),
            self.dbg_rx_mac.eq(phy.source.valid),
            self.dbg_tx_mac.eq(phy.sink.valid),
            self.dbg_rx_udp.eq(port.sink.valid),
            self.dbg_mac_in.eq(dep.sink.valid),
            self.dbg_mac_out.eq(dep.source.valid),
            self.dbg_arp_rx.eq(dep_last),
            self.dbg_arp_req.eq(arp_req),
            self.dbg_arp_rep.eq(arp_rep),
            self.dbg_pre_err.eq(0),
            self.dbg_crc_err.eq(0),
            self.dbg_led_shim_last.eq(
                stretch("dbg_led_shim_last", shim_last, "eth_rx")),
            self.dbg_led_cdc_last.eq(stretch("dbg_led_cdc_last", cdc_last)),
            self.dbg_led_mac_last.eq(stretch("dbg_led_mac_last", dep_last)),
            self.dbg_led_arp_req.eq(stretch("dbg_led_arp_req", arp_req)),
            self.dbg_led_arp_rep.eq(stretch("dbg_led_arp_rep", arp_rep)),
            self.dbg_led_tx_last.eq(stretch("dbg_led_tx_last", tx_last)),
            self.dbg_led_rx_err.eq(0),
        ]

        # Periodic UART dump: [0xAA][8 counters], ~170 ms apart:
        # shim_last, cdc_last, dep_last, arp_req, arp_rep, tx_last,
        # rx_bytes, tx_bytes.
        words = []
        for ev in (shim_last, cdc_last, dep_last, arp_req, arp_rep,
                   tx_last, phy.source.valid & phy.source.ready,
                   phy.sink.valid & phy.sink.ready):
            c = EventCounter(ev)
            self.submodules += c
            words.append(c.value)

        self.uart_txd = Signal()
        uart = UARTTX(self.uart_txd)
        self.submodules.uart_dump = uart

        timer = Signal(23, reset=(1 << 23) - 1)
        fire = Signal()
        self.sync += [
            If(fire, timer.eq((1 << 23) - 1)).Else(timer.eq(timer - 1)),
            fire.eq(timer == 0),
        ]
        idx = Signal(4)
        dumping = Signal()
        self.sync += [
            If(fire, dumping.eq(1), idx.eq(0)).Elif(
                dumping & ~uart.busy,
                If(idx == len(words) - 1,
                   dumping.eq(0),
                ).Else(idx.eq(idx + 1))),
        ]
        self.comb += [
            uart.strobe.eq(fire | (dumping & ~uart.busy)),
            uart.data.eq(Mux(fire, 0xAA, Array(words)[idx])),
        ]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    dut = Baseline()
    ios = {
        dut.rmii_tx_en, dut.rmii_txd, dut.rmii_crs_dv, dut.rmii_rxd,
        dut.dbg_crs_dv, dut.dbg_rx_udp, dut.dbg_rx_mac, dut.dbg_tx_mac,
        dut.dbg_pre_err, dut.dbg_crc_err, dut.dbg_arp_rx, dut.dbg_arp_req,
        dut.dbg_arp_rep, dut.dbg_mac_in, dut.dbg_mac_out,
        dut.dbg_led_shim_last, dut.dbg_led_cdc_last, dut.dbg_led_mac_last,
        dut.dbg_led_arp_req, dut.dbg_led_arp_rep, dut.dbg_led_tx_last,
        dut.dbg_led_rx_err,
        dut.cd_sys.clk, dut.cd_sys.rst, dut.uart_txd,
        dut.cd_eth_tx.clk, dut.cd_eth_tx.rst,
        dut.cd_eth_rx.clk, dut.cd_eth_rx.rst,
    }
    out = os.path.join(here, "liteeth_baseline.v")
    convert(dut, ios=ios, name="liteeth_baseline").write(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
