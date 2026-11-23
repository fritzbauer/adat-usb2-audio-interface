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

        # debug: short-frame pulses at the user sink and at the PHY sink
        # (bring-up: the AdatNet hello frames are 34 bytes and must be
        # padded; these show where they disappear, if they do)
        self.dbg_sink_frame = Signal()
        self.dbg_sink_short = Signal()
        self.dbg_phy_frame = Signal()
        self.dbg_phy_short = Signal()

        phy = LiteEthRMIIShim(tx_en=self.rmii_tx_en, txd=self.rmii_txd,
                              crs_dv=self.rmii_crs_dv, rxd=self.rmii_rxd)
        self.submodules.shim = phy

        core = LiteEthMACCore(phy=phy, dw=8,
                              with_sys_datapath=True,
                              with_preamble_crc=True,
                              with_padding=True,
                              with_store_and_forward=False)
        self.submodules.core = core

        self.comb += [
            self.user_sink.be.eq(1),
            self.user_sink.error.eq(0),
            self.user_sink.connect(core.sink),
            core.source.connect(self.user_source),
        ]

        # --- short-frame diagnostics -----------------------------------
        sink_len = Signal(16)
        phy_len = Signal(16)
        sink_acc = self.user_sink.valid & self.user_sink.ready
        phy_acc = phy.sink.valid & phy.sink.ready
        self.sync.eth_tx += [
            If(sink_acc,
               If(self.user_sink.last, sink_len.eq(0))
               .Else(sink_len.eq(sink_len + 1))),
            If(phy_acc,
               If(phy.sink.last, phy_len.eq(0))
               .Else(phy_len.eq(phy_len + 1))),
        ]
        self.comb += [
            self.dbg_sink_frame.eq(self.user_sink.valid
                                   & self.user_sink.ready
                                   & self.user_sink.last),
            self.dbg_sink_short.eq(self.user_sink.valid
                                   & self.user_sink.ready
                                   & self.user_sink.last
                                   & (sink_len < 100)),
            self.dbg_phy_frame.eq(phy.sink.valid & phy.sink.ready
                                  & phy.sink.last),
            self.dbg_phy_short.eq(phy.sink.valid & phy.sink.ready
                                  & phy.sink.last & (phy_len < 100)),
        ]

        # --- TX line tap (bring-up) -------------------------------------
        # Capture the first 16 bytes of every frame AS THEY ENTER THE RMII
        # SERIALIZER (i.e. after preamble/CRC insertion), plus the frame
        # length and a frame counter. Comparing this against the bridge's
        # user-stream bytes shows whether the MAC (or the bridge) drops
        # any leading header bytes. Sampled in eth_tx.
        self.dbg_cap = [Signal(8, name="dbg_cap%d" % i)
                        for i in range(16)]
        self.dbg_cap_len = Signal(16)
        self.dbg_cap_count = Signal(16)
        cap_arr = Array(self.dbg_cap)
        cap_idx = Signal(5)          # write slot 0..15, then hold
        cap_n = Signal(16)           # bytes in the current frame
        line_acc = Signal()
        self.comb += line_acc.eq(phy.sink.valid & phy.sink.ready)
        self.sync.eth_tx += [
            If(line_acc,
                If(cap_idx < 16,
                    cap_arr[cap_idx[:4]].eq(phy.sink.payload.data),
                ),
                If(phy.sink.last,
                    self.dbg_cap_len.eq(cap_n + 1),
                    self.dbg_cap_count.eq(self.dbg_cap_count + 1),
                    cap_n.eq(0),
                    cap_idx.eq(0),
                ).Else(
                    cap_n.eq(cap_n + 1),
                    If(cap_idx < 16,
                        cap_idx.eq(cap_idx + 1),
                    ),
                ),
            ),
        ]

        # --- RMII PIN monitor (bring-up) --------------------------------
        # Reassemble the serialized TX dibits EXACTLY as driven to the PHY
        # (rmii_tx_en / rmii_txd pads): first 16 wire bytes + length/count.
        # If this matches dbg_cap (serializer input), the FPGA pins are
        # correct and any remaining corruption is on the PHY/cable side.
        self.dbg_pin = [Signal(8, name="dbg_pin%d" % i)
                        for i in range(16)]
        self.dbg_pin_len = Signal(16)
        self.dbg_pin_count = Signal(16)
        pin_arr = Array(self.dbg_pin)
        pin_bits = Signal(2)
        pin_byte = Signal(8)
        pin_idx = Signal(5)
        pin_n = Signal(16)
        pin_active = Signal()
        pin_byte_now = Signal(8)
        self.comb += pin_byte_now.eq(
            pin_byte | (self.rmii_txd << (pin_bits * 2)))
        self.sync.eth_tx += [
            If(self.rmii_tx_en,
                pin_active.eq(1),
                If(pin_bits == 3,
                    pin_bits.eq(0),
                    pin_byte.eq(0),
                    pin_n.eq(pin_n + 1),
                    If(pin_idx < 16,
                        pin_arr[pin_idx[:4]].eq(pin_byte_now),
                        pin_idx.eq(pin_idx + 1),
                    ),
                ).Else(
                    pin_bits.eq(pin_bits + 1),
                    pin_byte.eq(pin_byte_now),
                ),
            ).Else(
                pin_bits.eq(0),
                pin_byte.eq(0),
                If(pin_active,
                    self.dbg_pin_len.eq(pin_n),
                    self.dbg_pin_count.eq(self.dbg_pin_count + 1),
                    pin_n.eq(0),
                    pin_idx.eq(0),
                    pin_active.eq(0),
                ),
            ),
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
        dut.dbg_sink_frame, dut.dbg_sink_short,
        dut.dbg_phy_frame, dut.dbg_phy_short,
        dut.cd_eth_tx.clk, dut.cd_eth_tx.rst,
        dut.cd_eth_rx.clk, dut.cd_eth_rx.rst,
    }
    ios |= set(dut.dbg_cap)
    ios |= {dut.dbg_cap_len, dut.dbg_cap_count}
    ios |= set(dut.dbg_pin)
    ios |= {dut.dbg_pin_len, dut.dbg_pin_count}

    out = os.path.join(here, "liteeth_etheradat_mac.v")
    result = convert(dut, ios=ios, name="liteeth_etheradat_mac")
    result.write(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
