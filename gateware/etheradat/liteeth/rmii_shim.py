"""Minimal RMII <-> 8-bit stream shim, used as the "PHY" for LiteEth's
LiteEthMACCore when generating the Verilog for the EtherADAT bridge.

The structure follows LiteEth's own LiteEthPHYRMIITX/LiteEthPHYRMIIRX
(enjoy-digital/liteeth, liteeth/phy/parallel/rmii.py) but:
- fixed 100 Mbit/s (no speed detection / 10 Mbit timer),
- no vendor SDR I/O primitives: the RMII signals are synchronous to the
  50 MHz reference clock this logic runs in, so plain register sampling
  and driving is exactly right,
- the 8<->2-bit conversion is hand-rolled: a straightforward serializer
  and a delayed byte assembler. (litex's stream.Converter(8, 2) proved
  to garble beats in the version installed here.)
- the pads are plain Signals (exposed as top-level ports by the
  generator), not platform resources.

TX contract: the sink must offer the bytes of a frame gaplessly (the
MAC core's store-and-forward packet FIFO guarantees this); TX_EN then
covers the whole frame.

RX contract: dibits are captured from CRS_DV rise (with non-null data)
to two CRS_DV-low cycles; the byte stream starts with the preamble (the
MAC core's preamble checker strips it) and the last byte carries
`last`.

This module is imported ONLY by the Verilog generators
(gen_etheradat_mac.py, gen_baseline.py) and their tests, in
an environment that has migen/litex/liteeth installed (NOT the
adatface project venv).
"""

from migen import *
from litex.gen import *
from litex.soc.interconnect import stream

from liteeth.common import eth_phy_description


class RMIITXShim(LiteXModule):
    """Serializes 8-bit sink beats into RMII dibits (100 Mbit/s).

    The MAC core hands us the complete frame including preamble and
    SFD (preamble inserter) and including the FCS (CRC32 inserter);
    TX_EN covers exactly those bytes, LSB-first dibit pairs on
    TXD1/TXD0. A byte is accepted on the last dibit cycle of the
    previous one, so gapless beats keep TX_EN high continuously.
    """

    def __init__(self, tx_en, txd):
        self.sink = sink = stream.Endpoint(eth_phy_description(8))

        # 4-dibit shift register + remaining-count FSM (22.11.2026).
        # The old pair/emitting version dropped the first and last dibit
        # of every byte: `tx_en` fell one cycle before the final dibit
        # was driven, so no valid Ethernet frame reached the PHY.
        shift = Signal(8)
        count = Signal(3)   # 0 = idle, 4..1 = dibits still to send

        self.comb += [
            txd.eq(shift[0:2]),       # registered data, LSB dibit first
            tx_en.eq(count != 0),     # registered enable
            # Accept the next byte on the last dibit of the current one
            # so frames are gapless while TX_EN stays high.
            sink.ready.eq(count <= 1),
        ]
        self.sync += [
            If(sink.valid & sink.ready,
               shift.eq(sink.payload.data),
               count.eq(4),
            ).Elif(count != 0,
                   shift.eq(shift >> 2),
                   count.eq(count - 1),
            ),
        ]


class RMIIRXShim(LiteXModule):
    """Captures RMII dibits into 8-bit source beats (100 Mbit/s).

    The dibits run through a fixed 2-cycle pipeline (LiteEth's "delay")
    so that the end-of-frame condition (two consecutive CRS_DV-low
    cycles, `crs_last`) lands exactly on the cycle at which the last
    data dibit is consumed by the byte assembler: the last byte's beat
    then carries `last` from the start.

    SFD realignment (found on hardware 21.11.2026): the LAN8720 asserts
    CRS_DV one or two cycles AFTER the first preamble dibits are driven
    (carrier detection is asynchronous, the data path is not), so the
    byte assembly would start mid-preamble, the SFD byte would land
    misaligned (0xD7 instead of 0xD5). Fix: scan the dibit stream for
    the SFD pattern (01,01,01,11) and resume capturing at the dibit
    right after it. The number of missed leading dibits no longer
    matters.

    Update 22.11.2026: the shim now emits ONLY frame bytes (dst MAC
    onwards), no synthetic preamble. LiteEth's preamble checker passed
    the preamble through on this Cyclone IV build (verified by UART
    byte taps), shifting the MAC/ARP headers by one byte; the baseline
    disables the MAC preamble/CRC stages (`with_preamble_crc=False`),
    so the stack expects the frame to start at the destination MAC.
    """

    MODE_SEARCH = 0
    MODE_PRE = 1
    MODE_DATA = 2

    def __init__(self, crs_dv_i, rxd_i):
        self.source = source = stream.Endpoint(eth_phy_description(8))
        self.comb += source.be.eq(1)

        crs_dv = Signal()
        rx_data = Signal(2)
        self.comb += [
            crs_dv.eq(crs_dv_i),
            rx_data.eq(rxd_i),
        ]

        # frame delimitation (LiteEth's RMII RX)
        crs_first = Signal()
        crs_last = Signal()
        crs_run = Signal()
        crs_dv_d = Signal()
        self.comb += [
            crs_first.eq(crs_dv & (rx_data != 0b00)),
            crs_last.eq(~crs_dv & ~crs_dv_d),
        ]
        self.sync += [
            crs_dv_d.eq(crs_dv),
            If(crs_first, crs_run.eq(1)),
            If(crs_last, crs_run.eq(0)),
        ]

        # 2-cycle dibit pipeline with capture tracking: the dibit seen
        # at cycle C is presented (with its capture flag) at C+2
        d0 = Signal(2)
        d1 = Signal(2)
        e0 = Signal()
        e1 = Signal()
        cap = Signal()
        self.comb += cap.eq(crs_first | crs_run)
        self.sync += [
            d0.eq(rx_data),
            d1.eq(d0),
            e0.eq(cap),
            e1.eq(e0),
        ]

        mode = Signal(2, reset=self.MODE_SEARCH)

        # 2-entry byte queue between the assembler and the source: the
        # synthetic preamble drains while the real dst-MAC bytes keep
        # arriving (they complete at one byte per 4 dibit cycles)
        f0_valid = Signal()
        f0_data = Signal(8)
        f0_last = Signal()
        f1_valid = Signal()
        f1_data = Signal(8)
        f1_last = Signal()

        # SFD scanner (SEARCH mode): last 4 captured dibits, newest pair
        # at [1:0]; the SFD pattern is (01,01,01,11) = 0x57
        sreg = Signal(8, reset=0)     # reset to a non-matching pattern
        self.sync += If(e1 & (mode == self.MODE_SEARCH),
                        sreg.eq(Cat(d1, sreg[0:6])))
        sfd_hit = Signal()
        # sreg holds the dibits presented BEFORE the current d1 (they
        # shift in at the end of each cycle), so the pattern check must
        # combine sreg[5:0] (the three dibits before the SFD close) with
        # the current d1: firing one cycle later would eat the first
        # dst-MAC dibit into the scanner.
        self.comb += sfd_hit.eq(e1 & (mode == self.MODE_SEARCH) &
                                (sreg[:6] == 0b01_01_01) & (d1 == 0b11))

        # Synthetic preamble (7x 0x55 + SFD) for the stock MAC framing.
        pre_cnt = Signal(3)
        pre_byte = Signal(8)
        self.comb += [
            pre_byte.eq(Mux(pre_cnt == 7, 0xD5, 0x55)),
            source.valid.eq((mode == self.MODE_PRE) |
                            ((mode == self.MODE_DATA) & f0_valid)),
            source.payload.data.eq(Mux(mode == self.MODE_PRE, pre_byte,
                                       f0_data)),
            source.last.eq((mode == self.MODE_DATA) & f0_last),
        ]

        # byte assembler (PRE and DATA modes): 4 delayed dibits -> 1
        # byte (first dibit in the LSB pair)
        shift = Signal(8)
        bits = Signal(2)               # dibits collected 0..3
        consume = Signal()
        self.comb += consume.eq(source.valid & source.ready)
        new_byte = Signal(8)
        self.comb += new_byte.eq(Cat(shift[2:8], d1))
        pushing = Signal()
        self.comb += pushing.eq((mode != self.MODE_SEARCH) & e1 &
                                (bits == 3))

        self.sync += [
            # mode FSM
            If(mode == self.MODE_SEARCH,
               If(sfd_hit,
                  mode.eq(self.MODE_PRE),
                  pre_cnt.eq(0),
                  bits.eq(0),
               ),
            ).Elif(mode == self.MODE_PRE,
               If(consume,
                  If(pre_cnt == 7,
                     mode.eq(self.MODE_DATA),
                  ).Else(
                     pre_cnt.eq(pre_cnt + 1),
                  ),
               ),
            ),

            # byte queue: pop on consume (DATA mode), push on completion
            If((mode == self.MODE_DATA) & consume,
               If(pushing & ~f1_valid,
                  # Only one byte queued: the new byte goes straight into
                  # f0. Advancing a stale f1_data would emit the byte we
                  # just consumed a second time on hardware (one extra
                  # byte after the 8th byte of an ICMP/UDP header).
                  f0_valid.eq(1),
                  f0_data.eq(new_byte),
                  f0_last.eq(crs_last),
                  f1_valid.eq(0),
               ).Elif(pushing,
                  # queue full and draining: f1 advances to f0, the new
                  # byte takes f1's slot
                  f0_valid.eq(1),
                  f0_data.eq(f1_data),
                  f0_last.eq(f1_last),
                  f1_valid.eq(1),
                  f1_data.eq(new_byte),
                  f1_last.eq(crs_last),
               ).Else(
                  f0_valid.eq(f1_valid),
                  f0_data.eq(f1_data),
                  f0_last.eq(f1_last),
                  f1_valid.eq(0),
               ),
            ).Elif(pushing & ~((mode == self.MODE_DATA) & f0_valid & f1_valid),
               If(~f0_valid,
                  f0_valid.eq(1),
                  f0_data.eq(new_byte),
                  f0_last.eq(crs_last),
               ).Elif(~f1_valid,
                  f1_valid.eq(1),
                  f1_data.eq(new_byte),
                  f1_last.eq(crs_last),
               ),
            ),  # else: queue overflow, byte dropped

            # dibit assembly runs in PRE and DATA modes (the dst MAC
            # dibits already arrive while the preamble is synthetic)
            If((mode != self.MODE_SEARCH) & e1,
               # the new dibit lands in the high pair and the older
               # ones shift down: after 4 dibits the first-received
               # pair sits in the LSB pair (LSB-first byte)
               shift.eq(new_byte),
               If(bits == 3,
                  bits.eq(0),
               ).Else(
                  bits.eq(bits + 1),
               ),
            ).Elif(~e1,
               # an incomplete byte at frame end is dropped
               bits.eq(0),
            ),

            # Re-arm the SFD scanner after the last byte has been taken.
            # Without this the shim stays in DATA mode and emits the next
            # frame's preamble as payload (observed on HW 22.11.2026).
            # Last in the list so it overrides the queue/assembler above.
            If(consume & source.last,
               mode.eq(self.MODE_SEARCH),
               sreg.eq(0),
               bits.eq(0),
               f0_valid.eq(0),
               f1_valid.eq(0),
            ),
        ]




class LiteEthRMIIShim(LiteXModule):
    """The complete shim (TX in eth_tx, RX in eth_rx domain).

    The pad signals are driven/read directly: tx_en/txd by the TX shim,
    crs_dv/rxd sampled by the RX shim.
    """

    dw = 8

    def __init__(self, tx_en, txd, crs_dv, rxd):
        self.submodules.tx = ClockDomainsRenamer("eth_tx")(
            RMIITXShim(tx_en, txd))
        self.submodules.rx = ClockDomainsRenamer("eth_rx")(
            RMIIRXShim(crs_dv, rxd))

        self.sink = self.tx.sink
        self.source = self.rx.source
