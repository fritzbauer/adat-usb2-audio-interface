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

This module is imported ONLY by gen_liteeth_core.py and its tests, in
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

        byte = Signal(8)
        pair = Signal(2)               # dibit pair index of the current byte
        emitting = Signal()

        self.comb += [
            # dibit pair `pair` = byte bits (2*pair+1 : 2*pair), LSB first
            txd.eq(byte >> (pair << 1)),
            tx_en.eq(emitting),
            # accept the first byte while idle, the next one on the
            # last dibit cycle of the current one
            sink.ready.eq(~emitting | (pair == 3)),
        ]
        self.sync += [
            If(sink.valid & sink.ready,
               # a byte is consumed at this edge: latch it and start
               # (or continue) the serialization seamlessly
               byte.eq(sink.payload.data),
               pair.eq(0),
               emitting.eq(1),
            ).Elif(emitting,
               pair.eq(pair + 1),
               If(pair == 3,
                  emitting.eq(0),
               ),
            ),
        ]


class RMIIRXShim(LiteXModule):
    """Captures RMII dibits into 8-bit source beats (100 Mbit/s).

    The dibits run through a fixed 2-cycle pipeline (LiteEth's "delay")
    so that the end-of-frame condition (two consecutive CRS_DV-low
    cycles, `crs_last`) lands exactly on the cycle at which the last
    data dibit is consumed by the byte assembler: the last byte's beat
    then carries `last` from the start.
    """

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

        # byte assembler: 4 delayed dibits -> 1 byte (first dibit in the
        # LSB pair), one-beat skid so the consumer's ready cannot
        # overwrite a held byte
        shift = Signal(8)
        bits = Signal(2)               # dibits collected 0..3
        beat_valid = Signal()
        beat_last = Signal()
        consume = Signal()
        self.comb += [
            consume.eq(source.valid & source.ready),
            source.valid.eq(beat_valid),
            source.payload.data.eq(shift),
            source.last.eq(beat_last),
        ]
        self.sync += [
            # an incomplete byte at frame end is dropped; a held
            # complete byte waits for the consumer
            If(~e1 & (~beat_valid | consume),
               beat_valid.eq(0),
               beat_last.eq(0),
            ),
            If(~e1,
               bits.eq(0),
            ).Elif(consume | ~beat_valid,
               # the new dibit lands in the high pair and the older
               # ones shift down: after 4 dibits the first-received
               # pair sits in the LSB pair (LSB-first byte)
               shift.eq(Cat(shift[2:8], d1)),
               bits.eq(bits + 1),
               beat_valid.eq(bits == 3),
               beat_last.eq(crs_last),   # aligned with the last dibit
               If(bits == 3,
                  bits.eq(0),
               ),
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
