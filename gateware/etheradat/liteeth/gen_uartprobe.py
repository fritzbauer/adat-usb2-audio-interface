#!/usr/bin/env python3
"""Generate uart_probe.v — a UART diagnostic for the RMII shim.

Two RMIIRXShim instances sample the same RMII pads, one in the plain
REF_CLK domain ("eth_rx") and one in the inverted domain ("negrx"),
each capturing the emitted byte stream into a 128-byte buffer. After
every frame (source.last) the frame is queued for the UART TX.

UART format (115200 8N1, sys domain = plain REF_CLK):
    0xAA <marker> <len> <len bytes>
        marker 0x01 = plain-edge shim, 0x02 = negedge shim
        len   = frame length in bytes, capped at 128
    heartbeat: a single 0x7E roughly every 1.3 s (proves the UART path)

Run from the repo's liteeth directory with the scratch venv:
    python gen_uartprobe.py
"""

import os
import struct
import sys

from migen import *
from migen.fhdl.verilog import convert
from litex.gen import LiteXModule
from litex.soc.interconnect import stream

from liteeth.common import eth_phy_description

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmii_shim import RMIIRXShim, RMIITXShim


class SlowTXGenerator(LiteXModule):
    """TXGenerator with a ~100 ms gap: 72 bytes/s fits the UART."""

    def __init__(self):
        self.sink = sink = stream.Endpoint(eth_phy_description(8))
        # NOTE: this generator feeds the TX shim DIRECTLY (no MAC core
        # to add the FCS), so the FCS must be part of the 64 bytes.
        core = (bytes.fromhex("ffffffffffff") +
                bytes.fromhex("02ad6e657401") +
                struct.pack(">H", 0x9999) +
                b"TXGENTEST")
        core += bytes(60 - len(core))
        # the TX shim expects preamble + SFD + frame + FCS, gapless
        frame = bytes(7 * [0x55]) + b"\xd5" + core + eth_fcs(core)
        lut = Array(list(frame))
        idx = Signal(7)                # 72 bytes now
        gap = Signal(24)
        self.comb += [
            sink.valid.eq(gap == 0),
            sink.payload.data.eq(lut[idx]),
            sink.last.eq(idx == len(frame) - 1),
        ]
        self.sync += [
            If(sink.valid & sink.ready,
               If(idx == (len(frame) - 1),
                  idx.eq(0),
                  gap.eq(5000000 - len(frame)),  # ~100 ms at 50 MHz
               ).Else(
                  idx.eq(idx + 1),
               ),
            ).Elif(gap != 0,
               gap.eq(gap - 1),
            ),
        ]


def eth_fcs(frame):
    import struct as _struct
    crc = 0xFFFFFFFF
    for b in frame:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1))
    return _struct.pack("<I", crc ^ 0xFFFFFFFF)


class TXDibitPacker(LiteXModule):
    """Records the TX pads as bytes: 4 dibits per byte, LSB-first
    (the same order the TX shim serializes in), so the captured bytes
    map 1:1 onto the TX shim's input stream."""

    def __init__(self, tx_en, txd):
        self.source = source = stream.Endpoint(eth_phy_description(8))

        shift = Signal(8)
        bits = Signal(2)
        running = Signal()
        hold = Signal(8)
        hold_valid = Signal()
        hold_last = Signal()
        consume = Signal()
        new_byte = Signal(8)
        completing = Signal()
        flushing = Signal()
        self.comb += [
            consume.eq(source.valid & source.ready),
            source.valid.eq(hold_valid),
            source.payload.data.eq(hold),
            source.last.eq(hold_last),
            new_byte.eq(Cat(shift[2:8], txd)),
            completing.eq(tx_en & (bits == 3)),
            flushing.eq(~tx_en & running),
        ]
        self.sync += [
            If(tx_en,
               If(~running,
                  running.eq(1),
                  bits.eq(0),
               ),
               shift.eq(new_byte),
               If(completing,
                  bits.eq(0),
               ).Else(
                  bits.eq(bits + 1),
               ),
            ).Elif(running,
               # frame end on the pads: flush a partial byte; when the
               # frame was a multiple of 4 dibits the last real byte is
               # already in hold — re-send it with last so the capture
               # sees the frame end (one duplicate byte, diagnostic only)
               running.eq(0),
               bits.eq(0),
            ),

            # hold register: a completion or the flush loads it; a plain
            # consume (no completion in the same cycle) empties it
            If(completing,
               hold.eq(new_byte),
               hold_valid.eq(1),
               hold_last.eq(0),
            ).Elif(flushing,
               hold.eq(shift),
               hold_valid.eq(1),
               hold_last.eq(1),
            ).Elif(consume,
               hold_valid.eq(0),
               hold_last.eq(0),
            ),
        ]


class UARTTX(LiteXModule):
    """115200 8N1 transmitter, sys domain. strobe (1 cycle) sends data."""

    def __init__(self, txd, divider=434):
        self.txd = txd
        self.data = Signal(8)
        self.strobe = Signal()
        self.busy = Signal()

        state = Signal(2)          # 0=idle 1=start 2=data 3=stop
        cnt = Signal(10)
        shreg = Signal(8)
        bitidx = Signal(3)
        pend_bit = Signal(reset=0)   # strobe arrived while busy
        pend_data = Signal(8)

        self.comb += [
            self.busy.eq(state != 0),
            txd.eq(Mux(state == 0, 1,
                   Mux(state == 1, 0, Mux(state == 2,
                       shreg[0], 1)))),
        ]
        self.sync += [
            If(state == 0,
               If(self.strobe,
                  state.eq(1),
                  cnt.eq(0),
                  shreg.eq(self.data),
                  bitidx.eq(0),
               ).Elif(pend_bit,
                  state.eq(1),
                  cnt.eq(0),
                  shreg.eq(pend_data),
                  pend_bit.eq(0),
                  bitidx.eq(0),
               ),
            ).Elif(state == 1,
               cnt.eq(cnt + 1),
               If(cnt == (divider - 1),
                  state.eq(2),
                  cnt.eq(0),
               ),
            ).Elif(state == 2,
               cnt.eq(cnt + 1),
               If(cnt == (divider - 1),
                  shreg.eq(Cat(shreg[1:8], Signal())),  # LSB first
                  If(bitidx == 7,
                     state.eq(3),
                  ).Else(
                     bitidx.eq(bitidx + 1),
                  ),
                  cnt.eq(0),
               ),
            ).Else(  # stop
               cnt.eq(cnt + 1),
               If(cnt == (divider - 1),
                  state.eq(0),
                  cnt.eq(0),
               ),
            ),
            # a strobe while busy queues the byte for the next slot
            If(self.strobe & (state != 0),
               pend_bit.eq(1),
               pend_data.eq(self.data),
            ),
        ]
        # sim-debug handles (plain references, no fragment impact)
        self._state = state
        self._cnt = cnt
        self._shreg = shreg
        self._pend_bit = pend_bit


class _FrameCapture(LiteXModule):
    """Writes a shim's byte stream into a 128-byte buffer (in the
    shim's own domain); exposes a synchronized frame_pending/len for
    the sys domain. Capture is suspended while a frame is pending so
    the buffer is stable while the UART dumps it."""

    def __init__(self, source, cd_name, mem_name):
        self.mem = mem = Memory(8, 128, name=mem_name)
        wr = mem.get_port(write_capable=True, mode=READ_FIRST,
                          clock_domain=cd_name)
        self.specials += mem, wr

        wr_ptr = Signal(7)
        pend = Signal(reset=0)             # in cd_name's domain
        clear_d = Signal(reset=0)
        clear = Signal(reset=0)            # sys -> cd_name, 2FF

        self.frame_pending = Signal()      # read in sys (synchronized)
        self.frame_len = Signal(7)         # read in sys (synchronized)
        self.clear_in = Signal()           # sys-domain clear pulse

        self.comb += [
            source.ready.eq(~pend),
            wr.adr.eq(wr_ptr),
            wr.dat_w.eq(source.payload.data),
            wr.we.eq(source.valid & ~pend),
        ]
        self.sync += [
            clear.eq(self.clear_in),
            clear_d.eq(clear),
            If(clear_d,
               pend.eq(0),
            ).Elif(source.valid & source.last,
               pend.eq(1),
            ),
            If(source.valid & source.last,
               self.frame_len.eq(wr_ptr + 1),
               wr_ptr.eq(0),                 # restart for the next frame
            ).Elif(source.valid & ~pend,
               If(wr_ptr != 127,
                  wr_ptr.eq(wr_ptr + 1),
               ),
            ),
        ]
        self.rd_port = mem.get_port(async_read=True, clock_domain="sys")
        self._pend = pend                  # for the sys-side synchronizer
        # sim-debug handles (plain references, no fragment impact)
        self._wr_ptr = wr_ptr
        self._we = wr.we


class SinkTap(LiteXModule):
    """Exposes the ACCEPTED byte stream of a sink endpoint (valid &
    ready) as a source, so a _FrameCapture can record what a consumer
    actually took — without touching the consumer's ready."""

    def __init__(self, sink):
        self.source = source = stream.Endpoint(eth_phy_description(8))
        self.comb += [
            source.valid.eq(sink.valid & sink.ready),
            source.payload.data.eq(sink.payload.data),
            source.last.eq(sink.last),
        ]


class UARTProbe(LiteXModule):
    def __init__(self):
        self.rmii_crs_dv = crs_dv_i = Signal()
        self.rmii_rxd = rxd_i = Signal(2)
        self.uart_txd = uart_txd_o = Signal()

        # clock domains (wired by the amaranth wrapper):
        #   sys    = plain REF_CLK  (UART, captures, pos shim)
        #   negrx  = inverted REF_CLK (negedge-sampling shim)
        self.clock_domains.cd_sys = ClockDomain()
        self.clock_domains.cd_eth_rx = ClockDomain()
        self.clock_domains.cd_negrx = ClockDomain()
        self.clock_domains.cd_eth_tx = ClockDomain()

        self.submodules.shim_pos = ClockDomainsRenamer("eth_rx")(
            RMIIRXShim(crs_dv_i, rxd_i))
        self.submodules.shim_neg = ClockDomainsRenamer("negrx")(
            RMIIRXShim(crs_dv_i, rxd_i))

        # TX diagnostics: a fixed frame every ~100 ms into the TX shim
        # (plain REF_CLK edge, RMII-spec convention), while a dibit
        # capture records exactly what the pads carry.
        self.rmii_tx_en = tx_en_o = Signal()
        self.rmii_txd = txd_o = Signal(2)
        self.submodules.txgen = ClockDomainsRenamer("eth_tx")(
            SlowTXGenerator())
        self.submodules.tx_shim = ClockDomainsRenamer("eth_tx")(
            RMIITXShim(tx_en_o, txd_o))
        self.comb += self.txgen.sink.connect(self.tx_shim.sink)

        self.submodules.cap_pos = _FrameCapture(
            self.shim_pos.source, "eth_rx", "cap_pos")
        self.submodules.cap_neg = _FrameCapture(
            self.shim_neg.source, "negrx", "cap_neg")

        # TX pad dibit capture: pack 4 pad dibits per byte, same
        # LSB-first convention as the shims, so the dumped bytes map
        # 1:1 onto what the TX shim should have been fed.
        self.submodules.txpack = ClockDomainsRenamer("eth_tx")(
            TXDibitPacker(tx_en_o, txd_o))
        self.submodules.cap_tx = _FrameCapture(
            self.txpack.source, "eth_tx", "cap_tx")

        # what the TX shim CONSUMED (accepted bytes), for comparison
        # against what the pads carried (marker 0x04)
        self.submodules.txtap = SinkTap(self.tx_shim.sink)
        self.submodules.cap_tap = _FrameCapture(
            self.txtap.source, "eth_tx", "cap_tap")

        uart_txd = Signal()
        self.submodules.uart = UARTTX(uart_txd)
        self.comb += uart_txd_o.eq(uart_txd)

        # frame flags synchronized into sys
        pend_pos_d = Signal()
        pend_neg_d = Signal()
        pend_pos = Signal()
        pend_neg = Signal()
        len_pos = Signal(7)
        len_neg = Signal(7)
        self.sync += [
            pend_pos_d.eq(self.cap_pos._pend),
            pend_pos.eq(pend_pos_d),
            pend_neg_d.eq(self.cap_neg._pend),
            pend_neg.eq(pend_neg_d),
            len_pos.eq(self.cap_pos.frame_len),
            len_neg.eq(self.cap_neg.frame_len),
        ]

        # heartbeat timer (declared before the dump FSM: its strobe
        # shares the uart.strobe mux below)
        hb = Signal(26)
        hb_fire = Signal()
        self.comb += hb_fire.eq(hb == 0)
        # one-shot: on fire, advance past 0 so the strobe lasts exactly
        # one cycle and re-arms only after a full wrap
        self.sync += [
            If(hb_fire, hb.eq(1)).Else(hb.eq(hb + 1)),
        ]

        # dump FSM
        state = Signal(3)      # 0=idle 1=marker 2=len 3=last 4..=data
        marker = Signal(8)
        length = Signal(7)
        idx = Signal(7)
        clear_pos = Signal()
        clear_neg = Signal()
        clear_tx = Signal()
        clear_tap = Signal()
        pend_tx_d = Signal()
        pend_tx = Signal()
        len_tx = Signal(7)
        pend_tap_d = Signal()
        pend_tap = Signal()
        len_tap = Signal(7)
        self.sync += [
            pend_tx_d.eq(self.cap_tx._pend),
            pend_tx.eq(pend_tx_d),
            len_tx.eq(self.cap_tx.frame_len),
            pend_tap_d.eq(self.cap_tap._pend),
            pend_tap.eq(pend_tap_d),
            len_tap.eq(self.cap_tap.frame_len),
        ]
        self.comb += [
            self.cap_pos.clear_in.eq(clear_pos),
            self.cap_neg.clear_in.eq(clear_neg),
            self.cap_tx.clear_in.eq(clear_tx),
            self.cap_tap.clear_in.eq(clear_tap),
            self.uart.strobe.eq((state != 0) |
                                ((state == 0) &
                                 (pend_pos | pend_neg | pend_tap |
                                  pend_tx | hb_fire))),
            self.uart.data.eq(Mux(state == 0, 0x7E,
                             Mux(state == 1, 0xAA,
                             Mux(state == 2, marker,
                             Mux(state == 3, length,
                             Mux(marker == 0x01,
                                 self.cap_pos.rd_port.dat_r,
                             Mux(marker == 0x02,
                                 self.cap_neg.rd_port.dat_r,
                             Mux(marker == 0x04,
                                 self.cap_tap.rd_port.dat_r,
                                 self.cap_tx.rd_port.dat_r)))))))),
            self.cap_pos.rd_port.adr.eq(idx),
            self.cap_neg.rd_port.adr.eq(idx),
            self.cap_tx.rd_port.adr.eq(idx),
            self.cap_tap.rd_port.adr.eq(idx),
        ]
        self.sync += [
            If(state == 0,
               If(pend_pos,
                  state.eq(1),
                  marker.eq(0x01),
                  length.eq(len_pos),
                  idx.eq(0),
                  clear_pos.eq(1),
               ).Elif(pend_neg,
                  state.eq(1),
                  marker.eq(0x02),
                  length.eq(len_neg),
                  idx.eq(0),
                  clear_neg.eq(1),
               ).Elif(pend_tap,
                  state.eq(1),
                  marker.eq(0x04),
                  length.eq(len_tap),
                  idx.eq(0),
                  clear_tap.eq(1),
               ).Elif(pend_tx,
                  state.eq(1),
                  marker.eq(0x03),
                  length.eq(len_tx),
                  idx.eq(0),
                  clear_tx.eq(1),
               ),
            ).Elif(state == 1,
               clear_pos.eq(0),
               clear_neg.eq(0),
               clear_tx.eq(0),
               clear_tap.eq(0),
               If(~self.uart.busy, state.eq(2)),
            ).Elif(state == 2,
               If(~self.uart.busy, state.eq(3)),
            ).Elif(state == 3,
               If(~self.uart.busy,
                  If(length == 0,
                     state.eq(0),
                  ).Else(
                     state.eq(4),
                  ),
               ),
            ).Else(  # data
               If(~self.uart.busy,
                  If(idx == (length - 1),
                     state.eq(0),
                     idx.eq(0),
                  ).Else(
                     idx.eq(idx + 1),
                  ),
               ),
            ),
        ]

        # heartbeat: the timer is declared above; nothing else here.


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    dut = UARTProbe()

    ios = {
        dut.rmii_crs_dv, dut.rmii_rxd, dut.uart_txd,
        dut.rmii_tx_en, dut.rmii_txd,
        dut.cd_sys.clk, dut.cd_sys.rst,
        dut.cd_eth_rx.clk, dut.cd_eth_rx.rst,
        dut.cd_negrx.clk, dut.cd_negrx.rst,
        dut.cd_eth_tx.clk, dut.cd_eth_tx.rst,
    }

    out = os.path.join(here, "uart_probe.v")
    result = convert(dut, ios=ios, name="uart_probe")
    result.write(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
