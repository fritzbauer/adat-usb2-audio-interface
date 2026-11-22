"""Vendored, hardware-correct Depacketizer for the EtherADAT baseline.

LiteX's stock `Depacketizer` (litex/soc/interconnect/packet.py @ 6438a25)
drives `sr_shift` from FSM actions, i.e. from a register, but uses it in
the SAME clock edge to gate the header shift register. On the Cyclone IV
build this loses the first header byte: the MAC depacketizer saw
`ff ff ff ff ff f8 e4 ...` instead of `ff x 6 f8 e4 ...`, and the ARP
depacketizer decoded opcode 0x01f8 instead of 0x0001 (confirmed with
UART byte taps 22.11.2026).

Here `sr_shift` is combinational from the FSM state, so byte 0 of a
frame is captured immediately. Only the aligned (dw=8, whole-byte
header) case needed by LiteEth's MAC/ARP depacketizers is implemented.
"""

from migen import *
from litex.gen import *
from litex.soc.interconnect import stream


class FixedDepacketizer(LiteXModule):
    def __init__(self, sink_description, source_description, header,
                 dummy_prefix=False, skip_bytes=0):
        self.sink = sink = stream.Endpoint(sink_description)
        self.source = source = stream.Endpoint(source_description)
        self.header = Signal(header.length * 8)

        data_width = len(sink.data)
        assert header.length * 8 % data_width == 0
        header_words = header.length * 8 // data_width
        total_words = header_words + skip_bytes

        sr = Signal(header.length * 8, reset_less=True)
        count = Signal(max=max(total_words, 2))
        skip_cnt = Signal(max=max(skip_bytes + 1, 2), reset=0)
        data_copy_first = Signal(reset=1)

        self.comb += [
            self.header.eq(sr),
            header.decode(self.header, source),
            source.data.eq(sink.data),
        ]
        if hasattr(sink, "error") and hasattr(source, "error"):
            self.comb += source.error.eq(sink.error)
        sink_be_name = stream.byte_enable_name(sink)
        source_be_name = stream.byte_enable_name(source)
        if sink_be_name is not None and source_be_name is not None:
            self.comb += [
                getattr(source, source_be_name).eq(
                    getattr(sink, sink_be_name)),
            ]

        self.fsm = fsm = FSM(reset_state="IDLE")
        # Combinational shift enable: capture byte 0 in the same cycle in
        # which the FSM first sees it (the stock version is one late).
        sr_shift = Signal()
        self.comb += sr_shift.eq(
            (fsm.ongoing("IDLE") | fsm.ongoing("HEADER-RECEIVE")) &
            sink.valid & sink.ready & (skip_cnt >= skip_bytes))
        self.sync += [
            If(sink.valid & sink.ready & (skip_cnt < skip_bytes),
               skip_cnt.eq(skip_cnt + 1)),
            If(sr_shift, sr.eq(Cat(sr[data_width:], sink.data))),
        ]

        fsm.act("IDLE",
                sink.ready.eq(1),
                NextValue(count, 1),
                If(sink.valid & sink.ready, NextState("HEADER-RECEIVE")))

        fsm.act("HEADER-RECEIVE",
                sink.ready.eq(1),
                If(sink.valid & sink.ready,
                   NextValue(count, count + 1),
                   If(count == total_words - 1,
                      NextValue(data_copy_first, 1),
                      NextState("ALIGNED-DATA-COPY"),
                      NextValue(count, count + 1)),
                   If(sink.last & sink.ready, NextState("IDLE"))))

        send_dummy = Signal(reset=int(dummy_prefix))
        data_copy = [
            source.valid.eq(sink.valid),
            source.last.eq(sink.last),
            source.first.eq(data_copy_first & sink.valid),
            sink.ready.eq(source.ready),
            If(source.valid & source.ready,
               NextValue(data_copy_first, 0),
               If(source.last,
                  NextValue(skip_cnt, 0),
                  If(dummy_prefix, NextValue(send_dummy, 1)),
                  NextState("IDLE"))),
        ]
        fsm.act("ALIGNED-DATA-COPY",
                If(dummy_prefix & send_dummy,
                   # One leading dummy byte compensates the beat shift of
                   # the crossbar/ARP path on this Cyclone IV build.
                   source.valid.eq(1),
                   source.data.eq(0),
                   source.first.eq(1),
                   source.last.eq(0),
                   source.be.eq(1),
                   sink.ready.eq(0),
                   If(source.ready, NextValue(send_dummy, 0)),
                ).Else(*data_copy))
