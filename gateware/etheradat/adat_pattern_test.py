#!/usr/bin/env python3
"""Hardcoded ADAT output test (bring-up step: ADAT TX + loopback RX).

Two ADAT transmitters send a 1 kHz / 2 kHz sine on all 8 channels:
- toslink 1 TX -> Audient iD14 ADAT input (channels 1-8),
- toslink 2 TX -> loopback cable -> toslink 2 RX.

An ADATReceiver on toslink 2 RX lets us verify the receive path on the
same board. The UART (clocked from the 50 MHz board oscillator, the
proven probe implementation) dumps every 100 ms:

    AA <synced2> <addr2> <sample2 lo mid hi> <underflow> 55

This design has no MAC/bridge: it validates the board, the audio PLLs,
the ADAT transmitters and the receiver independently of Ethernet.
Verified on hardware 23.11.2026: synced2 = 1 (loopback works); the
Audient iD14 reported "ADAT Clock Validity = off" even while the sine
was transmitting (its ADAT input is the known-flaky part, see the
status document).

Build: python etheradat/scripts/generate_quartus_files_adatpat.py <dir>
"""

import os

from amaranth            import *
from amaranth.build      import Attrs, Pins, Resource, Subsignal
from amaranth.lib.cdc    import FFSynchronizer

from luna import top_level_cli

from adat import ADATTransmitter, ADATReceiver

_uart_dbg = Resource("uart_dbg", 0,
    Subsignal("tx", Pins("J_2:34", dir="o"),
              Attrs(io_standard="3.3-V LVCMOS")))


def _sine_lut(n=48, amp=(1 << 23) - 1):
    import math
    return [int(round(amp * math.sin(2 * math.pi * i / n))) & 0xFFFFFF
            for i in range(n)]


class PatternFeeder(Elaboratable):
    """Commits a sine on 8 channels to one ADATTransmitter."""

    def __init__(self, tx, phase_step, lut):
        self.tx = tx
        self.phase_step = phase_step
        self.lut = lut

    def elaborate(self, platform):
        m = Module()
        sync = m.d.sync
        tx = self.tx
        phase = Signal(6)
        chn = Signal(3)

        sample = Signal(24)
        m.d.comb += sample.eq(Array(self.lut)[(phase + chn * 6) % 48])
        m.d.comb += [
            tx.addr_in.eq(chn),
            tx.sample_in.eq(sample),
            tx.valid_in.eq(tx.ready_out),
            tx.last_in.eq(chn == 7),
        ]
        with m.If(tx.ready_out):
            with m.If(chn == 7):
                sync += [chn.eq(0),
                         phase.eq((phase + self.phase_step) % 48)]
            with m.Else():
                sync += chn.eq(chn + 1)
        return m


class AdatPatternTest(Elaboratable):
    def elaborate(self, platform):
        m = Module()
        clk = platform.request(platform.default_clk)
        m.domains.sys = ClockDomain()
        m.d.comb += ClockSignal("sys").eq(clk.i)

        m.submodules.car = platform.clock_domain_generator(clk_input=clk.i)

        leds = platform.request("leds")
        platform.add_resources([_uart_dbg])
        uart = platform.request("uart_dbg", 0)
        tos = [platform.request("toslink", i + 1) for i in range(3)]

        lut = _sine_lut()

        # output 1: 1 kHz, output 2: 2 kHz
        tx = [ADATTransmitter(fifo_depth=9 * 4) for _ in range(2)]
        for o in range(2):
            m.submodules[f"tx{o}"] = tx[o]
            m.submodules[f"feed{o}"] = PatternFeeder(tx[o], 1 + o, lut)
        m.d.comb += [
            tos[0].tx.o.eq(tx[0].adat_out),
            tos[1].tx.o.eq(tx[1].adat_out),
        ]

        # receiver on input 2 (loopback from output 2)
        rx = DomainRenamer("fast")(
            ADATReceiver(platform.fast_domain_clock_freq))
        m.submodules.rx = rx
        m.d.comb += rx.adat_in.eq(tos[1].rx.i)

        synced = Signal()
        addr = Signal(3)
        sample = Signal(24)
        underflow = Signal()
        m.submodules += [
            FFSynchronizer(rx.synced_out, synced, o_domain="sys"),
            FFSynchronizer(rx.addr_out, addr, o_domain="sys"),
            FFSynchronizer(rx.sample_out, sample, o_domain="sys"),
            FFSynchronizer(tx[0].underflow_out | tx[1].underflow_out,
                           underflow, o_domain="sys"),
        ]

        # proven probe UART (115200 8N1 from the 50 MHz local clock)
        DIV = 434
        data = Signal(8)
        start = Signal()
        busy = Signal()
        phase = Signal(range(DIV + 1))
        bit_i = Signal(4)
        shift = Signal(8)
        txd = Signal(reset=1)
        m.d.comb += uart.tx.eq(txd)
        with m.If(busy):
            with m.If(phase == DIV - 1):
                m.d.sys += phase.eq(0)
                with m.If(bit_i == 9):
                    m.d.sys += [busy.eq(0), txd.eq(1)]
                with m.Elif(bit_i == 8):
                    m.d.sys += [bit_i.eq(9), txd.eq(1)]
                with m.Else():
                    m.d.sys += [bit_i.eq(bit_i + 1), txd.eq(shift[0]),
                                shift.eq(shift >> 1)]
            with m.Else():
                m.d.sys += phase.eq(phase + 1)
        with m.Else():
            with m.If(start):
                m.d.sys += [busy.eq(1), shift.eq(data), bit_i.eq(0),
                            phase.eq(0), txd.eq(0)]

        # dump every 100 ms: AA synced addr s0 s1 s2 uf 55
        period = 5_000_000
        timer = Signal(range(period))
        fire = Signal()
        idx = Signal(4)
        dumping = Signal()
        m.d.sys += fire.eq(0)
        with m.If(timer == period - 1):
            m.d.sys += [timer.eq(0), fire.eq(1)]
        with m.Else():
            m.d.sys += timer.eq(timer + 1)
        with m.If(fire):
            m.d.sys += [dumping.eq(1), idx.eq(0)]
        with m.Elif(dumping & ~busy):
            with m.If(idx == 7):
                m.d.sys += dumping.eq(0)
            with m.Else():
                m.d.sys += idx.eq(idx + 1)
        m.d.comb += [
            start.eq(fire | (dumping & ~busy)),
            data.eq(Array([0xAA, synced,
                           Cat(addr, 0, 0, 0, 0, 0)[0:8],
                           sample[0:8], sample[8:16], sample[16:24],
                           underflow, 0x55])[idx]),
        ]

        m.d.comb += [
            leds.host.eq(synced),
            leds.usb1.eq(tx[0].ready_out),
            leds.usb2.eq(underflow),
        ]
        return m


if __name__ == "__main__":
    os.environ.setdefault("LUNA_PLATFORM",
                          "platforms:ADATFaceCycloneIV_EP4CGX150")
    top_level_cli(AdatPatternTest)
