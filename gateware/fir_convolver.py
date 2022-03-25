#!/usr/bin/env python3
#
# Copyright (c) 2021 Hans Baier <hansfbaier@gmail.com>
# SPDX-License-Identifier: CERN-OHL-W-2.0

from amaranth import Elaboratable, Module, Signal, Array, Memory, signed, Shape, Cat
from amaranth.sim import Tick
from amlib.stream import StreamInterface

import numpy as np
import math

from amlib.test       import GatewareTestCase, sync_test_case

class FIRConvolver(Elaboratable):
    def __init__(self,
                 taps: [],
                 samplerate:     int=48000,
                 clockfrequency:  int=60e6,
                 bitwidth:       int=24) -> None:

        self.signal_in  = StreamInterface(name="signal_stream_in", payload_width=bitwidth)
        self.signal_out = StreamInterface(name="signal_stream_out", payload_width=bitwidth)

        self.USE_CROSSFEED = True
        self.tapcount = len(taps) #4096 synthesizes
        self.slices = math.ceil(self.tapcount/(clockfrequency/samplerate)) #4

        self.bitwidth = bitwidth
        self.size_of_slizes = self.bitwidth * self.slices
        self.number_of_slices = self.tapcount // self.slices

        print(f"Creating {self.slices} slices for {self.tapcount} taps.")

        assert self.tapcount % self.slices == 0, f"Tapcount {self.tapcount} cannot be evenly distributed on {self.slices} slizes."

        taps_fp = taps[:, 0]
        taps2_fp = taps[:, 1]

        self.taps1_memory = Memory(width=self.size_of_slizes, depth=self.number_of_slices, name=f"taps1_memory")
        self.taps2_memory = Memory(width=self.size_of_slizes, depth=self.number_of_slices, name=f"taps2_memory")
        self.samples1_memory = Memory(width=self.size_of_slizes, depth=self.number_of_slices, name=f"samples1_memory")
        self.samples2_memory = Memory(width=self.size_of_slizes, depth=self.number_of_slices, name=f"samples2_memory")

        taps_fp_mod = []
        taps2_fp_mod = []
        for i in range(0,self.tapcount, self.slices):
            val1 = 0
            val2 = 0
            for j in range(self.slices):
                val1 += taps_fp[i+j] << j * self.bitwidth
                val1 += taps2_fp[i+j] << j * self.bitwidth
            taps_fp_mod.append(val1)
            taps2_fp_mod.append(val2)

        self.taps1_memory.init = taps_fp_mod
        self.taps2_memory.init = taps2_fp_mod

        #for ILA
        self.fsm_state_out = Signal(5)

    def elaborate(self, platform) -> Module:
        m = Module()

        taps1_read_port = self.taps1_memory.read_port()
        taps2_read_port = self.taps2_memory.read_port()
        samples1_write_port = self.samples1_memory.write_port()
        samples2_write_port = self.samples2_memory.write_port()
        samples1_read_port = self.samples1_memory.read_port()
        samples2_read_port = self.samples2_memory.read_port()

        m.submodules += [taps1_read_port, taps2_read_port, samples1_write_port, samples2_write_port, samples1_read_port, samples2_read_port]

        set1 = Signal()
        set2 = Signal()

        previous_sample1 = Signal(self.size_of_slizes)
        previous_sample2 = Signal.like(previous_sample1)
        current_sample1 = Signal.like(previous_sample1)
        current_sample2 = Signal.like(previous_sample1)
        carryover1 = Signal(signed(self.bitwidth))
        carryover1_2 = Signal.like(carryover1)
        carryover2 = Signal.like(carryover1)
        carryover2_2 = Signal.like(carryover1)

        madd_values = Array(Signal(signed(self.bitwidth * 2), name=f"madd_values_{i}") for i in range(self.slices * 2))
        sumSignalL = Signal(signed(self.bitwidth*2))
        sumSignalR = Signal.like(sumSignalL)

        ix = Signal(range(self.number_of_slices+1))

        m.d.comb += [
            self.signal_in.ready.eq(0),
            taps1_read_port.addr.eq(ix),
            taps2_read_port.addr.eq(ix),
            samples1_read_port.addr.eq(ix),
            samples2_read_port.addr.eq(ix),
        ]

        m.d.sync += [
            self.signal_out.valid.eq(0),
            samples1_write_port.en.eq(0),
            samples2_write_port.en.eq(0),

            previous_sample1.eq(samples1_read_port.data),
            previous_sample2.eq(samples2_read_port.data),

            carryover1.eq(samples1_read_port.data[:self.bitwidth]),
            carryover1_2.eq(carryover1),
            carryover2.eq(samples2_read_port.data[:self.bitwidth]),
            carryover2_2.eq(carryover2),
        ]

        with m.FSM(reset="IDLE"):
            with m.State("IDLE"):
                m.d.comb += self.fsm_state_out.eq(1)
                with m.If(self.signal_in.valid & self.signal_in.first & ~set1):  # left channel
                    sample1_value = Cat(
                        samples1_read_port.data[:-self.bitwidth],
                        self.signal_in.payload.as_signed()
                    )

                    m.d.sync += [
                        samples1_write_port.data.eq(sample1_value),
                        samples1_write_port.addr.eq(0),
                        samples1_write_port.en.eq(1)
                    ]

                    m.d.sync += set1.eq(1)
                with m.Elif(self.signal_in.valid & self.signal_in.last & ~set2): # right channel
                    sample2_value = Cat(
                        samples2_read_port.data[:-self.bitwidth],
                        self.signal_in.payload.as_signed()
                    )
                    m.d.sync += [
                        samples2_write_port.data.eq(sample2_value),
                        samples2_write_port.addr.eq(0),
                        samples2_write_port.en.eq(1)
                    ]
                    m.d.sync += set2.eq(1)

                with m.If(set1 & set2):
                    for i in range(self.slices * 2):
                        m.d.sync += [
                            ix.eq(1),
                            madd_values[i].eq(0),
                            previous_sample1.eq(0),
                            previous_sample2.eq(0),
                            current_sample1.eq(0),
                            current_sample2.eq(0),
                            carryover1.eq(0),
                            carryover1_2.eq(0),
                            carryover2.eq(0),
                            carryover2_2.eq(0),
                        ]

                    m.next = "MAC"
                with m.Else():
                    m.d.comb += self.signal_in.ready.eq(1)
            with m.State("MAC"):
                m.d.comb += self.fsm_state_out.eq(2)

                with m.If(ix <= self.number_of_slices - 1):
                    for i in range(self.slices):
                        left_sample = samples1_read_port.data[i*self.bitwidth:(i+1)*self.bitwidth].as_signed()
                        right_sample = samples2_read_port.data[i*self.bitwidth:(i+1)*self.bitwidth].as_signed()
                        main_tap = taps1_read_port.data[i*self.bitwidth:(i+1)*self.bitwidth].as_signed()
                        bleed_tap = taps2_read_port.data[i*self.bitwidth:(i+1)*self.bitwidth].as_signed()

                        if self.USE_CROSSFEED:
                            m.d.sync += [
                                madd_values[i].eq(madd_values[i] + (left_sample * main_tap)  # left
                                                + (right_sample * bleed_tap)),  # right_bleed
                                madd_values[i + self.slices].eq(madd_values[i + self.slices] + (right_sample * main_tap) # right
                                                + (left_sample * bleed_tap))  # left_bleed
                            ]
                        else:
                            m.d.sync += [
                                madd_values[i].eq(madd_values[i] + (left_sample * main_tap)),
                                madd_values[i + self.slices].eq(madd_values[i + self.slices] + (right_sample * main_tap)),
                            ]

                with m.If(ix > 1):
                    m.d.sync += [
                        samples1_write_port.data.eq(Cat((previous_sample1 >> self.bitwidth)[:((self.slices-1)*self.bitwidth)], carryover1_2)),
                        samples1_write_port.addr.eq(ix - 2),
                        samples1_write_port.en.eq(1),

                        samples2_write_port.data.eq(Cat((previous_sample2 >> self.bitwidth)[:((self.slices-1)*self.bitwidth)], carryover2_2)),
                        samples2_write_port.addr.eq(ix - 2),
                        samples2_write_port.en.eq(1),
                    ]

                with m.If(ix == self.number_of_slices + 1):
                    m.next = "SUM"
                with m.Else():
                    m.d.sync += ix.eq(ix+1)

            with m.State("SUM"):
                m.d.comb += self.fsm_state_out.eq(3)

                sumL = 0
                sumR = 0
                for i in range(self.slices):
                    sumL += madd_values[i]
                    sumR += madd_values[i + self.slices]

                m.d.sync += [
                    sumSignalL.eq(sumL),
                    sumSignalR.eq(sumR),
                ]
                m.next = "OUTPUT LEFT"
            with m.State("OUTPUT LEFT"):
                m.d.comb += self.fsm_state_out.eq(4)
                m.d.sync += ix.eq(0)
                m.d.sync += [
                    self.signal_out.payload.eq(sumSignalL >> (self.bitwidth-1)),
                    self.signal_out.valid.eq(1),
                    self.signal_out.first.eq(1),
                    self.signal_out.last.eq(0),
                ]
                with m.If(self.signal_out.ready):
                    m.next = "OUTPUT RIGHT"

            with m.State("OUTPUT RIGHT"):
                m.d.comb += self.fsm_state_out.eq(5)
                m.d.sync += [
                    self.signal_out.payload.eq(sumSignalR >> (self.bitwidth-1)),
                    self.signal_out.valid.eq(1),
                    self.signal_out.first.eq(0),
                    self.signal_out.last.eq(1),
                ]
                m.d.sync += [
                    set1.eq(0),
                    set2.eq(0),
                ]

                with m.If(self.signal_out.ready):
                    m.next = "IDLE"

        return m


class FixedPointFIRFilterTest(GatewareTestCase):
    FRAGMENT_UNDER_TEST = FIRConvolver

    tapcount = 32
    bitwidth = 24
    tapdata = [int(1 * 2 ** (bitwidth-1)-1), int(0.5 * 2 ** (bitwidth-1)-1), int(0.25 * 2 ** (bitwidth-1)-1), int(0.1 * 2 ** (bitwidth-1)-1)]

    taps = np.zeros((tapcount, 2))
    for i in range(len(tapdata)):
        taps[i, 0] = int(tapdata[i])
        taps[i, 1] = int(tapdata[i])
    taps = taps.astype(np.int32)
    print(taps)
    FRAGMENT_ARGUMENTS = dict(taps=taps, samplerate=48000)

    def wait(self, n_cycles: int):
        for _ in range(n_cycles):
            yield

    def wait_ready(self, dut):
        waitcount = 0
        while (yield dut.signal_in.ready == 0):
            waitcount += 1
            yield from self.wait(1)
        print("Waitcount {}".format(waitcount))

    @sync_test_case
    def test_fir(self):
        dut = self.dut
        max = int(2**15 - 1)
        min = -max
        yield dut.signal_out.ready.eq(1)
        yield from self.wait_ready(dut)
        yield Tick()
        yield dut.signal_in.first.eq(1)
        yield dut.signal_in.last.eq(0)
        yield dut.signal_in.payload.eq(max)
        yield dut.signal_in.valid.eq(1)
        yield from self.wait_ready(dut)
        yield dut.signal_in.valid.eq(1)
        yield dut.signal_in.first.eq(0)
        yield dut.signal_in.last.eq(1)
        yield dut.signal_in.payload.eq(min)
        yield Tick()
        for i in range(60):
            yield from self.wait_ready(dut)
            yield dut.signal_in.valid.eq(1)
            yield dut.signal_in.first.eq(1)
            yield dut.signal_in.last.eq(0)
            yield dut.signal_in.payload.eq(i+100)
            yield Tick()
            yield from self.wait_ready(dut)
            yield dut.signal_in.valid.eq(1)
            yield dut.signal_in.first.eq(0)
            yield dut.signal_in.last.eq(1)
            yield dut.signal_in.payload.eq(-i-200)
            yield Tick()
            print("end of loop")
