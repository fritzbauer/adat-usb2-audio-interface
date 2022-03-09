#!/usr/bin/env python3
#
# Copyright (c) 2021 Hans Baier <hansfbaier@gmail.com>
# SPDX-License-Identifier: CERN-OHL-W-2.0

from scipy import signal
from amaranth import *
from amaranth.hdl.mem import ReadPort, WritePort, Memory
from amaranth.sim import Tick
from pprint import pformat
import numpy as np
from amlib.stream import StreamInterface
from scipy.io import wavfile

from amlib.test       import GatewareTestCase, sync_test_case

class FIRConvolver(Elaboratable):
    def __init__(self,
                 samplerate:     int=48000,
                 bitwidth:       int=24,
                 fraction_width: int=24,
                 verbose:        bool=True) -> None:

        self.signal_in  = StreamInterface(name="signal_stream_in", payload_width=bitwidth)#Signal(signed(bitwidth))
        self.signal_out = StreamInterface(name="signal_stream_out", payload_width=bitwidth)

        # convert to fixed point representation
        self.tapcount = 2800
        self.slices = 4
        self.slice_size = self.tapcount // self.slices

        #taps2 = signal.firwin(self.tapcount-1, 3000, pass_zero='highpass', fs=48000)
        #taps = [0] * self.tapcount
        #taps = [
        #    0.5,0.4,0.3,0.2,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,
        #    0.5/2, 0.4/2, 0.3/2, 0.2/2, 0.1/2, 0.09/2, 0.08/2, 0.07/2, 0.06/2, 0.05/2, 0.04/2, 0.03/2, 0.02/2, 0.01/2
        #]
        #taps[0] = 1 >> 24
        self.bitwidth = bitwidth
        self.fraction_width = fraction_width
        assert bitwidth <= fraction_width, f"Bitwidth {bitwidth} must not exceed {fraction_width}"

        #For testing
        #taps = signal.firwin(self.tapcount,2000,pass_zero='lowpass',fs=48000)
        #taps2 = signal.firwin(self.tapcount, 500, pass_zero='lowpass', fs=48000)
        sample_rate, sig = wavfile.read("IR.wav")

        assert  sample_rate == samplerate, f"Unsupported samplerate {sample_rate} for IR file expected {samplerate}"
        taps_fp = sig[:self.tapcount,0]
        taps2_fp = sig[:self.tapcount,1]

        #self.taps_memories, self.taps2_memories, self.samples_memories, self.samples2_memories = [],[],[],[]
#
        #for i in range(self.slices):
        #    self.taps_memories.append(Memory(width=self.bitwidth, depth=self.slice_size, name=f"taps_memory{i}"))
        #    self.taps_memories[i].init = taps_fp[i*self.slice_size:(i+1)*self.slice_size]
        #    self.taps2_memories.append(Memory(width=self.bitwidth, depth=self.slice_size, name=f"taps2_memory{i}"))
        #    self.taps2_memories[i].init = taps2_fp[i * self.slice_size:(i + 1) * self.slice_size]
#
        #    self.samples_memories.append(Memory(width=self.bitwidth, depth=self.slice_size, name=f"samples_memory{i}"))
        #    self.samples2_memories.append(Memory(width=self.bitwidth, depth=self.slice_size, name=f"samples2_memory{i}"))

        self.taps_memories = Array(Memory(width=self.bitwidth, depth=self.slice_size+1, name=f"taps_memory{i}") for i in range(self.slices))
        self.taps2_memories = Array(Memory(width=self.bitwidth, depth=self.slice_size+1, name=f"taps2_memory{i}") for i in range(self.slices))
        self.samples_memories = Array(Memory(width=self.bitwidth, depth=self.slice_size+1, name=f"samples_memory{i}") for i in range(self.slices))
        self.samples2_memories = Array(Memory(width=self.bitwidth, depth=self.slice_size+1, name=f"samples2_memory{i}") for i in range(self.slices))


        for i in range(self.slices):
            self.taps_memories[i].init = taps_fp[i * self.slice_size:(i + 1) * self.slice_size]
            self.taps2_memories[i].init = taps2_fp[i * self.slice_size:(i + 1) * self.slice_size]

        self.fsm_state_out = Signal(5)


    def insert(self, m: Module, value: int, offset_var, memories: [], slice: int, sample_index: int):
        m.d.sync += offset_var.eq(offset_var+1)
        with m.If(offset_var == self.tapcount):
            m.d.sync += offset_var.eq(1)
        #memory_port_number = Signal(self.slices)
        #m.d.comb += memory_port_number.eq(offset // self.slice_size)
        m.d.comb += [
            slice.eq(offset_var // self.slice_size),
            sample_index.eq(offset_var % self.slice_size),
        ]
        m.d.comb += [
            memories[slice].data.eq(self.signal_in.payload),
            memories[slice].addr.eq(sample_index),
            memories[slice].en.eq(1)
        ]

    # we use the array indices flipped, ascending from zero
    # so x[0] is x_n, x[1] is x_n-
    # 1, x[2] is x_n-2 ...
    # in other words: higher indices are past values, 0 is most recent
    def get_at(self, m: Module, index: int, offset_var: int, memories: [], slice: int, sample_index: int):
        #get_at_index = (offset_var - 1) - index
        with m.If(((offset_var - 1) - index) >=  0):

            m.d.comb += [
                slice.eq(((offset_var - 1) - index) // self.slice_size),
                sample_index.eq(((offset_var - 1) - index) % self.slice_size),
                memories[slice].addr.eq(sample_index),
            ]
        with m.Else():
            m.d.comb += [
                slice.eq(((offset_var - 1) - index + self.tapcount) // self.slice_size),
                sample_index.eq(((offset_var - 1) - index + self.tapcount) % self.slice_size),
                memories[slice].addr.eq(sample_index),
            ]


    def elaborate(self, platform) -> Module:
        m = Module()

        #taps_read_ports, taps2_read_ports, samples_write_ports, samples2_write_ports, samples_read_ports, samples2_read_ports = [], [], [], [], [], []

        taps_read_ports = Array(self.taps_memories[i].read_port() for i in range(self.slices))
        taps2_read_ports = Array(self.taps2_memories[i].read_port() for i in range(self.slices))
        samples_write_ports = Array(self.samples_memories[i].write_port() for i in range(self.slices))
        samples2_write_ports = Array(self.samples2_memories[i].write_port() for i in range(self.slices))
        samples_read_ports = Array(self.samples_memories[i].read_port() for i in range(self.slices))
        samples2_read_ports = Array(self.samples2_memories[i].read_port() for i in range(self.slices))

        m.submodules += [taps_read_ports, taps2_read_ports, samples_write_ports, samples2_write_ports, samples_read_ports, samples2_read_ports]

        slices = Array(Signal(range(self.slices), name=f"slices1_{i}") for i in range(self.slices))
        slices2 = Array(Signal(range(self.slices), name=f"slices2_{i}") for i in range(self.slices))
        sample_indexes = Array(Signal(range(self.slice_size), name=f"sample1_index{i}") for i in range(self.slices))
        sample2_indexes = Array(Signal(range(self.slice_size), name=f"sample2_index{i}") for i in range(self.slices))

        a_values = Array(Signal(signed(self.bitwidth),name=f"a{i}") for i in range(self.slices*2))
        b_values = Array(Signal(signed(self.bitwidth),name=f"b{i}") for i in range(self.slices*2))
        madd_values = Array(Signal(signed(self.bitwidth + self.fraction_width),name=f"madd{i}") for i in range(self.slices*2))

        sumSignalL = Signal(self.bitwidth + self.fraction_width)
        sumSignalR = Signal.like(sumSignalL)

        ix = Signal(range(self.tapcount + 1))
        offset = Signal(Shape.cast(range(self.tapcount)).width)
        offset2 = Signal.like(offset)

        m.d.sync += [
            #samples_write_port.en.eq(0),
            #samples2_write_port.en.eq(0),
            #self.signal_out.valid.eq(0),
        ]
        for i in range(self.slices):
            m.d.comb += [
                samples_write_ports[i].en.eq(0),
                samples2_write_ports[i].en.eq(0),
            ]

        m.d.comb += self.signal_out.valid.eq(0)
        m.d.comb += self.signal_in.ready.eq(0)


        set1 = Signal()
        set2 = Signal()
        with m.FSM(reset="IDLE"):
            with m.State("IDLE"):
                m.d.comb += self.fsm_state_out.eq(1)
                with m.If(self.signal_in.valid & self.signal_in.last & ~set2): # convolve right channel
                    self.insert(m, self.signal_in, offset_var=offset2, memories=samples2_write_ports, slice=slices2[0], sample_index=sample2_indexes[0])
                    m.d.sync += set2.eq(1)

                with m.Elif(self.signal_in.valid & self.signal_in.first & ~set1): #left channel
                    self.insert(m, self.signal_in, offset_var=offset, memories=samples_write_ports, slice=slices[0], sample_index=sample_indexes[0])
                    m.d.sync += set1.eq(1)

                with m.If(set1 & set2):
                    m.d.sync += ix.eq(0)
                    for i in range(self.slices * 2):
                        m.d.sync += madd_values[i].eq(0)
                    m.next = "STORE"
                with m.Else():
                    m.d.comb += self.signal_in.ready.eq(1)


            with m.State("STORE"):
                m.d.comb += self.fsm_state_out.eq(2)


                m.next = "MAC"
            with m.State("MAC"):
                m.d.comb += self.fsm_state_out.eq(3)
                #for i in range(self.slices):
                #    m.d.comb += samples_read_ports[i].addr.eq(self.slice_size)
                #    m.d.comb += samples2_read_ports[i].addr.eq(self.slice_size)

                for i in range(self.slices):
                    self.get_at(m, ix + self.slice_size * i, offset_var=offset, memories=samples_read_ports,
                                slice=slices[i], sample_index=sample_indexes[i])
                    self.get_at(m, ix + self.slice_size * i, offset_var=offset2, memories=samples2_read_ports, slice=slices2[i], sample_index=sample2_indexes[i])


                    m.d.comb += [
                        taps_read_ports[i].addr.eq(ix),
                        taps2_read_ports[i].addr.eq(ix),
                        a_values[i].eq(samples_read_ports[slices2[i]].data),
                        a_values[i + self.slices].eq(samples2_read_ports[slices2[i]].data),
                        b_values[i].eq(taps_read_ports[i].data),
                        b_values[i + self.slices].eq(taps2_read_ports[i].data),
                    ]

                #for i in range(self.slices*2):
                #    m.d.sync += madd_values[i].eq(madd_values[i] + (a_values[i] * b_values[i]))
                # this is probably it - if it fits in the available LABS:
                for i in range(self.slices):
                    m.d.sync += madd_values[i].eq(madd_values[i] + (a_values[i] * b_values[i]) + (a_values[i+self.slices] * b_values[i+self.slices]))
                    m.d.sync += madd_values[i+self.slices].eq(madd_values[i] + (a_values[i+self.slices] * b_values[i]) + (a_values[i] * b_values[i+self.slices]))

                with m.If(ix == self.slice_size - 1):
                    sumL = 0
                    sumR = 0
                    for i in range(self.slices):
                        sumL += madd_values[i]
                        sumR += madd_values[i + self.slices]

                    m.d.sync += [
                        sumSignalL.eq(sumL),
                        sumSignalR.eq(sumR),
                    ]
                    m.next = "OUTPUT"
                with m.Else():
                    m.d.sync += ix.eq(ix + 1)

            with m.State("OUTPUT"):
                m.d.comb += self.fsm_state_out.eq(4)

                m.d.comb += [
                    self.signal_out.payload.eq(sumSignalL >> self.fraction_width),
                    self.signal_out.valid.eq(1),
                    self.signal_out.first.eq(1),
                    self.signal_out.last.eq(0),
                ]
                with m.If(self.signal_out.ready):
                    m.next = "OUT_RIGHT"

            with m.State("OUT_RIGHT"):
                m.d.comb += self.fsm_state_out.eq(5)
                m.d.comb += [
                    self.signal_out.payload.eq(sumSignalR >> self.fraction_width),
                    # self.signal_out.payload.eq(sum),
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
    FRAGMENT_ARGUMENTS = dict(samplerate=48000)

    def wait(self, n_cycles: int):
        for _ in range(n_cycles):
            yield

    def wait_ready(self, dut):
        waitcount = 0
        #ready = yield dut.ready_out
        #while (ready == 0):
        #yield dut.signal_in.valid.eq(0)
        #yield print(dut.signal_in.ready)
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
        yield dut.signal_in.payload.eq(max-100)
        yield Tick()
        #yield from self.wait_ready(dut)
        #yield dut.signal_in.payload.eq(2222)
        #for _ in range(5): yield
        #yield dut.signal_in.valid.eq(0)
        #for _ in range(60): yield
        #yield dut.signal_in.valid.eq(1)
        #for _ in range(60): yield
        #yield dut.signal_in.payload.eq(0)
        #return
        for i in range(240):
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
            yield dut.signal_in.payload.eq(i+200)
            yield Tick()
            print("end of loop")
