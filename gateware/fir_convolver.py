#!/usr/bin/env python3
#
# Copyright (c) 2021 Hans Baier <hansfbaier@gmail.com>
# SPDX-License-Identifier: CERN-OHL-W-2.0

from amaranth import Elaboratable, Module, Signal, Array, Memory, signed, Shape
from amaranth.hdl.mem import Memory
from amaranth.sim import Tick
from amlib.stream import StreamInterface
import soundfile as sf
import math

from amlib.test       import GatewareTestCase, sync_test_case

class FIRConvolver(Elaboratable):
    def __init__(self,
                 samplerate:     int=48000,
                 clockfrequency:  int=60e6,
                 bitwidth:       int=24) -> None:

        self.signal_in  = StreamInterface(name="signal_stream_in", payload_width=bitwidth)#Signal(signed(bitwidth))
        self.signal_out = StreamInterface(name="signal_stream_out", payload_width=bitwidth)

        # convert to fixed point representation
        self.tapcount = 4096 #3072 synthesizes
        self.slices = 4 # math.ceil(self.tapcount/(clockfrequency/samplerate)) #4
        self.slice_size = self.tapcount // self.slices

        print(f"Creating {self.slices} slices for {self.tapcount} taps.")

        assert  self.tapcount % self.slices == 0, f"Tapcount {self.tapcount} cannot be evenly distributed on {self.slices} slizes."

        self.bitwidth = bitwidth

        # For testing
        # taps = signal.firwin(self.tapcount,2000,pass_zero='lowpass',fs=48000)
        # taps2 = signal.firwin(self.tapcount, 500, pass_zero='lowpass', fs=48000)
        # taps_fp = [int(x * 2 ** (bitwidth)) for x in taps]
        # taps2_fp = [int(x * 2 ** (bitwidth)) for x in taps2]

        sig, sample_rate = sf.read('IR.wav')
        taps_fp = [int(x * 2 ** (bitwidth)) for x in sig[:self.tapcount, 0]]
        taps2_fp = [int(x * 2 ** (bitwidth)) for x in sig[:self.tapcount, 1]]

        assert sample_rate == samplerate, f"Unsupported samplerate {sample_rate} for IR file expected {samplerate}"

        self.taps1_memories = Array(Memory(width=self.bitwidth, depth=self.slice_size, name=f"taps1_memory_{i}") for i in range(self.slices))
        self.taps2_memories = Array(Memory(width=self.bitwidth, depth=self.slice_size, name=f"taps2_memory_{i}") for i in range(self.slices))
        self.samples1_memories = Array(Memory(width=self.bitwidth, depth=self.slice_size, name=f"samples1_memory_{i}") for i in range(self.slices))
        self.samples2_memories = Array(Memory(width=self.bitwidth, depth=self.slice_size, name=f"samples2_memory_{i}") for i in range(self.slices))


        for i in range(self.slices):
            self.taps1_memories[i].init = taps_fp[i * self.slice_size:(i + 1) * self.slice_size]
            self.taps2_memories[i].init = taps2_fp[i * self.slice_size:(i + 1) * self.slice_size]

        self.fsm_state_out = Signal(5)

    def insert_sample(self, m: Module, value: int, offset_var, memories: [], memory_number: int, sample_index: int):
        m.d.sync += offset_var.eq(offset_var+1)
        with m.If(offset_var == self.tapcount):
            m.d.sync += offset_var.eq(1)

        m.d.comb += [
            memory_number.eq(offset_var // self.slice_size),
            sample_index.eq(offset_var % self.slice_size),
            memories[memory_number].data.eq(self.signal_in.payload),
            memories[memory_number].addr.eq(sample_index),
            memories[memory_number].en.eq(1)
        ]

    # we use the array indices flipped, ascending from zero
    # so x[0] is x_n, x[1] is x_n-
    # 1, x[2] is x_n-2 ...
    # in other words: higher indices are past values, 0 is most recent
    def get_sample_at(self, m: Module, index: int, offset_var: int, memories: [], memory_number: int, sample_index: int):
        with m.If(((offset_var - 1) - index) >=  0):
            m.d.comb += [
                memory_number.eq(((offset_var - 1) - index) // self.slice_size),
                sample_index.eq(((offset_var - 1) - index) % self.slice_size),
                memories[memory_number].addr.eq(sample_index),
            ]
        with m.Else():
            m.d.comb += [
                memory_number.eq(((offset_var - 1) - index + self.tapcount) // self.slice_size),
                sample_index.eq(((offset_var - 1) - index + self.tapcount) % self.slice_size),
                memories[memory_number].addr.eq(sample_index),
            ]

    def elaborate(self, platform) -> Module:
        m = Module()

        taps1_read_ports = Array(self.taps1_memories[i].read_port() for i in range(self.slices))
        taps2_read_ports = Array(self.taps2_memories[i].read_port() for i in range(self.slices))
        samples1_write_ports = Array(self.samples1_memories[i].write_port() for i in range(self.slices))
        samples2_write_ports = Array(self.samples2_memories[i].write_port() for i in range(self.slices))
        samples1_read_ports = Array(self.samples1_memories[i].read_port() for i in range(self.slices))
        samples2_read_ports = Array(self.samples2_memories[i].read_port() for i in range(self.slices))

        m.submodules += [taps1_read_ports, taps2_read_ports, samples1_write_ports, samples2_write_ports, samples1_read_ports, samples2_read_ports]

        memory1_number = Array(Signal(range(self.slices), name=f"memory1_number_{i}") for i in range(self.slices))
        memory2_number = Array(Signal(range(self.slices), name=f"memory2_number_{i}") for i in range(self.slices))
        sample1_indexes = Array(Signal(range(self.slice_size), name=f"sample1_index_{i}") for i in range(self.slices))
        sample2_indexes = Array(Signal(range(self.slice_size), name=f"sample2_index_{i}") for i in range(self.slices))

        sample_values = Array(Signal(signed(self.bitwidth),name=f"sample_values_{i}") for i in range(self.slices*2))
        taps_values = Array(Signal(signed(self.bitwidth),name=f"taps_values_{i}") for i in range(self.slices*2))
        madd_values = Array(Signal(signed(self.bitwidth),name=f"madd_values_{i}") for i in range(self.slices*2))

        sumSignalL = Signal(signed(self.bitwidth))
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
                samples1_write_ports[i].en.eq(0),
                samples2_write_ports[i].en.eq(0),
            ]

        m.d.comb += self.signal_out.valid.eq(0)
        m.d.comb += self.signal_in.ready.eq(0)


        set1 = Signal()
        set2 = Signal()
        with m.FSM(reset="IDLE"):
            with m.State("IDLE"):
                m.d.comb += self.fsm_state_out.eq(1)
                with m.If(self.signal_in.valid & self.signal_in.last & ~set2): # right channel
                    self.insert_sample(m, self.signal_in, offset_var=offset2, memories=samples2_write_ports,
                                       memory_number=memory2_number[0], sample_index=sample2_indexes[0])
                    m.d.sync += set2.eq(1)

                with m.Elif(self.signal_in.valid & self.signal_in.first & ~set1): #left channel
                    self.insert_sample(m, self.signal_in, offset_var=offset, memories=samples1_write_ports,
                                       memory_number=memory1_number[0], sample_index=sample1_indexes[0])
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
                    self.get_sample_at(m, ix + self.slice_size * i, offset_var=offset, memories=samples1_read_ports,
                                       memory_number=memory1_number[i], sample_index=sample1_indexes[i])
                    self.get_sample_at(m, ix + self.slice_size * i, offset_var=offset2, memories=samples2_read_ports,
                                       memory_number=memory2_number[i], sample_index=sample2_indexes[i])

                    m.d.comb += [
                        taps1_read_ports[i].addr.eq(ix),
                        taps2_read_ports[i].addr.eq(ix),
                        sample_values[i].eq(samples1_read_ports[memory1_number[i]].data),
                        sample_values[i + self.slices].eq(samples2_read_ports[memory2_number[i]].data),
                        taps_values[i].eq(taps1_read_ports[i].data),
                        taps_values[i + self.slices].eq(taps2_read_ports[i].data),
                    ]

                #for i in range(self.slices*2):
                #    m.d.sync += madd_values[i].eq(madd_values[i] + (a_values[i] * b_values[i]))
                # this is probably it - if it fits in the available LABS:
                for i in range(self.slices):
                    m.d.sync += madd_values[i].eq(
                        madd_values[i] + (sample_values[i] * taps_values[i] >> self.bitwidth)  #left
                        + (sample_values[i+self.slices] * taps_values[i+self.slices] >> self.bitwidth) #right_bleed
                    )
                    m.d.sync += madd_values[i+self.slices].eq(
                        madd_values[i+self.slices] + (sample_values[i+self.slices] * taps_values[i] >> self.bitwidth)  #right
                        + (sample_values[i] * taps_values[i+self.slices] >> self.bitwidth) #left_bleed
                    )

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
                    self.signal_out.payload.eq(sumSignalL),
                    self.signal_out.valid.eq(1),
                    self.signal_out.first.eq(1),
                    self.signal_out.last.eq(0),
                ]
                with m.If(self.signal_out.ready):
                    m.next = "OUT_RIGHT"

            with m.State("OUT_RIGHT"):
                m.d.comb += self.fsm_state_out.eq(5)
                m.d.comb += [
                    self.signal_out.payload.eq(sumSignalR),
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
