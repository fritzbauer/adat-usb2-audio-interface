#!/usr/bin/env python3
#
# Copyright (c) 2021 Hans Baier <hansfbaier@gmail.com>
# SPDX-License-Identifier: CERN-OHL-W-2.0

from scipy import signal
from amaranth import *
from amaranth.hdl.mem import ReadPort, WritePort, Memory
from pprint import pformat
import numpy as np
from amlib.stream import StreamInterface

from amlib.test       import GatewareTestCase, sync_test_case

class FIRConvolver(Elaboratable):
    def __init__(self,
                 samplerate:     int,
                 bitwidth:       int=24,
                 fraction_width: int=24,
                 cutoff_freq:    int=20000,
                 filter_order:   int=24,
                 filter_type:    str='lowpass',
                 weight:         list=None,
                 mac_loop:       bool=True,
                 verbose:        bool=True) -> None:

        #self.enable_in  = Signal()
        #self.ready_out  = Signal()
        self.signal_in  = StreamInterface(name="signal_stream_in", payload_width=bitwidth)#Signal(signed(bitwidth))
        #self.signal_out = Signal(signed(bitwidth))
        self.signal_out = StreamInterface(name="signal_stream_out", payload_width=bitwidth)

        #if type(cutoff_freq) == int:
        cutoff = cutoff_freq / samplerate
        #    taps = signal.firwin(filter_order, cutoff, fs=samplerate, pass_zero=filter_type, window='hamming')
        #elif type(cutoff_freq) == list and len(cutoff_freq) == 2:
        #    Fs = samplerate
        #    Fpb = cutoff_freq[0]
        #    Fsb = cutoff_freq[1]
        #    bands = np.array([0., Fpb/Fs, Fsb/Fs, .5])
        #    pass_zero = filter_type == True or filter_type == 'lowpass'
        #    desired = [1, 0] if pass_zero else [0, 1]
        #    taps = signal.remez(filter_order, bands, desired, weight)
        #else:
        #    raise TypeError('cutoff_freq parameter must be int or list of start/stop band frequencies')
        # convert to fixed point representation
        self.tapcount = 64
        self.slices = 4
        taps = signal.firwin(self.tapcount,2000,pass_zero='lowpass',fs=48000)
        self.bitwidth = bitwidth
        self.fraction_width = fraction_width
        assert bitwidth <= fraction_width, f"Bitwidth {bitwidth} must not exceed {fraction_width}"

        self.taps = Memory(width=self.bitwidth, depth=self.tapcount, name="taps_memory")
        self.taps.init = taps_fp = [int(x * 2**(fraction_width)) for x in taps]

        self.samples = Memory(width=self.bitwidth, depth=self.tapcount, name="samples_memory")

        self.mac_loop = mac_loop

        if verbose:
            if type(cutoff_freq) == int:
                print(f"{filter_order}-order windowed FIR with cutoff: {cutoff * samplerate}")
            else:
                print(f"{filter_order}-order FIR with start/stop band: {cutoff_freq} weight: {weight}")
            print(f"taps: {pformat(taps)}")
            print(f"taps ({bitwidth}.{fraction_width} fixed point): {taps_fp}\n")

        def conversion_error(coeff, fp_coeff):
            val = 2**(bitwidth - 1)
            fp_product = fp_coeff * val
            fp_result  = fp_product >> fraction_width
            fp_error   = fp_result - (coeff * val)
            return fp_error

        num_coefficients = len(taps_fp)
        conversion_errors = [abs(conversion_error(taps[i], taps_fp[i])) for i in range(num_coefficients)]
        if verbose:
            print("a, fixed point conversion errors: {}".format(conversion_errors))
        for i in range(num_coefficients):
            assert (conversion_errors[i] < 1.0)

    def insert(self, m: Module, value: int, offset: int, memory: WritePort):
        m.d.sync += offset.eq(offset+1)
        with m.If(offset == self.tapcount):
            m.d.sync += offset.eq(1)
        #memory[offset] = value
        m.d.sync += [
            memory.data.eq(self.signal_in.payload),
            memory.addr.eq(offset),
            memory.en.eq(1)
        ]

    def get_at(self, m: Module, index: int, offset: int, memory: ReadPort):
        with m.If((offset - 1) - index >= 0):
            m.d.comb += memory.addr.eq((offset - 1) - index)
        with m.Else():
            m.d.comb += memory.addr.eq((offset - 1) - index + self.tapcount)

    def elaborate(self, platform) -> Module:
        m = Module()

        taps_write_port = self.taps.write_port()
        m.submodules += taps_write_port

        samples_write_port = self.samples.write_port()
        m.submodules += samples_write_port

        taps_read_ports = Array(self.taps.read_port(domain='comb') for i in range(self.slices))
        m.submodules += taps_read_ports

        samples_read_ports = Array(self.samples.read_port(domain='comb') for i in range(self.slices))
        m.submodules += samples_read_ports

        a_values = Array(Signal(signed(self.bitwidth),name=f"a{i}") for i in range(self.slices))
        b_values = Array(Signal(signed(self.bitwidth),name=f"b{i}") for i in range(self.slices))
        madd_values = Array(Signal(signed(self.bitwidth + self.fraction_width),name=f"madd{i}") for i in range(self.slices))


        # we use the array indices flipped, ascending from zero
        # so x[0] is x_n, x[1] is x_n-
        # 1, x[2] is x_n-2 ...
        # in other words: higher indices are past values, 0 is most recent
        #x = Array(Signal(signed(width), name=f"x{i}") for i in range(n))

        m.d.sync += [
            samples_write_port.en.eq(0),
            taps_write_port.en.eq(0),
            self.signal_out.valid.eq(0),
        ]

        ix = Signal(range(self.tapcount + 1))
        offset = Signal(Shape.cast(range(self.tapcount)).width)  #Signal(unsigned(range(self.tapcount)))

        for i in range(self.slices):
            self.get_at(m, ix + self.tapcount // self.slices * i, offset=offset, memory=samples_read_ports[i])
            m.d.comb += [
                taps_read_ports[i].addr.eq(ix + self.tapcount // self.slices * i),
                a_values[i].eq(samples_read_ports[i].data),
                b_values[i].eq(taps_read_ports[i].data),
            ]

        m.d.comb += self.signal_in.ready.eq(0)

        with m.FSM(reset="IDLE"):
            with m.State("IDLE"):
                m.d.comb += self.signal_in.ready.eq(1)
                with m.If(self.signal_in.valid & self.signal_in.last): #only convolve right channel
                    self.insert(m, self.signal_in, offset=offset, memory=samples_write_port)

                    m.d.sync += ix.eq(0)

                    for i in range(self.slices):
                        madd_values[i].eq(0)

                    m.next = "STORE"
                with m.Elif(self.signal_in.valid & self.signal_in.first): #just forward unmodified left channel for now
                    m.d.sync += [
                        self.signal_out.payload.eq(self.signal_in.payload),
                        self.signal_out.valid.eq(1),
                        self.signal_out.first.eq(1),
                        self.signal_out.last.eq(0),
                    ]
            with m.State("STORE"):
                m.next = "MAC"
            with m.State("MAC"):
                for i in range(self.slices):
                    m.d.sync += madd_values[i].eq(madd_values[i] + (a_values[i] * b_values[i]))

                with m.If(ix == self.tapcount//self.slices - 1):
                    m.next = "OUTPUT"
                with m.Else():
                    m.d.sync += ix.eq(ix + 1)

            with m.State("OUTPUT"):
                sumS = Signal(self.bitwidth+self.fraction_width)
                sum = 0
                for i in range(self.slices):
                    sum += madd_values[i]

                m.d.comb += sumS.eq(sum)
                m.d.sync += [
                    self.signal_out.payload.eq(sum >> self.fraction_width),
                    self.signal_out.valid.eq(1),
                    self.signal_out.first.eq(0),
                    self.signal_out.last.eq(1),
                ]
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
        yield print(dut.signal_in.ready)
        while (yield dut.signal_in.ready == 0):
            waitcount += 1
            yield from self.wait(1)
        print("Waitcount {}".format(waitcount))

    @sync_test_case
    def test_fir(self):
        dut = self.dut
        max = int(2**15 - 1)
        min = -max
        yield dut.signal_in.last.eq(1)
        yield dut.signal_in.valid.eq(1)
        yield dut.signal_in.payload.eq(max)
        yield from self.wait_ready(dut)
        yield dut.signal_in.payload.eq(2222)
        for _ in range(5): yield
        yield dut.signal_in.valid.eq(0)
        for _ in range(60): yield
        yield dut.signal_in.valid.eq(1)
        for _ in range(60): yield
        yield dut.signal_in.payload.eq(0)
        for i in range(10):
           yield dut.signal_in.payload.eq(i)
           yield from self.wait_ready(dut)
           yield dut.signal_in.payload.eq(i*10)
           yield from self.wait_ready(dut)
           yield dut.signal_in.payload.eq(i*20)
           yield from self.wait_ready(dut)
           yield dut.signal_in.payload.eq(i*30)
           yield from self.wait_ready(dut)
           print("end of loop")
