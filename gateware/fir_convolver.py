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
        self.tapcount = 256
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
            m.d.sync += offset.eq(0)
        #memory[offset] = value
        m.d.sync += [
            memory.data.eq(self.signal_in.payload),
            memory.addr.eq(offset),
            memory.en.eq(1)
        ]

    def get_at(self, m: Module, index: int, offset: int, memory: ReadPort):
        with m.If(offset - index > 0):
            m.d.sync += memory.addr.eq(offset - (index - 1))
        with m.Else():
            m.d.sync += memory.addr.eq(offset - (index-1) + self.tapcount)

    def elaborate(self, platform) -> Module:
        m = Module()

        taps_write_port = self.taps.write_port()
        taps_read_port1 = self.taps.read_port(domain='comb')
        taps_read_port2 = self.taps.read_port(domain='comb')
        taps_read_port3 = self.taps.read_port(domain='comb')
        taps_read_port4 = self.taps.read_port(domain='comb')
        m.submodules += [taps_write_port, taps_read_port1, taps_read_port2, taps_read_port3, taps_read_port4]

        samples_write_port = self.samples.write_port()
        samples_read_port1 = self.samples.read_port(domain='comb')
        samples_read_port2 = self.samples.read_port(domain='comb')
        samples_read_port3 = self.samples.read_port(domain='comb')
        samples_read_port4 = self.samples.read_port(domain='comb')
        m.submodules += [samples_write_port, samples_read_port1,samples_read_port2,samples_read_port3,samples_read_port4]

        #n = 1024 #len(self.taps)
        slices = 4
        width = self.bitwidth + self.fraction_width
        #taps = Array(Const(n, signed(width)) for n in self.taps)

        # we use the array indices flipped, ascending from zero
        # so x[0] is x_n, x[1] is x_n-
        # 1, x[2] is x_n-2 ...
        # in other words: higher indices are past values, 0 is most recent
        #x = Array(Signal(signed(width), name=f"x{i}") for i in range(n))

        m.d.sync += [
            samples_write_port.en.eq(0),
            taps_write_port.en.eq(0)
        ]
        #if self.mac_loop:
        ix = Signal(range(self.tapcount + 1))
        madd1 = Signal(signed(self.bitwidth + self.fraction_width))
        #madd1a = Signal.like(madd1)
        #madd1b = Signal.like(madd1)
        madd2 = Signal.like(madd1)
        madd3 = Signal.like(madd1)
        madd4 = Signal.like(madd1)
        #madd5 = Signal(signed(self.bitwidth))
        a1 = Signal(signed(self.bitwidth))
        a2 = Signal.like(a1)
        a3 = Signal.like(a1)
        a4 = Signal.like(a1)
        #a5 = Signal.like(a1)
        b1 = Signal(signed(self.bitwidth))
        b2 = Signal.like(b1)
        b3 = Signal.like(b1)
        b4 = Signal.like(b1)
        #b5 = Signal.like(b1)
        offset = Signal(Shape.cast(range(self.tapcount)).width)  #Signal(unsigned(range(self.tapcount)))

        with m.FSM(reset="IDLE"):
            with m.State("IDLE"):
                m.d.comb += self.signal_in.ready.eq(1)
                with m.If(self.signal_in.valid & self.signal_in.last):
                    m.d.sync += [
                        self.signal_out.valid.eq(0),
                        ix.eq(offset-1),
                        samples_read_port1.addr.eq(ix),
                        samples_read_port2.addr.eq(ix + self.tapcount//slices * 1),
                        samples_read_port3.addr.eq(ix + self.tapcount//slices * 2),
                        samples_read_port4.addr.eq(ix + self.tapcount//slices * 3),

                        taps_read_port1.addr.eq(0),
                        taps_read_port2.addr.eq(self.tapcount//slices * 1),
                        taps_read_port3.addr.eq(self.tapcount//slices * 2),
                        taps_read_port4.addr.eq(self.tapcount//slices * 3),
                        madd1.eq(0),
                        madd2.eq(0),
                        madd3.eq(0),
                        madd4.eq(0),
                    ]
                    m.next = "MAC"
                with m.Elif(self.signal_in.valid & self.signal_in.first):
                    m.d.sync += [
                        self.signal_out.payload.eq(self.signal_in.payload),
                        self.signal_out.valid.eq(1),
                        self.signal_out.first.eq(1),
                        self.signal_out.last.eq(0),
                    ]

            with m.State("MAC"):
                m.d.sync += [
                    self.signal_out.valid.eq(0),
                    madd1.eq(madd1 + ((a1 * b1))),
                    #madd1a.eq(a1 * b1),
                    #madd1b.eq((a1 * b1) >> self.fraction_width),
                    madd2.eq(madd2 + ((a2 * b2))),
                    madd3.eq(madd3 + ((a3 * b3))),
                    madd4.eq(madd4 + ((a4 * b4))),
                ]
                m.d.comb += self.signal_in.ready.eq(0)
                #with m.If(ix == n/slices):
                with m.If(ix == self.tapcount//slices):
                    m.next = "OUTPUT"
                with m.Else():
                    self.get_at(m, ix + 0,                         offset=offset, memory=samples_read_port1)
                    self.get_at(m, ix + self.tapcount//slices * 1, offset=offset, memory=samples_read_port2)
                    self.get_at(m, ix + self.tapcount//slices * 2, offset=offset, memory=samples_read_port3)
                    self.get_at(m, ix + self.tapcount//slices * 3, offset=offset, memory=samples_read_port4)

                    m.d.sync += [
                        taps_read_port1.addr.eq(ix+0),
                        taps_read_port2.addr.eq(ix + self.tapcount//slices * 1),
                        taps_read_port3.addr.eq(ix + self.tapcount//slices * 2),
                        taps_read_port4.addr.eq(ix + self.tapcount//slices * 3),
                        ix.eq(ix + 1)
                    ]
                    m.d.comb += [
                        a1.eq(samples_read_port1.data),
                        a2.eq(samples_read_port2.data),
                        a3.eq(samples_read_port3.data),
                        a4.eq(samples_read_port4.data),
                        b1.eq(taps_read_port1.data),
                        b2.eq(taps_read_port2.data),
                        b3.eq(taps_read_port3.data),
                        b4.eq(taps_read_port4.data),
                    ]

            with m.State("OUTPUT"):
                m.d.comb += self.signal_in.ready.eq(0)

                m.d.sync += [
                    self.signal_out.payload.eq((madd1 + madd2 + madd3 + madd4) >> self.fraction_width),
                    self.signal_out.valid.eq(1),
                    self.signal_out.first.eq(0),
                    self.signal_out.last.eq(1),
                ]
                m.next = "IDLE"

        #else:
        #    m.d.comb += self.signal_out.eq(
        #        sum([((x[i] * taps[i]) >> self.fraction_width) for i in range(n)]))

        with m.If(self.signal_in.valid & self.signal_in.ready):
            #m.d.sync += [x[i + 1].eq(x[i]) for i in range(n - 1)]
            #m.d.sync += x[0].eq(self.signal_in)
            self.insert(m, self.signal_in, offset=offset, memory=samples_write_port )
            #m.d.sync += [
            #    samples_write_port.data.eq(self.signal_in),
            #    samples_write_port.addr.eq(ix),
            #    samples_write_port.en.eq(1)
            #]

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
        #for _ in range(20): yield
        yield dut.signal_in.payload.eq(max)
        yield from self.wait_ready(dut)
        yield dut.signal_in.payload.eq(2222)
        for _ in range(5): yield
        yield dut.signal_in.valid.eq(0)
        for _ in range(60): yield
        yield dut.signal_in.valid.eq(1)
        for _ in range(60): yield
        yield dut.signal_in.payload.eq(0)
        #for _ in range(100): yield
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
