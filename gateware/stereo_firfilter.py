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
from fixedpointfirfilter_memory import  FixedPointFIRFilterMemory

class StereoFIRFilter(Elaboratable):
    def __init__(self,
                 samplerate:     int,
                 clockspeed:      int,
                 bitwidth:       int=24,
                 verbose:        bool=True) -> None:

        self.samplerate = samplerate
        self.clockspeed = clockspeed
        self.bitwidth = bitwidth
        self.verbose = verbose
        self.signal_in  = StreamInterface(name="signal_stream_in", payload_width=bitwidth)
        self.signal_out = StreamInterface(name="signal_stream_out", payload_width=bitwidth)

        tapcount = 128
        self.taps1 = signal.firwin(tapcount, cutoff=2000, pass_zero='lowpass', fs=self.samplerate)
        self.taps2 = signal.firwin(tapcount, cutoff=2000, pass_zero='lowpass', fs=self.samplerate)


    def elaborate(self, platform) -> Module:
        m = Module()

        m.submodules.fir1 = fir_left = FixedPointFIRFilterMemory(samplerate=self.samplerate,clockspeed=self.clockspeed, fir=self.taps1,
                                                   bitwidth=self.bitwidth, fraction_width=self.bitwidth, verbose=self.verbose)
        #m.submodules.fir2 = fir_left_bleed = FixedPointFIRFilterMemory(samplerate=self.samplerate, clockspeed=self.clockspeed, fir=self.taps1,
        #                                           bitwidth=self.bitwidth, fraction_width=self.bitwidth,
        #                                           verbose=self.verbose)
        #m.submodules.fir3 = fir_right = FixedPointFIRFilterMemory(samplerate=self.samplerate, clockspeed=self.clockspeed, fir=self.taps2,
        #                                           bitwidth=self.bitwidth, fraction_width=self.bitwidth,
        #                                           verbose=self.verbose)
        #m.submodules.fir4 = fir_right_bleed = FixedPointFIRFilterMemory(samplerate=self.samplerate, clockspeed=self.clockspeed, fir=self.taps2,
        #                                           bitwidth=self.bitwidth, fraction_width=self.bitwidth,
        #                                           verbose=self.verbose)

        left = Signal(self.bitwidth)
        left_bleed = Signal.like(left)
        right = Signal.like(left)
        right_bleed = Signal.like(left)

        set_left = Signal()
        set_left_bleed = Signal.like(set_left)
        set_right = Signal.like(set_left)
        set_right_bleed = Signal.like(set_left)

        #m.d.comb += [
        #    self.signal_in.ready.eq(fir_left.ready_out & fir_right_bleed.ready_out & self.signal_in.first), #left
        #    self.signal_in.ready.eq(fir_right.ready_out & fir_left_bleed.ready_out & self.signal_in.last), #right
#
        #    fir_left.signal_in.eq(self.signal_in.payload),
        #    fir_right_bleed.signal_in.eq(self.signal_in.payload),
        #    fir_right.signal_in.eq(self.signal_in.payload),
        #    fir_right_bleed.signal_in.eq(self.signal_in.payload),
#
        #    fir_left.enable_in.eq(self.signal_in.valid & self.signal_in.first),
        #    fir_left_bleed.enable_in.eq(self.signal_in.valid & self.signal_in.last),
        #    fir_right.enable_in.eq(self.signal_in.valid & self.signal_in.last),
        #    fir_right_bleed.enable_in.eq(self.signal_in.valid & self.signal_in.first),
        #    #output
        #]

        m.d.comb += [
            self.signal_in.ready.eq(fir_left.ready_out | self.signal_in.last),  # left
            fir_left.signal_in.eq(self.signal_in.payload),
            fir_left.enable_in.eq(self.signal_in.valid & self.signal_in.first),
        ]

        with m.FSM(reset="WAIT"):
            with m.State("WAIT"):
                with m.If(fir_left.valid_out):
                    m.d.sync += [
                        left.eq(fir_left.signal_out),
                        set_left.eq(1),
                    ]

                #with m.If(fir_left_bleed.valid_out):
                #    m.d.sync += [
                #        left_bleed.eq(fir_left_bleed.signal_out),
                #        set_left_bleed.eq(1),
                #    ]
#
                #with m.If(fir_right.valid_out):
                #    m.d.sync += [
                #        left.eq(fir_right.signal_out),
                #        set_right.eq(1),
                #    ]
#
                #with m.If(fir_right_bleed.valid_out):
                #    m.d.sync += [
                #        left.eq(fir_right_bleed.signal_out),
                #        set_right_bleed.eq(1),
                #    ]
#
                #with m.If(set_left & set_left_bleed & set_right & set_right_bleed):
                #    m.next = "LEFT_OUT"

                with m.If(set_left):
                    m.next = "LEFT_OUT"

            with m.State("LEFT_OUT"):
                m.d.sync += [
                    #self.signal_out.payload.eq(left + right_bleed),
                    self.signal_out.payload.eq(left),
                    self.signal_out.first.eq(1),
                    self.signal_out.last.eq(0),
                    self.signal_out.valid.eq(1),
                ]
                m.next = "RIGHT_OUT"

            with m.State("RIGHT_OUT"):
                m.d.sync += [
                    #self.signal_out.payload.eq(right + left_bleed),
                    self.signal_out.payload.eq(left),
                    self.signal_out.first.eq(0),
                    self.signal_out.last.eq(1),
                    self.signal_out.valid.eq(1),
                ]

                m.next = "RESET"
            with m.State("RESET"):
                m.d.sync += [
                    set_left.eq(0),
                    set_left_bleed.eq(0),
                    set_right.eq(0),
                    set_right_bleed.eq(0),
                ]


        return m

    def read_wave(file):
        reader = open(file)

        _, sampwidth, framerate, nframes, _, _ = reader.getparams()
        frames = reader.readframes(nframes)

        reader.close()

        dtypes = {1: np.int8, 2: np.int16, 4: np.int32}

        if sampwidth not in dtypes:
            raise ValueError('unsupported sample width')

        data = np.frombuffer(frames, dtype=dtypes[sampwidth])

        num_channels = reader.getnchannels()
        if num_channels == 2:
            data = data[::2]

        return Wave(data, framerate)


class StereoFIRFilterTest(GatewareTestCase):
    FRAGMENT_UNDER_TEST = StereoFIRFilter
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
