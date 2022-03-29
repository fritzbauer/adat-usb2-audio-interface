#!/usr/bin/env python3
#
# Copyright (c) 2021 Hans Baier <hansfbaier@gmail.com>
# SPDX-License-Identifier: CERN-OHL-W-2.0

from amaranth import Elaboratable, Module, Signal, Array, Memory, signed, Shape, Cat
from amaranth.sim import Tick
from amlib.stream import StreamInterface

import numpy as np
import math

from amlib.test import GatewareTestCase, sync_test_case
from enum import Enum
import binascii

class ConvolutionMode(Enum):
    CROSSFEED = 1
    STEREO = 2
    MONO = 3

class FIRConvolver(Elaboratable):
    def __init__(self,
                 taps: [],
                 samplerate:     int=48000,
                 clockfrequency:  int=60e6,
                 bitwidth:       int=24,
                 convolutionMode: ConvolutionMode=ConvolutionMode.MONO) -> None:

        self.signal_in  = StreamInterface(name="signal_stream_in", payload_width=bitwidth)
        self.signal_out = StreamInterface(name="signal_stream_out", payload_width=bitwidth)

        self.tapcount = len(taps) #4096 synthesizes
        self.bitwidth = bitwidth
        self.convolutionMode = convolutionMode

        # in order to process more taps than we have clock cycles per sample we will run parallel calculations.
        # The amount of parallel calculations per channel is defined by the "slices" variable
        self.slices = math.ceil(self.tapcount / (clockfrequency / samplerate))  # 4
        self.size_of_slizes = self.bitwidth * self.slices #how many bits per slice
        self.samples_per_slice = self.tapcount // self.slices #how many samples per slice

        print(f"Creating {self.slices} slices for {self.tapcount} taps.")

        assert self.tapcount % self.slices == 0, f"Tapcount {self.tapcount} cannot be evenly distributed on {self.slices} slizes."

        taps_fp = taps[:self.tapcount, 0]
        taps2_fp = taps[:self.tapcount, 1]

        self.taps1_memory = Memory(width=self.size_of_slizes, depth=self.samples_per_slice)
        self.taps2_memory = Memory(width=self.size_of_slizes, depth=self.samples_per_slice)
        self.samples1_memory = Memory(width=self.size_of_slizes, depth=self.samples_per_slice)
        self.samples2_memory = Memory(width=self.size_of_slizes, depth=self.samples_per_slice)

        taps_fp_mod = []
        taps2_fp_mod = []
        for i in range(0,self.tapcount, self.slices):
            val1 = 0
            val2 = 0
            for j in range(self.slices):
                val1 += int(taps_fp[i+j]) << (self.slices-j-1) * self.bitwidth
                val2 += int(taps2_fp[i+j]) << (self.slices-j-1) * self.bitwidth
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

        ix = Signal(range(self.samples_per_slice + 1))

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

                # store new sample for left channel
                with m.If(self.signal_in.valid & self.signal_in.first & ~set1):
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

                # store new sample for right channel
                with m.Elif(self.signal_in.valid & self.signal_in.last & ~set2):
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

                # prepare MAC calculations
                with m.If(set1 & set2):
                    for i in range(self.slices * 2):
                        m.d.sync += [
                            ix.eq(0),
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

                # do the actual MAC calculation
                with m.If(ix <= self.samples_per_slice - 1):
                    for i in range(self.slices):
                        left_sample = samples1_read_port.data[i*self.bitwidth:(i+1)*self.bitwidth].as_signed()
                        right_sample = samples2_read_port.data[i*self.bitwidth:(i+1)*self.bitwidth].as_signed()
                        main_tap = taps1_read_port.data[i*self.bitwidth:(i+1)*self.bitwidth].as_signed()
                        bleed_tap = taps2_read_port.data[i*self.bitwidth:(i+1)*self.bitwidth].as_signed()

                        if self.convolutionMode == ConvolutionMode.CROSSFEED:
                            m.d.sync += [
                                madd_values[i].eq(madd_values[i] + (left_sample * main_tap)  # left
                                                + (right_sample * bleed_tap)),  # right_bleed
                                madd_values[i + self.slices].eq(madd_values[i + self.slices] + (right_sample * main_tap) # right
                                                + (left_sample * bleed_tap))  # left_bleed
                            ]
                            break
                        elif self.convolutionMode == ConvolutionMode.STEREO:
                            m.d.sync += [
                                madd_values[i].eq(madd_values[i] + (left_sample * main_tap)),
                                madd_values[i + self.slices].eq(madd_values[i + self.slices] + (right_sample * bleed_tap)),
                            ]
                        elif self.convolutionMode == ConvolutionMode.MONO:
                            m.d.sync += [
                                madd_values[i].eq(madd_values[i] + (left_sample * main_tap)),
                                madd_values[i + self.slices].eq(madd_values[i + self.slices] + (right_sample * main_tap)),
                            ]

                # shift the samples buffer by one sample to prepare for the next arriving sample in the IDLE state
                with m.If(ix > 1):
                    m.d.sync += [
                        samples1_write_port.data.eq(Cat((previous_sample1 >> self.bitwidth)[:((self.slices-1)*self.bitwidth)], carryover1_2)),
                        samples1_write_port.addr.eq(ix - 2),
                        samples1_write_port.en.eq(1),

                        samples2_write_port.data.eq(Cat((previous_sample2 >> self.bitwidth)[:((self.slices-1)*self.bitwidth)], carryover2_2)),
                        samples2_write_port.addr.eq(ix - 2),
                        samples2_write_port.en.eq(1),
                    ]

                with m.If(ix == self.samples_per_slice + 1):
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
                    self.signal_out.payload.eq(sumSignalL >> self.bitwidth),
                    self.signal_out.valid.eq(1),
                    self.signal_out.first.eq(1),
                    self.signal_out.last.eq(0),
                ]
                with m.If(self.signal_out.ready):
                    m.next = "OUTPUT RIGHT"

            with m.State("OUTPUT RIGHT"):
                m.d.comb += self.fsm_state_out.eq(5)
                m.d.sync += [
                    self.signal_out.payload.eq(sumSignalR >> self.bitwidth),
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
    samplerate = 48000
    clockfrequency = samplerate * tapcount / 4 # we want to test for 4 slices

    #hardcoded 4-tap IR data which helps checking data during simulation
    tapdata1 = [1, 0.5, 0.25, 0.1]
    tapdata2 = [0.05, 0.025, 0.01, 0]
    taps = np.zeros((tapcount, 2), dtype=np.int32)
    for i in range(len(tapdata1)):
        taps[i, 0] = int(tapdata1[i] * 2 ** (bitwidth-1)-1)
        taps[i, 1] = int(tapdata2[i] * 2 ** (bitwidth-1)-1)

    #taps = taps.astype(np.int32)
    FRAGMENT_ARGUMENTS = dict(taps=taps, samplerate=48000, clockfrequency=clockfrequency, bitwidth=bitwidth, convolutionMode=ConvolutionMode.MONO)

    def wait(self, n_cycles: int):
        for _ in range(n_cycles):
            yield Tick()

    def wait_ready(self, dut, out_signal):
        waitcount = 0
        while ((yield dut.signal_out.valid == 0) & (yield dut.signal_in.ready == 0)):
            waitcount += 1
            yield from self.wait(1)
        if (yield dut.signal_out.valid == 1):
            payload = yield dut.signal_out.payload
            out_signal.append(int.from_bytes(payload.to_bytes(3, 'little', signed=False), 'little',
                                             signed=True))  # parse 24bit signed
            yield Tick()
            payload = yield dut.signal_out.payload
            out_signal.append(int.from_bytes(payload.to_bytes(3, 'little', signed=False), 'little',
                                             signed=True))  # parse 24bit signed
        #print("Waitcount {}".format(waitcount))


    def calculate_expected_result(self, taps, testdata):
        output = np.zeros((len(testdata),2), dtype=np.int32)
        for sample in range(len(testdata)):
            sumL = 0
            sumR = 0
            for tap in range(len(taps)):
                if tap > sample:
                    break
                sumL += int(testdata[sample - tap, 0]) * int(taps[tap, 0])
                sumR += int(testdata[sample - tap, 1]) * int(taps[tap, 0])
            #if sample >= len(testdata) - 1:
            #    print(f"samples: {testdata[sample-1:]}")
            #    print(f"taps: {taps[:5]}")
            #    print(int(testdata[sample - tap, 0]) * int(taps[tap, 0]))
            output[sample, 0] = sumL >> self.bitwidth-1
            output[sample, 1] = sumR >> self.bitwidth-1

        return output

    @sync_test_case
    def test_fir(self):
        dut = self.dut
        #print(self.taps)
        max = int(2**(self.bitwidth-1) - 1)
        min = -max
        testdata = np.zeros((61,2))
        testdata[0,0] = max
        testdata[0,1] = min
        for i in range(1,60):
            testdata[i,0] = int(i+100) #left channel
            testdata[i,1] = int(-i-200) #right channel

        yield dut.signal_out.ready.eq(1)

        out_signal = [] #np.zero((1,2))
        print(len(testdata))
        for i in range(len(testdata)):
            #yield from self.wait_ready(dut, out_signal)
            yield Tick()
            yield dut.signal_in.first.eq(1)
            yield dut.signal_in.last.eq(0)
            yield dut.signal_in.payload.eq(int(testdata[i, 0]))
            yield dut.signal_in.valid.eq(1)
            yield Tick()
            yield dut.signal_in.valid.eq(1)
            yield dut.signal_in.first.eq(0)
            yield dut.signal_in.last.eq(1)
            yield dut.signal_in.payload.eq(int(testdata[i, 1]))
            yield Tick()
            #yield out_signal.append(dut.signal_out.payload)
            yield from self.wait_ready(dut, out_signal)
            #yield from self.wait_ready(dut, out_signal)

        yield from self.wait(10)
        #get the last two samples
        yield from self.wait_ready(dut,out_signal)


        print(len(out_signal))

        print(out_signal)

        while out_signal[0] == 0:
            out_signal.remove(0)

        expected_result = self.calculate_expected_result(self.taps, testdata)
        print(len(expected_result))
        print(expected_result)
        for i in range(len(expected_result)):
            assert out_signal[i*2]-2 <= expected_result[i, 0] <= out_signal[i*2]+2, f"counter was: {i}"
            assert out_signal[i * 2+1]-2 <= expected_result[i, 1] <= out_signal[i * 2+1]+2, f"counter was: {i}"



#class FixedPointFIRFilterTest(GatewareTestCase):
#    FRAGMENT_UNDER_TEST = FIRConvolver
#
#    tapcount = 32
#    bitwidth = 24
#    tapdata = [int(1 * 2 ** (bitwidth-1)-1), int(0.5 * 2 ** (bitwidth-1)-1), int(0.25 * 2 ** (bitwidth-1)-1), int(0.1 * 2 ** (bitwidth-1)-1)]
#
#    taps = np.zeros((tapcount, 2))
#    for i in range(len(tapdata)):
#        taps[i, 0] = int(tapdata[i])
#        taps[i, 1] = int(tapdata[i])
#    taps = taps.astype(np.int32)
#    print(taps)
#    FRAGMENT_ARGUMENTS = dict(taps=taps, samplerate=48000)
#
#    def wait(self, n_cycles: int):
#        for _ in range(n_cycles):
#            yield
#
#    def wait_ready(self, dut):
#        waitcount = 0
#        while (yield dut.signal_in.ready == 0):
#            waitcount += 1
#            yield from self.wait(1)
#        print("Waitcount {}".format(waitcount))
#
#    @sync_test_case
#    def test_fir(self):
#        dut = self.dut
#        max = int(2**15 - 1)
#        min = -max
#        yield dut.signal_out.ready.eq(1)
#        yield from self.wait_ready(dut)
#        yield Tick()
#        yield dut.signal_in.first.eq(1)
#        yield dut.signal_in.last.eq(0)
#        yield dut.signal_in.payload.eq(max)
#        yield dut.signal_in.valid.eq(1)
#        yield from self.wait_ready(dut)
#        yield dut.signal_in.valid.eq(1)
#        yield dut.signal_in.first.eq(0)
#        yield dut.signal_in.last.eq(1)
#        yield dut.signal_in.payload.eq(min)
#        yield Tick()
#        for i in range(60):
#            yield from self.wait_ready(dut)
#            yield dut.signal_in.valid.eq(1)
#            yield dut.signal_in.first.eq(1)
#            yield dut.signal_in.last.eq(0)
#            yield dut.signal_in.payload.eq(i+100)
#            yield Tick()
#            yield from self.wait_ready(dut)
#            yield dut.signal_in.valid.eq(1)
#            yield dut.signal_in.first.eq(0)
#            yield dut.signal_in.last.eq(1)
#            yield dut.signal_in.payload.eq(-i-200)
#            yield Tick()
#            print("end of loop")
#