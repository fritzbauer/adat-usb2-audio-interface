#!/usr/bin/env python3
#
# Copyright (c) 2022 Rouven Broszeit <roubro1991@gmx.de>
# SPDX-License-Identifier: CERN-OHL-W-2.0

from amaranth import Elaboratable, Module, Signal, Array, Memory, signed, Cat, Mux, Shape
from amaranth.sim import Tick
from amlib.stream import StreamInterface
from amlib.test import GatewareTestCase, sync_test_case

import numpy as np
import math
from enum import Enum
from fftgen_wrapper import FFTGenWrapper, FFTGenWrapperTest

class ConvolutionMode(Enum):
    CROSSFEED = 1
    STEREO = 2
    MONO = 3

class StereoConvolutionFFT(Elaboratable):
    """A stereo convolution module which uses the MAC (Multiply-Accumulate) algorithm

        Parameters
        ----------
        taps : int[][]
            A two dimensional numpy array containing the stereo impulse response data: np.zeros((tapcount, 2), dtype=np.int32)

        samplerate : int
            The samplerate of the signal.

        clockfrequency : int
            The frequency of the sync domain. This is needed to evaluate how many parallel multiplications need to
            be done in order to process the data in realtime.

        bitwidth : int
            The bitwidth of the signal.

        convolutionMode : ConvolutionMode
            Either of:
            CROSSFEED (1)
                Applies the IR data as crossfeed
                Channel 1 = Channel 1 * IR-Channel1 + Channel2 * IR-Channel2
                Channel 2 = Channel 2 * IR-Channel1 + Channel1 * IR-Channel2
            STEREO (2)
                Applies each channel of the IR data to each signal channel:
                Channel 1 = Channel 1 * IR-Channel1
                Channel 2 = Channel2 * IR-Channel2
            MONO (3)
                Applies only the first IR channel two both signal channels:
                Channel 1 = Channel 1 * IR-Channel1
                Channel 2 = Channel 2 * IR-Channel1


        Attributes
        ----------
        signal_in : StreamInterface
        signal_out : StreamInterface
    """
    def __init__(self,
                 taps: []=np.zeros((16,2)),
                 samplerate:     int=48000,
                 clockfrequency:  int=60e6,
                 bitwidth:       int=24,
                 convolutionMode: ConvolutionMode=ConvolutionMode.MONO,
                 debug=False, testing=False) -> None:

        self.signal_in  = StreamInterface(name="signal_stream_in", payload_width=bitwidth)
        self.signal_out = StreamInterface(name="signal_stream_out", payload_width=bitwidth)

        #self._tapcount = 32
        self._tapcount = len(taps) #4096 synthesizes
        self._bitwidth = bitwidth
        self._convolutionMode = convolutionMode
        self._debug = debug
        self._testing = testing

        #self._stepsize = 8
        self._stepsize = 128  # fftsize-tapcount +1 #or 32
        self._fftsize = 2 << (self._stepsize - 1).bit_length()
        self._slices = self._tapcount // self._stepsize
        self._fft_output_bitwidth = self._bitwidth + Shape.cast(range(self._fftsize)).width

        print(f"Creating {self._slices} slices for {self._tapcount} taps. Stepsize: {self._stepsize}; FFTSize: {self._fftsize}")
        assert self._tapcount % self._slices == 0, f"Tapcount {self._tapcount} cannot be evenly distributed on {self._slices} slizes."

        self._taps1_memory = Memory(width=self._fft_output_bitwidth*2, depth=self._slices*self._fftsize)
        self._taps2_memory = Memory(width=self._fft_output_bitwidth*2, depth=self._slices*self._fftsize)
        self._samples1_memory = Memory(width=self._fft_output_bitwidth*2, depth=self._slices*self._fftsize)
        self._samples2_memory = Memory(width=self._fft_output_bitwidth*2, depth=self._slices*self._fftsize)
        # The buffer is used for multiple purposes:
        #   * Store the right input samples
        #   * Store the left output samples
        #   * Store the left-right interleaved output samples
        self._buffer1_memory = Memory(width=self._fft_output_bitwidth*2, depth=self._fftsize)
        self._buffer2_memory = Memory(width=self._fft_output_bitwidth * 2, depth=self._fftsize)

        #tapsfft1 = np.zeros((self._slices * self._fftsize), dtype=np.int64)
        #tapsfft2 = np.zeros((self._slices * self._fftsize), dtype=np.int64)

        tapsfft1 = [0] * (self._slices * self._fftsize)
        tapsfft2 = [0] * (self._slices * self._fftsize)
        for j in range(self._slices):
            #print(f"stepsize: {self._stepsize}")
            #print(f"fftsize: {self._fftsize}")
            #print(np.shape(taps[j * self._stepsize:(j + 1) * self._stepsize, 0]))
            #print(np.shape(np.abs(np.fft.rfft(taps[j * self._stepsize:(j + 1) * self._stepsize, 0], n=self._fftsize))))
            #print(np.abs(np.fft.rfft(taps[j * self._stepsize:(j + 1) * self._stepsize, 0], n=self._fftsize)))
            t1 = np.abs(np.fft.rfft(taps[j * self._stepsize:(j + 1) * self._stepsize, 0], n=self._fftsize))
            t2 = np.abs(np.fft.rfft(taps[j * self._stepsize:(j + 1) * self._stepsize, 1], n=self._fftsize))
            for k in range(len(t1)):
                tapsfft1[j * self._fftsize + k] = int(t1[k].real) << self._fft_output_bitwidth + int(t1[k].imag)
                tapsfft2[j * self._fftsize + k] = int(t2[k].real) << self._fft_output_bitwidth + int(t2[k].imag)
            #tapsfft1[j*self._fftsize: (j+1)*self._fftsize] = int(np.abs(np.fft.rfft(taps[j * self._stepsize:(j + 1) * self._stepsize, 0], n=self._fftsize)))
            #tapsfft2[j*self._fftsize: (j+1)*self._fftsize] = np.fft.rfft(taps[j * self._stepsize:(j + 1) * self._stepsize, 1], n=self._fftsize)

        # TODO: This does not synthesize for larger FFT sizes, but works for smaller?!
        self._taps1_memory.init = tapsfft1
        self._taps2_memory.init = tapsfft2

        self.fsm_state = Signal(4)
        self.dbg_fft_out_counter = Signal(range(self._fftsize))
        self.dbg_in_out_counter = Signal(range(self._fftsize))
        self.dbg_madd1 = Signal(self._fft_output_bitwidth * 2 * 2)

        if not self._testing:
            self.fft = FFTGenWrapper(self._fftsize, self._bitwidth, self._fft_output_bitwidth, 25, inverse=False)
        else:
            self.fft = FFTGenWrapperTest(self._fftsize, self._bitwidth, self._fft_output_bitwidth)

    def elaborate(self, platform) -> Module:
        m = Module()

        if False: #just connect the FFT to the inputs and outputs for some ILA analyze of the FFT
            m.submodules.fft = fft = self.fft

            valid = Signal()
            with m.If(self.signal_in.valid):
                m.d.sync += valid.eq(1)

            m.d.comb += [
                self.signal_in.ready.eq(1),
                fft.sample_in.eq(self.signal_in.payload),
                fft.valid_in.eq(valid),
                self.signal_out.valid.eq(fft.valid_out),
                self.signal_out.payload.eq(fft.sample_out),
            ]

            in_out_counter = Signal(range(self._fftsize + 1))
            fft_out_counter = Signal.like(in_out_counter)
            mac_counter = Signal.like(in_out_counter)
            slices_counter = Signal(range(self._slices))

            m.d.comb += [
                self.dbg_fft_out_counter.eq(fft_out_counter),
                self.dbg_in_out_counter.eq(in_out_counter),
            ]

            return m

        taps1_read_port = self._taps1_memory.read_port()
        taps2_read_port = self._taps2_memory.read_port()
        samples1_write_port = self._samples1_memory.write_port()
        samples2_write_port = self._samples2_memory.write_port()
        samples1_read_port = self._samples1_memory.read_port()
        samples2_read_port = self._samples2_memory.read_port()
        m.submodules += [taps1_read_port, taps2_read_port, samples1_write_port, samples2_write_port, samples1_read_port, samples2_read_port]

        buffer1_write_port = self._buffer1_memory.write_port()
        buffer2_write_port = self._buffer2_memory.write_port()
        buffer1_read_port = self._buffer1_memory.read_port()
        buffer2_read_port = self._buffer2_memory.read_port()
        m.submodules += [buffer1_write_port, buffer2_write_port, buffer1_read_port, buffer2_read_port]

        m.submodules.fft = fft = self.fft

        read_fft = Signal(2)
        in_out_counter = Signal(range(self._fftsize+1))
        fft_out_counter = Signal.like(in_out_counter)
        mac_counter = Signal.like(in_out_counter)
        slices_counter = Signal(range(self._slices))

        madd1 = Signal(self._fft_output_bitwidth * 2 * 2)
        madd2 = Signal.like(madd1)
        carryover1 = Signal(self._fft_output_bitwidth * 2)
        carryover1_2 = Signal.like(carryover1)
        carryover2 = Signal.like(carryover1)
        carryover2_2 = Signal.like(carryover1)

        if self._debug:
            m.d.comb += [
                self.dbg_fft_out_counter.eq(fft_out_counter),
                self.dbg_in_out_counter.eq(in_out_counter),
                self.dbg_madd1.eq(madd1),
            ]

        if self._testing:
            gotSamples = Signal(range(self._fftsize*3))
            fftCount = Signal.like(gotSamples)
            m.d.comb += [
                gotSamples.eq(fft.gotSamples),
                fftCount.eq(fft.fftCount),
            ]

        m.d.comb += [
            self.signal_in.ready.eq(0),
            fft.valid_in.eq(0),
            #buffer_read_port.addr.eq(fft_out_counter),
        ]



        m.d.comb += [
            # self.signal_out.valid.eq(0),
            samples1_write_port.en.eq(0),
            samples2_write_port.en.eq(0),
            buffer1_write_port.en.eq(0),
            buffer2_write_port.en.eq(0),
            self.signal_out.valid.eq(0),
            self.signal_out.first.eq(0),
            self.signal_out.last.eq(0),
        ]

        with m.FSM(reset="READ_SAMPLES"):

            # READ_SAMPLES state passes on the samples of the first channel to the fft
            # and stores the channels of the second channel in the buffer_memory
            with m.State("READ_SAMPLES"):
                m.d.comb += self.fsm_state.eq(1)

                with m.If(in_out_counter < self._stepsize): #offby1? stepsize-1?
                    m.d.comb += self.signal_in.ready.eq(1)
                    with m.If(self.signal_in.valid):  # & self.signal_in.ready):
                        with m.If(in_out_counter == 0):
                            m.d.comb += [
                                fft.reset_in.eq(1)
                            ]

                        with m.If(self.signal_in.first):
                            m.d.comb += [
                                fft.valid_in.eq(1),
                                fft.sample_in.eq(self.signal_in.payload << self._bitwidth),
                            ]

                        with m.Elif(self.signal_in.last):
                            m.d.comb += [
                                buffer2_write_port.en.eq(1),
                                buffer2_write_port.addr.eq(in_out_counter),
                                buffer2_write_port.data.eq(self.signal_in.payload << self._bitwidth),
                            ]

                            m.d.sync += [
                                in_out_counter.eq(in_out_counter + 1)
                            ]
                with m.Else():
                    m.d.sync += [
                        fft_out_counter.eq(0),
                        read_fft.eq(0),
                    ]
                    m.next = "LEFT_FFT_PADDING"
            # This state writes the zero padding to the left channels fft and starts reading the FFT output
            with m.State("LEFT_FFT_PADDING"):
                m.d.comb += self.fsm_state.eq(2)
                with m.If(in_out_counter < self._fftsize-1):#offby1? stepsize-1?
                    m.d.comb += [
                        fft.valid_in.eq(1),
                        fft.sample_in.eq(0), #zero padding
                    ]
                    m.d.sync += [
                        in_out_counter.eq(in_out_counter + 1),
                    ]

                    with m.If((fft.valid_out | (read_fft == 1)) & fft.valid_in):
                        m.d.comb += [
                            samples1_write_port.en.eq(1),
                            samples1_write_port.addr.eq(fft_out_counter),
                            samples1_write_port.data.eq(fft.sample_out),
                        ]

                        m.d.sync += [
                            fft_out_counter.eq(fft_out_counter + 1),
                            read_fft.eq(1),
                        ]
                with m.Else():
                    m.d.sync += [
                        in_out_counter.eq(0),
                    ]
                    m.next = "READ_LEFT_WRITE_RIGHT_FFT"

            with m.State("READ_LEFT_WRITE_RIGHT_FFT"):
                m.d.comb += self.fsm_state.eq(3)

                # write left fft data to memory
                with m.If((fft.valid_out | (read_fft == 1)) & fft.valid_in):
                    m.d.comb += [
                        samples1_write_port.en.eq(1),
                        samples1_write_port.addr.eq(fft_out_counter),
                        samples1_write_port.data.eq(fft.sample_out),
                    ]

                    m.d.sync += [
                        fft_out_counter.eq(fft_out_counter + 1),
                        read_fft.eq(1),
                    ]

                # send right samples to fft
                m.d.comb += [
                    # Mux only needed for easier debugging in GTKWave - confirm it does not read beyond border
                    buffer2_read_port.addr.eq(Mux(in_out_counter < self._stepsize, in_out_counter, 0)),
                    fft.sample_in.eq(Mux(in_out_counter < self._stepsize, buffer2_read_port.data, 0))
                ]

                #with m.If(in_out_counter > 0): # takes 1 cycle to load the data from memory
                m.d.comb += [
                    fft.valid_in.eq(1),
                ]

                # increment counters
                m.d.sync += [
                    #fft_out_counter.eq(fft_out_counter + 1),
                    in_out_counter.eq(in_out_counter + 1),
                ]

                with m.If(fft_out_counter == self._fftsize-1): #offby1? stepsize-1?
                    m.d.sync += fft_out_counter.eq(0)
                    m.next = "RIGHT_FFT_PAD_AND_READ"
            with m.State("RIGHT_FFT_PAD_AND_READ"):
                m.d.comb += self.fsm_state.eq(4)
                # fft_out_counter should be 0 again when entering this stage !!!confirm this
                m.d.comb += [
                    fft.valid_in.eq(1), #we want to continue sending to the FFT 0 values until we got all data for the right channel
                    fft.sample_in.eq(0),
                ]
                with m.If(fft.valid_out | (read_fft == 2)): #right channel
                    m.d.sync += [
                        read_fft.eq(2)
                    ]

                    m.d.comb += [
                        samples2_write_port.en.eq(1),
                        samples2_write_port.addr.eq(fft_out_counter),
                        samples2_write_port.data.eq(fft.sample_out),
                    ]
                    m.d.sync += [
                        fft_out_counter.eq(fft_out_counter + 1)
                    ]

                    #with m.If(fft.valid_out):
                    #    m.d.sync += [
                    #        fft_out_counter.eq(0)
                    #    ]
                with m.Else(): #left channel
                    # write left fft data to memory
                    m.d.comb += [
                        samples1_write_port.en.eq(1),
                        samples1_write_port.addr.eq(fft_out_counter),
                        samples1_write_port.data.eq(fft.sample_out),
                    ]
                    m.d.sync += [
                        fft_out_counter.eq(fft_out_counter + 1)
                    ]

                with m.If(in_out_counter < self._fftsize):
                    # send right padding to fft
                    m.d.comb += [
                        # Mux only needed for easier debugging in GTKWave - confirm it does not read beyond border
                        buffer2_read_port.addr.eq(Mux(in_out_counter < self._stepsize, in_out_counter, 0)),
                        fft.sample_in.eq(Mux(in_out_counter < self._stepsize, buffer2_read_port.data, 0)),

                    ]
                    # increment counters
                    m.d.sync += [
                        in_out_counter.eq(in_out_counter + 1),
                    ]

                with m.If(fft_out_counter == self._fftsize-1):
                    m.d.sync += [
                        mac_counter.eq(0),
                        slices_counter.eq(0),
                        fft_out_counter.eq(0),
                        madd1.eq(0),
                        madd2.eq(0),
                        carryover1.eq(0),
                        carryover2.eq(0),
                        read_fft.eq(0),
                    ]
                    m.next = "MAC"

                    #with m.If(in_out_counter > 0):  # takes 1 cycle to load the data from memory
                    #    m.d.comb += [
                    #        fft.valid_in.eq(1),
                    #    ]
            with m.State("MAC"):
                m.d.comb += self.fsm_state.eq(5)
                #MAC_LEFT_CHANNEL

                #crossfeed
                #this.re * b.re - this.im * b.im,
                #this.re * b.im + this.im * b.re
                samples1_real = samples1_read_port.data[self._bitwidth:self._bitwidth*2]
                samples1_imag = samples1_read_port.data[:self._bitwidth]
                taps1_real = taps1_read_port.data[self._bitwidth:self._bitwidth * 2]
                taps1_imag = taps1_read_port.data[:self._bitwidth]

                samples2_real = samples2_read_port.data[self._bitwidth:self._bitwidth * 2]
                samples2_imag = samples2_read_port.data[:self._bitwidth]
                taps2_real = taps2_read_port.data[self._bitwidth:self._bitwidth * 2]
                taps2_imag = taps2_read_port.data[:self._bitwidth]


                channel1_real = samples1_real * taps1_real - samples1_imag * taps1_imag + samples2_real * taps2_real - samples2_imag * taps2_imag
                channel1_imag = samples1_real * taps1_imag + samples1_imag * taps1_real + samples2_real * taps2_imag + samples2_imag * taps2_real
                channel2_real = samples2_real * taps1_real - samples2_imag * taps1_imag + samples1_real * taps2_real - samples1_imag * taps2_imag
                channel2_imag = samples2_real * taps1_imag + samples2_imag * taps1_real + samples1_real * taps2_imag + samples1_imag * taps2_real

                m.d.comb += [
                    samples1_read_port.addr.eq(slices_counter * self._fftsize + mac_counter),
                    samples2_read_port.addr.eq(slices_counter * self._fftsize + mac_counter),
                    taps1_read_port.addr.eq(slices_counter * self._fftsize + mac_counter),
                    taps2_read_port.addr.eq(slices_counter * self._fftsize + mac_counter),

                    #samples1_write_port.addr.eq(Mux(slices_counter > 1, (slices_counter - 1), 0) * self._fftsize + mac_counter),
                    #samples2_write_port.addr.eq(Mux(slices_counter > 1, (slices_counter - 1), 0) * self._fftsize + mac_counter),
                    samples1_write_port.addr.eq((slices_counter - 1) * self._fftsize + mac_counter),
                    samples2_write_port.addr.eq((slices_counter - 1) * self._fftsize + mac_counter),
                    samples1_write_port.data.eq(carryover1_2),
                    samples2_write_port.data.eq(carryover2_2),
                    samples1_write_port.en.eq(Mux(slices_counter > 1, 1, 0)),
                    samples2_write_port.en.eq(Mux(slices_counter > 1, 1, 0)),
                ]

                m.d.sync += [
                    madd1.eq(madd1 + Cat([channel1_imag, channel1_real])),
                    madd2.eq(madd2 + Cat([channel2_imag, channel2_real])),
                    slices_counter.eq(slices_counter + 1),
                ]
                with m.If(slices_counter > 1):
                    m.d.sync += [
                        carryover1.eq(samples1_read_port.data),
                        carryover1_2.eq(carryover1),
                        carryover2.eq(samples2_read_port.data),
                        carryover2_2.eq(carryover2),
                    ]

                with m.If(slices_counter == self._slices-1): #offby1?

                    #reset FFT when we start the IFFT part since we left it in the middle of some zero padding before
                    with m.If(mac_counter == 0):
                        m.d.comb += fft.reset_in.eq(1)

                    m.d.comb += [
                        fft.valid_in.eq(1),
                        # conjugate for ifft
                        fft.sample_in.eq(
                            Cat(
                                [-1*madd1[0:self._fft_output_bitwidth*2] >> self._fft_output_bitwidth,
                                 madd1[self._fft_output_bitwidth*2: self._fft_output_bitwidth*4] >> self._fft_output_bitwidth]
                            )
                        ),
                    ]

                    m.d.comb += [
                        buffer2_write_port.en.eq(1),
                        buffer2_write_port.addr.eq(mac_counter),
                        buffer2_write_port.data.eq(
                            Cat(
                                [-1 * madd2[0:self._fft_output_bitwidth * 2] >> self._fft_output_bitwidth,
                                 madd2[self._fft_output_bitwidth * 2: self._fft_output_bitwidth * 4] >> self._fft_output_bitwidth]
                            )
                        ),
                    ]

                    m.d.sync += [
                        mac_counter.eq(mac_counter + 1),
                        slices_counter.eq(0),
                        carryover1.eq(0),
                        carryover2.eq(0),
                        madd1.eq(0),
                    ]

                #read left ifft output
                with m.If((fft.valid_out | (read_fft == 1)) & fft.valid_in):
                    m.d.comb += [
                        buffer1_write_port.en.eq(1),
                        buffer1_write_port.addr.eq(fft_out_counter),  # interleaved left channel
                        # TODO: do we need to calculate abs()?
                        buffer1_write_port.data.eq(fft.sample_out[self._fft_output_bitwidth: self._fft_output_bitwidth * 2] >> (self._fft_output_bitwidth - self._bitwidth))
                        # >> Shape.cast(range(self._fftsize)).width),
                    ]
                    m.d.sync += [
                        fft_out_counter.eq(fft_out_counter + 1),
                        read_fft.eq(1),
                    ]

                with m.If(mac_counter == self._fftsize - 1): #offby1?
                    m.d.sync += [
                        in_out_counter.eq(0)
                    ]
                    m.next ="SEND_RIGHT_READ_LEFT_IFFT"
            with m.State("SEND_RIGHT_READ_LEFT_IFFT"):
                m.d.comb += self.fsm_state.eq(6)
                # TODO: Store the MAC samples after _stepsize these will be needed for the next MAC in the next block
                #with m.If(fft.valid_out):

                # send right channel to IFFT
                with m.If(in_out_counter > 0):
                    m.d.comb += [
                        fft.valid_in.eq(1),
                        buffer2_read_port.addr.eq(Mux(in_out_counter < self._fftsize, in_out_counter, 0)),
                        fft.sample_in.eq(Mux(in_out_counter < self._fftsize, buffer2_read_port.data, 0)),
                    ]

                m.d.sync += [
                    in_out_counter.eq(in_out_counter + 1),
                ]

                #read IFFT output
                with m.If((fft.valid_out | (read_fft == 2) ) & fft.valid_in):
                    m.d.comb += [
                        buffer2_write_port.en.eq(1),
                        buffer2_write_port.addr.eq(fft_out_counter),  # interleaved left channel
                        # TODO: do we need to calculate abs()?
                        buffer2_write_port.data.eq(fft.sample_out[self._fft_output_bitwidth: self._fft_output_bitwidth * 2] >> (
                                        self._fft_output_bitwidth - self._bitwidth))
                        # >> Shape.cast(range(self._fftsize)).width),
                    ]
                    m.d.sync += [
                        fft_out_counter.eq(fft_out_counter + 1),
                    ]

                    with m.If(fft.valid_out):
                        m.d.sync += [
                            read_fft.eq(2),
                            fft_out_counter.eq(1),
                        ]

                #read right FFT
                with m.Elif((read_fft == 1) & fft.valid_in):
                    m.d.comb += [
                        buffer1_write_port.en.eq(1),
                        buffer1_write_port.addr.eq(fft_out_counter),  # interleaved left channel
                        # TODO: do we need to calculate abs()?
                        buffer1_write_port.data.eq(fft.sample_out[self._fft_output_bitwidth: self._fft_output_bitwidth * 2] >> (self._fft_output_bitwidth - self._bitwidth))
                        # >> Shape.cast(range(self._fftsize)).width),
                    ]
                    m.d.sync += [
                        fft_out_counter.eq(fft_out_counter + 1),
                        read_fft.eq(1),
                    ]

                #with m.If(slices_counter == self._slices - 1): # offby1?
                #    m.d.comb += [
                #        fft.valid_in.eq(1),
                #        # conjugate for ifft
                #        fft.sample_in.eq(Cat([-1*madd1[0:self._bitwidth], madd1[self._bitwidth: self._bitwidth*2]])),
                #    ]
                #    m.d.sync += [
                #        mac_counter.eq(mac_counter + 1),
                #        slices_counter.eq(0),
                #        madd1.eq(0),
                #    ]

                with m.If((fft_out_counter == self._fftsize - 1) & (read_fft == 2)):  # offby1?
                    m.d.sync += [
                        #fft_out_counter.eq(0),
                        #slices_counter.eq(0),
                        #madd1.eq(0),
                        in_out_counter.eq(0),
                        #mac_counter.eq(0),
                    ]
                    m.next = "OUTPUT_DATA"

            with m.State("OUTPUT_DATA"):
                m.d.comb += self.fsm_state.eq(7)
                with m.If(self.signal_out.ready):
                    m.d.comb += [
                        buffer1_read_port.addr.eq(in_out_counter//2),
                        buffer2_read_port.addr.eq(in_out_counter//2),
                        self.signal_out.valid.eq(1),
                        self.signal_out.payload.eq(Mux(
                            in_out_counter % 2 == 0,
                            buffer1_read_port.data >> Shape.cast(range(self._fftsize)).width,
                            buffer2_read_port.data >> Shape.cast(range(self._fftsize)).width
                        )),
                        self.signal_out.first.eq(in_out_counter % 2 == 0),
                        self.signal_out.last.eq(in_out_counter % 2 == 1),
                    ]
                    m.d.sync += [
                        in_out_counter.eq(in_out_counter + 1),
                    ]

                with m.If(in_out_counter == self._stepsize*2-1):
                    m.d.sync += [
                        in_out_counter.eq(0),
                    ]
                    m.next = "READ_SAMPLES"
        return m


class FFTMock:
    def __init__(self, tapcount, out_bitwidth):
        self._tapcount = tapcount
        self.gotSamples = 0
        self.outputSamples = 0
        self._out_bitwidth = out_bitwidth

    def FFT(self, x, reset):
        if reset:
            self.gotSamples = 0
            self.outputSamples = 0

        output = np.zeros((len(self._tapcount), 2), dtype=np.int32)
        for i in len(output):
            self.gotSamples += 1
            if i > self._tapcount//3*2:
                self.outputSamples += 1

            output[i, 0] = ((10000 + self.gotSamples * 1000 + self.outputSamples) << self._out_bitwidth) \
                            + 30000 + self.gotSamples * 1000 + self.outputSamples

        for i in len(output):
            self.gotSamples += 1
            if i > self._tapcount//3*2:
                self.outputSamples += 1

            output[i, 1] = ((10000 + self.gotSamples * 1000 + self.outputSamples) << self._out_bitwidth) \
                            + 30000 + self.gotSamples * 1000 + self.outputSamples

        return output

class StereoConvolutionFFTTest(GatewareTestCase):
    FRAGMENT_UNDER_TEST = StereoConvolutionFFT
    testSamplecount = 120
    tapcount = 32
    bitwidth = 24
    samplerate = 48000
    # For testing it is unrealistic to process 4000 taps. We specify a slow clockfrequency of 384kHz (48000*32/4) in order to
    # test the calculation for 4 slices. Testing for a single slice would not uncover any parallel-processing issues in the code.
    clockfrequency = samplerate * tapcount / 4 # we want to test for 4 slices

    #some test IR-data
    tapdata1 = [
        8388607, 8388607, -8388608, 7805659, -777420, -2651895, 1181562, -3751702,
        2024355, -1085865, 1194588, -341596, -138844, -133784, -204981, 33373,
        -636104, -988353, -1313180, -851631, -160023, 370339, 391865, 22927,
        -288476, -281780, 6684, 241364, 174375, -151480, -496185, -655125
    ]
    tapdata2 = [
        7881750, 7461102, -1293164, 2060193, 2268606, 1214028, 225034, -1235788,
        486778, 501926, 466836, -94304, 191358, 533261, 402351, 185156,
        111725, 264777, 243255, 136264, 111589, 216495, 296642, 274774,
        243960, 263830, 298115, 283377, 232100, 182950, 140385, 87647
    ]

    taps = np.zeros((tapcount, 2), dtype=np.int32)
    for i in range(len(tapdata1)):
        taps[i, 0] = int(tapdata1[i])
        taps[i, 1] = int(tapdata2[i])

    convolutionMode = ConvolutionMode.CROSSFEED
    FRAGMENT_ARGUMENTS = dict(taps=taps, samplerate=48000, clockfrequency=clockfrequency, bitwidth=bitwidth,
                              convolutionMode=convolutionMode, debug=True, testing=True)

    def wait(self, n_cycles: int):
        for _ in range(n_cycles):
            yield Tick()

    def wait_ready(self, dut):
        yield dut.signal_in.valid.eq(0)
        while (yield dut.signal_in.ready == 0):
            yield from self.wait(1)

    def get_output(self, dut, out_signal):
        while (yield dut.signal_out.valid == 0):
            yield from self.wait(1)

        payload = yield dut.signal_out.payload
        out_signal.append(int.from_bytes(payload.to_bytes(3, 'little', signed=False), 'little', signed=True))  # parse 24bit signed
        yield Tick()
        payload = yield dut.signal_out.payload
        out_signal.append(int.from_bytes(payload.to_bytes(3, 'little', signed=False), 'little', signed=True))  # parse 24bit signed


    def calculate_expected_result(self, taps, testdata, convolutionMode):
        output = np.zeros((len(testdata), 2), dtype=np.int32)

        return output
        fftMock = FFTMock(16,24+4)
        fft = fftMock.FFT(None, False)
        #ifft = fftMock.FFT(None, True)





        output_samples = testdata.copy()
        samplecount = len(testdata[:, 0])
        tapcount = len(taps[:, 0])

        stepsize = 128  # fftsize-tapcount +1 #or 32
        fftsize = 2 << (stepsize - 1).bit_length()
        slices = tapcount // stepsize
        offsets = range(0, samplecount, stepsize)
        previous_samples1 = np.zeros((slices, fftsize), dtype=np.complex128)
        previous_samples2 = np.zeros((slices, fftsize), dtype=np.complex128)

        tapsfft1 = np.zeros((slices, fftsize), dtype=np.complex128)
        tapsfft2 = np.zeros((slices, fftsize), dtype=np.complex128)
        for j in range(slices):
            # tapsfft1[j] = np.fft.fft(taps[j*stepsize:(j+1)*stepsize, 0], n=fftsize)
            # tapsfft2[j] = np.fft.fft(taps[j*stepsize:(j+1)*stepsize, 1], n=fftsize)
            z1 = np.zeros((fftsize))
            z2 = np.zeros((fftsize))
            z1[:stepsize] = taps[j * stepsize:(j + 1) * stepsize, 0]
            z2[:stepsize] = taps[j * stepsize:(j + 1) * stepsize, 1]
            tapsfft1[j] = FFTConvolver.FFT(z1)
            tapsfft2[j] = FFTConvolver.FFT(z2)

        buffer1 = np.zeros(samplecount + fftsize, dtype=np.complex128)
        buffer2 = np.zeros(samplecount + fftsize, dtype=np.complex128)

        for n in offsets:
            if (n // stepsize) % 200 == 0:
                print(f"{n}/{samplecount}")

            np.roll(previous_samples1, fftsize)
            np.roll(previous_samples2, fftsize)
            z1 = np.zeros((fftsize))
            z2 = np.zeros((fftsize))
            z1[:stepsize] = samples[n:n + stepsize, 0]
            z2[:stepsize] = samples[n:n + stepsize, 0]
            previous_samples1[0] = FFTConvolver.FFT(z1)
            previous_samples2[0] = FFTConvolver.FFT(z2)

            slice1 = np.zeros((fftsize), dtype=np.complex128)
            slice2 = np.zeros((fftsize), dtype=np.complex128)

            for j in range(slices):
                convolved1 = previous_samples1[j] * tapsfft1[j] + previous_samples2[j] * tapsfft2[j]
                convolved2 = previous_samples2[j] * tapsfft1[j] + previous_samples1[j] * tapsfft2[j]
                slice1 += convolved1
                slice2 += convolved2
            # print(f"Slice1: {slice1}")

            modifiedsignal1 = FFTConvolver.IFFT(slice1)
            modifiedsignal2 = FFTConvolver.IFFT(slice2)

            # print(f"Modifiedshape: {np.shape(modifiedsignal1)}: {modifiedsignal1}")
            buffer1[n:n + fftsize] += modifiedsignal1
            buffer2[n:n + fftsize] += modifiedsignal2

        for i in range(samplecount):
            # if i < 40:
            #    print(f"Output: res: {int(res[i])}; abs: {np.abs(res[i])}; real: {np.real(res[i])}; img: {np.imag(res[i])}")
            output_samples[i, 0] = int(np.real(buffer1[i])) >> self.bitwidth  # (self.bitwidth+12)
            output_samples[i, 1] = int(np.real(buffer2[i])) >> self.bitwidth  # (self.bitwidth+12)

        #for sample in range(len(testdata)):
        #    sumL = 0
        #    sumR = 0
        #    for tap in range(len(taps)):
        #        if tap > sample:
        #            break
        #        if convolutionMode == ConvolutionMode.CROSSFEED:
        #            sumL += int(testdata[sample - tap, 0]) * int(taps[tap, 0]) + int(testdata[sample - tap, 1]) * int(taps[tap, 1])
        #            sumR += int(testdata[sample - tap, 1]) * int(taps[tap, 0]) + int(testdata[sample - tap, 0]) * int(taps[tap, 1])
        #        elif convolutionMode == ConvolutionMode.STEREO:
        #            sumL += int(testdata[sample - tap, 0]) * int(taps[tap, 0])
        #            sumR += int(testdata[sample - tap, 1]) * int(taps[tap, 1])
        #        elif convolutionMode == ConvolutionMode.MONO:
        #            sumL += int(testdata[sample - tap, 0]) * int(taps[tap, 0])
        #            sumR += int(testdata[sample - tap, 1]) * int(taps[tap, 0])
#
        #    output[sample, 0] = sumL >> self.bitwidth
        #    output[sample, 1] = sumR >> self.bitwidth

        return output

    @sync_test_case
    def test_fir(self):
        dut = self.dut
        testdata_raw = [[812420, 187705], [800807, 152271], [788403, 109422], [789994, 65769], [773819, 12803],
                    [747336, -40589], [744825, -84371], [729641, -141286], [706089, -190230], [687227, -238741],
                    [674577, -293106], [679382, -354421], [673939, -404084], [670470, -448995], [698245, -493213],
                    [727041, -527915], [749620, -566963], [777583, -578647], [793651, -596892], [807524, -608824],
                    [819352, -600195], [813153, -594125], [811380, -574884], [804773, -549522], [803946, -519619],
                    [798627, -484158], [795990, -457567], [784727, -441965], [781253, -423321], [772247, -406696],
                    [737346, -396969], [727344, -397146], [709987, -398310], [691059, -397030], [657034, -408094],
                    [626680, -421474], [602569, -440591], [568337, -465274], [542343, -489621], [503093, -522351],
                    [449579, -560565], [379701, -601463], [310374, -645896], [224866, -689011], [131545, -735663],
                    [16957, -783650], [-111726, -827044], [-241836, -883122], [-370953, -935652], [-485623, -981734],
                    [-583078, -1036719], [-654543, -1084068], [-710559, -1134423], [-733894, -1178457],
                    [-741472, -1222978], [-730563, -1261715], [-713004, -1283693], [-695040, -1300765],
                    [-669049, -1313577], [-656220, -1333732], [-651277, -1349524], [-657046, -1361661],
                    [-666656, -1369770], [-663344, -1368362], [-667356, -1376221], [-675844, -1380697],
                    [-675575, -1378351], [-670833, -1370430], [-663705, -1343927], [-654934, -1325950],
                    [-630174, -1292244], [-601587, -1244248], [-586781, -1199700], [-577894, -1134632],
                    [-576851, -1061743], [-593347, -977451], [-603222, -883179], [-606804, -777615], [-602340, -671055],
                    [-602439, -575690], [-581846, -480232], [-563534, -413668], [-547941, -360161], [-519823, -314832],
                    [-491941, -277750], [-458161, -247667], [-421273, -221640], [-391277, -193937], [-373996, -167258],
                    [-369452, -141699], [-398689, -124104], [-429195, -104058], [-483597, -90449], [-534715, -76876],
                    [-580318, -75685], [-629828, -68031], [-662375, -69266], [-688125, -77877],
                    [-696417, -84852], [-706783, -99115], [-709124, -115825], [-694132, -152064], [-670392, -183572],
                    [-650166, -220606], [-600539, -254463], [-557703, -287120], [-512840, -317225], [-459318, -333891],
                    [-435701, -348684], [-404251, -355299], [-390665, -352357], [-376465, -343641], [-378383, -344794],
                    [-369697, -338782], [-362845, -332500], [-351513, -319435], [-321284, -298249], [-301254, -274747],
                    [-256613, -242747], [-208538, -203862]]

        testdata = np.zeros((self.testSamplecount, 2))
        for i in range(len(testdata_raw)):
            testdata[i, 0] = testdata_raw[i][0]
            testdata[i, 1] = testdata_raw[i][1]

        yield dut.signal_out.ready.eq(1)

        out_signal = []

        #yield from self.wait_ready(dut)
        #yield dut.signal_in.first.eq(1)
        #yield dut.signal_in.last.eq(0)

        for i in range(len(testdata)):
            yield from self.wait_ready(dut)
            yield dut.signal_in.first.eq(1)
            yield dut.signal_in.last.eq(0)
            yield dut.signal_in.payload.eq(int(testdata[i, 0]))
            yield dut.signal_in.valid.eq(1)
            yield Tick()
            yield from self.wait_ready(dut)
            yield dut.signal_in.valid.eq(1)
            yield dut.signal_in.first.eq(0)
            yield dut.signal_in.last.eq(1)
            yield dut.signal_in.payload.eq(int(testdata[i, 1]))
            yield Tick()
            yield from self.wait_ready(dut)
            #yield from self.get_output(dut, out_signal)

        expected_result = self.calculate_expected_result(self.taps, testdata, self.convolutionMode)
        print(f"Length of expected data: {len(expected_result)}")
        print(f"Expected data: {expected_result}")

        print(f"Length of received data: {len(out_signal)/2}")
        print(f"Received data: {out_signal}")

        for i in range(len(expected_result)):
            assert out_signal[i*2] - 1 <= expected_result[i, 0] <= out_signal[i*2] + 1, f"counter was: {i}"
            assert out_signal[i * 2+1] - 1 <= expected_result[i, 1] <= out_signal[i * 2+1] + 1 , f"counter was: {i}"
