#!/usr/bin/env python3
#
# Copyright (c) 2021 Hans Baier <hansfbaier@gmail.com>
# SPDX-License-Identifier: CERN-OHL-W-2.0
import os

from amaranth              import *
from amaranth.lib.fifo     import AsyncFIFOBuffered, AsyncFIFO
from amaranth.lib.cdc      import FFSynchronizer

from amaranth_library.stream       import connect_stream_to_fifo
from amaranth_library.io.i2s       import I2STransmitter
from amaranth_library.io.max7219   import SerialLEDArray, NumberToSevenSegmentHex
from amaranth_library.debug.ila    import StreamILA, ILACoreParameters

from luna                import top_level_cli
from luna.usb2           import USBDevice, USBIsochronousInMemoryEndpoint, USBIsochronousOutStreamEndpoint, USBIsochronousInStreamEndpoint

from usb_protocol.types                       import USBRequestType, USBStandardRequests

from luna.gateware.usb.usb2.device            import USBDevice
from luna.gateware.usb.usb2.endpoints.stream  import USBMultibyteStreamInEndpoint
from luna.gateware.usb.usb2.request           import StallOnlyRequestHandler

from adat import ADATTransmitter, ADATReceiver
from adat import EdgeToPulse

from usb_stream_to_channels import USBStreamToChannels
from channels_to_usb_stream import ChannelsToUSBStream
from bundle_multiplexer     import BundleMultiplexer
from bundle_demultiplexer   import BundleDemultiplexer
from requesthandlers        import UAC2RequestHandlers

from usb_descriptors import USBDescriptors

class USB2AudioInterface(Elaboratable):
    """ USB Audio Class v2 interface """
    # one isochronous packet typically has 6 or 7 samples of 8 channels of 32 bit samples
    # 6 samples * 8 channels * 4 bytes/sample = 192 bytes
    # 7 samples * 8 channels * 4 bytes/sample = 224 bytes
    MAX_PACKET_SIZE = 224 * 4

    USE_ILA = False
    ILA_MAX_PACKET_SIZE = 512

    def elaborate(self, platform):
        m = Module()

        number_of_channels      = platform.number_of_channels
        number_of_channels_bits = Shape.cast(range(number_of_channels)).width
        audio_bits              = 24

        m.submodules.car = platform.clock_domain_generator()

        #
        # USB
        #
        ulpi1 = platform.request("ulpi", 1)
        m.submodules.usb1 = usb1 = USBDevice(bus=ulpi1)

        descriptors = USBDescriptors(max_packet_size=self.MAX_PACKET_SIZE, \
                                     number_of_channels=number_of_channels, \
                                     ila_max_packet_size=self.ILA_MAX_PACKET_SIZE, \
                                     use_ila=self.USE_ILA).create_descriptors()

        control_ep = usb1.add_control_endpoint()
        control_ep.add_standard_request_handlers(descriptors, blacklist=[
            lambda setup:   (setup.type    == USBRequestType.STANDARD)
                          & (setup.request == USBStandardRequests.SET_INTERFACE)
        ])

        class_request_handler = UAC2RequestHandlers()
        control_ep.add_request_handler(class_request_handler)

        # Attach class-request handlers that stall any vendor or reserved requests,
        # as we don't have or need any.
        stall_condition = lambda setup : \
            (setup.type == USBRequestType.VENDOR) | \
            (setup.type == USBRequestType.RESERVED)
        control_ep.add_request_handler(StallOnlyRequestHandler(stall_condition))

        usb1_ep1_out = USBIsochronousOutStreamEndpoint(
            endpoint_number=1, # EP 1 OUT
            max_packet_size=self.MAX_PACKET_SIZE)
        usb1.add_endpoint(usb1_ep1_out)

        usb1_ep1_in = USBIsochronousInMemoryEndpoint(
            endpoint_number=1, # EP 1 IN
            max_packet_size=4)
        usb1.add_endpoint(usb1_ep1_in)

        usb1_ep2_in = USBIsochronousInStreamEndpoint(
            endpoint_number=2, # EP 2 IN
            max_packet_size=self.MAX_PACKET_SIZE)
        usb1.add_endpoint(usb1_ep2_in)

        m.d.comb += [
            usb1.connect          .eq(1),
            # Connect our device as a high speed device
            usb1.full_speed_only  .eq(0),
        ]

        audio_in_frame_bytes = \
            self.calculate_usb_input_frame_size(m, usb1_ep1_out, usb1_ep2_in, number_of_channels)

        sof_counter, usb_to_output_fifo_level, usb_to_output_fifo_depth = \
            self.create_sample_rate_feedback_circuit(m, usb1, usb1_ep1_in)

        audio_in_active = self.detect_active_audio_in(m, usb1, usb1_ep2_in)

        #
        # USB <-> Channel Stream conversion
        #
        m.submodules.usb_to_channel_stream = usb_to_channel_stream = \
            DomainRenamer("usb")(USBStreamToChannels(number_of_channels))

        m.submodules.channels_to_usb_stream = channels_to_usb_stream = \
            DomainRenamer("usb")(ChannelsToUSBStream(number_of_channels, max_packet_size=self.MAX_PACKET_SIZE))

        no_channels = Signal(range(number_of_channels * 2), reset=2)
        m.d.comb += [
            usb_to_channel_stream.no_channels_in.eq(no_channels),
            channels_to_usb_stream.no_channels_in.eq(no_channels),
            channels_to_usb_stream.audio_in_active.eq(audio_in_active),
        ]

        with m.Switch(class_request_handler.output_interface_altsetting_nr):
            with m.Case(2):
                m.d.comb += no_channels.eq(number_of_channels)
            with m.Default():
                m.d.comb += no_channels.eq(2)

        m.submodules.usb_to_output_fifo = usb_to_output_fifo = \
            AsyncFIFO(width=audio_bits + number_of_channels_bits + 2, depth=usb_to_output_fifo_depth, w_domain="usb", r_domain="sync")

        m.submodules.bundle_demultiplexer = bundle_demultiplexer = BundleDemultiplexer()
        m.submodules.bundle_multiplexer   = bundle_multiplexer   = DomainRenamer("fast")(BundleMultiplexer())

        adat_transmitters = []
        adat_receivers    = []
        adat_pads         = []
        for i in range(1, 5):
            transmitter = ADATTransmitter(fifo_depth=8)
            setattr(m.submodules, f"adat{i}_transmitter", transmitter)
            adat_transmitters.append(transmitter)

            receiver = DomainRenamer("fast")(ADATReceiver(platform.fast_domain_clock_freq))
            setattr(m.submodules, f"adat{i}_receiver", receiver)
            adat_receivers.append(receiver)

            adat_pads.append(platform.request("toslink", i))

        #
        # signal path: USB ===> ADAT transmitters
        #
        audio_bits_end     = audio_bits
        channel_bits_start = audio_bits
        channel_bits_end   = channel_bits_start + number_of_channels_bits
        first_bit_pos      = channel_bits_end
        last_bit_pos       = first_bit_pos + 1

        m.d.comb += [
            # convert USB stream to audio stream
            usb_to_channel_stream.usb_stream_in.stream_eq(usb1_ep1_out.stream),
            *connect_stream_to_fifo(usb_to_channel_stream.channel_stream_out, usb_to_output_fifo),

            usb_to_output_fifo.w_data[channel_bits_start:channel_bits_end]
                .eq(usb_to_channel_stream.channel_stream_out.channel_nr),

            usb_to_output_fifo.w_data[first_bit_pos]
                .eq(usb_to_channel_stream.channel_stream_out.first),

            usb_to_output_fifo.w_data[last_bit_pos]
                .eq(usb_to_channel_stream.channel_stream_out.last),

            usb_to_output_fifo.r_en  .eq(bundle_demultiplexer.channel_stream_in.ready),
            usb_to_output_fifo_level .eq(usb_to_output_fifo.w_level),

            # demultiplex channel stream to the different transmitters
            bundle_demultiplexer.channel_stream_in.payload.eq(usb_to_output_fifo.r_data[0:audio_bits_end]),
            bundle_demultiplexer.channel_stream_in.channel_nr.eq(usb_to_output_fifo.r_data[channel_bits_start:channel_bits_end]),
            bundle_demultiplexer.channel_stream_in.last.eq(usb_to_output_fifo.r_data[-1]),
            bundle_demultiplexer.channel_stream_in.valid.eq(usb_to_output_fifo.r_rdy & usb_to_output_fifo.r_en),
            bundle_demultiplexer.no_channels_in.eq(no_channels),
        ]

        # wire up transmitters / receivers
        for i in range(4):
            m.d.comb += [
                # transmitters
                adat_transmitters[i].sample_in           .eq(bundle_demultiplexer.bundles_out[i].payload),
                adat_transmitters[i].addr_in             .eq(bundle_demultiplexer.bundles_out[i].channel_nr),
                adat_transmitters[i].last_in             .eq(bundle_demultiplexer.bundles_out[i].last),
                adat_transmitters[i].valid_in            .eq(bundle_demultiplexer.bundles_out[i].valid),
                bundle_demultiplexer.bundles_out[i].ready.eq(adat_transmitters[i].ready_out),
                adat_transmitters[i].user_data_in .eq(0),

                adat_pads[i].tx.eq(adat_transmitters[i].adat_out),

                # receivers
                adat_receivers[i].adat_in.eq(adat_pads[i].rx),

                # wire up receive FIFO to ADAT receiver
                bundle_multiplexer.no_channels_in[i]        .eq(8),
                bundle_multiplexer.bundles_in[i].payload    .eq(adat_receivers[i].sample_out),
                bundle_multiplexer.bundles_in[i].channel_nr .eq(adat_receivers[i].addr_out),
                bundle_multiplexer.bundles_in[i].valid      .eq(adat_receivers[i].output_enable),
                bundle_multiplexer.bundles_in[i].last       .eq(adat_receivers[i].addr_out == 7),
                bundle_multiplexer.bundle_active_in[i]      .eq(adat_receivers[i].synced_out),
            ]

        #
        # signal path: ADAT receivers ===> USB
        #
        m.submodules.input_to_usb_fifo = input_to_usb_fifo = \
            AsyncFIFOBuffered(width=audio_bits + number_of_channels_bits + 2, depth=16*8, w_domain="fast", r_domain="usb")

        chnr_start    = audio_bits
        chnr_end      = chnr_start + number_of_channels_bits
        channel_nr    = input_to_usb_fifo.r_data[chnr_start:chnr_end]
        first_channel = 0
        last_channel  = (number_of_channels - 1)

        m.d.comb += [
            # wire up receive FIFO to bundle multiplexer
            input_to_usb_fifo.w_data[0:chnr_start]        .eq(bundle_multiplexer.channel_stream_out.payload),
            input_to_usb_fifo.w_data[chnr_start:chnr_end] .eq(bundle_multiplexer.channel_stream_out.channel_nr),
            input_to_usb_fifo.w_en                        .eq(bundle_multiplexer.channel_stream_out.valid & input_to_usb_fifo.w_rdy),
            bundle_multiplexer.channel_stream_out.ready.eq(input_to_usb_fifo.w_rdy),

            # convert audio stream to USB stream
            channels_to_usb_stream.channel_stream_in.payload    .eq(input_to_usb_fifo.r_data[0:chnr_start]),
            channels_to_usb_stream.channel_stream_in.channel_nr .eq(channel_nr),
            channels_to_usb_stream.channel_stream_in.first      .eq(channel_nr == first_channel),
            channels_to_usb_stream.channel_stream_in.last       .eq(channel_nr == last_channel),
            channels_to_usb_stream.channel_stream_in.valid      .eq(input_to_usb_fifo.r_rdy),

            channels_to_usb_stream.data_requested_in .eq(usb1_ep2_in.data_requested),
            channels_to_usb_stream.frame_finished_in .eq(usb1_ep2_in.frame_finished),

            input_to_usb_fifo.r_en.eq(channels_to_usb_stream.channel_stream_in.ready),

            # wire up USB audio IN
            usb1_ep2_in.stream.stream_eq(channels_to_usb_stream.usb_stream_out),
        ]

        #
        # USB => output FIFO level debug signals
        #
        min_fifo_level = Signal.like(usb_to_output_fifo_level, reset=usb_to_output_fifo_depth)
        max_fifo_level = Signal.like(usb_to_output_fifo_level)

        with m.If(usb_to_output_fifo_level > max_fifo_level):
            m.d.sync += max_fifo_level.eq(usb_to_output_fifo_level)

        with m.If(usb_to_output_fifo_level < min_fifo_level):
            m.d.sync += min_fifo_level.eq(usb_to_output_fifo_level)

        # I2S DACs
        m.submodules.dac1_transmitter = dac1 = DomainRenamer("usb")(I2STransmitter(sample_width=audio_bits))
        m.submodules.dac2_transmitter = dac2 = DomainRenamer("usb")(I2STransmitter(sample_width=audio_bits))
        dac1_pads = platform.request("i2s", 1)
        dac2_pads = platform.request("i2s", 2)

        # Internal Logic Analyzer
        if self.USE_ILA:
            self.setup_ila(locals())

        usb_aux1 = platform.request("usb_aux", 1)
        usb_aux2 = platform.request("usb_aux", 2)

        # board status LEDs
        leds = platform.request("leds")
        m.d.comb += [
            leds.active1.eq(usb1.tx_activity_led | usb1.rx_activity_led),
            leds.suspended1.eq(usb1.suspended),
            leds.active2.eq(0),
            leds.suspended2.eq(0),
            leds.usb1.eq(usb_aux1.vbus),
            leds.usb2.eq(usb_aux2.vbus),
        ]
        m.d.comb += [getattr(leds, f"sync{i + 1}").eq(adat_receivers[i].synced_out) for i in range(4)]

        #
        # DEBUG display
        #
        adat1_underflow_count = Signal(16)

        with m.If(adat_transmitters[0].underflow_out):
            m.d.sync += adat1_underflow_count.eq(adat1_underflow_count + 1)
            m.d.sync += min_fifo_level.eq(0)

        with m.If(sof_counter == 0):
            m.d.sync += max_fifo_level.eq(0)

        spi = platform.request("spi")
        m.submodules.sevensegment = sevensegment = (NumberToSevenSegmentHex(width=32))
        m.submodules.led_display  = led_display  = (SerialLEDArray(divisor=10, init_delay=24e6))
        m.d.sync += [
            sevensegment.number_in[0:8].eq(adat1_underflow_count),
            sevensegment.number_in[8:16].eq(input_to_usb_fifo.w_level),
            sevensegment.number_in[16:24].eq(no_channels),
            sevensegment.number_in[24:32].eq(max_fifo_level),
            sevensegment.dots_in.eq(leds),
            Cat(led_display.digits_in).eq(sevensegment.seven_segment_out),
        ]
        m.d.comb += [
            *led_display.connect_to_resource(spi),
            led_display.valid_in.eq(1),
        ]

        return m

    def detect_active_audio_in(self, m, usb1, usb1_ep2_in):
        audio_in_seen   = Signal()
        audio_in_active = Signal()

        # detect if we don't have a USB audio IN packet
        with m.If(usb1.sof_detected):
            m.d.usb += [
                audio_in_active.eq(audio_in_seen),
                audio_in_seen.eq(0),
            ]

        with m.If(usb1_ep2_in.data_requested):
            m.d.usb += audio_in_seen.eq(1)

        return audio_in_active

    def calculate_usb_input_frame_size(self, m: Module, usb1_ep1_out, usb1_ep2_in, number_of_channels):
        """calculate the number of bytes one packet of audio input contains"""

        audio_in_frame_byte_counter   = Signal(range(self.MAX_PACKET_SIZE), reset=24 * number_of_channels)
        audio_in_frame_bytes_counting = Signal()

        with m.If(usb1_ep1_out.stream.valid & usb1_ep1_out.stream.ready):
            with m.If(audio_in_frame_bytes_counting):
                m.d.usb += audio_in_frame_byte_counter.eq(audio_in_frame_byte_counter + 1)

            with m.If(usb1_ep1_out.stream.first):
                m.d.usb += [
                    audio_in_frame_byte_counter.eq(1),
                    audio_in_frame_bytes_counting.eq(1),
                ]
            with m.Elif(usb1_ep1_out.stream.last):
                m.d.usb += audio_in_frame_bytes_counting.eq(0)

        audio_in_frame_bytes = Signal.like(audio_in_frame_byte_counter)
        with m.If(usb1_ep1_out.stream.last):
            m.d.usb += audio_in_frame_bytes.eq(audio_in_frame_byte_counter + 1)

        m.d.comb += usb1_ep2_in.bytes_in_frame.eq(audio_in_frame_bytes),

        return audio_in_frame_bytes

    def create_sample_rate_feedback_circuit(self, m: Module, usb1, usb1_ep1_in):
        #
        # USB rate feedback
        #

        # feedback endpoint
        feedbackValue      = Signal(32, reset=0x60000)
        bitPos             = Signal(5)

        # this tracks the number of ADAT frames in N microframes
        # with 12.288MHz / 8kHz = 1536 samples per microframe
        # we have N = 256, so we need
        # math.ceil(math.log2(1536 * 256)) = 19 bits
        adat_clock_counter      = Signal(19)

        # according to USB2 standard chapter 5.12.4.2
        # we need at least 2**13 / 2**8 = 2**5 = 32 SOF-frames of
        # sample master frequency counter to get the minimal
        # precision for the sample frequency estimate
        # / 2**8 because the ADAT-clock = 256 times = 2**8
        # the sample frequency
        # we average over 256 microframes, because that gives
        # us the maximum precision needed by the feedback endpoint
        sof_counter             = Signal(8)

        # since samples are constantly consumed from the FIFO
        # half the maximum USB packet size should be more than enough
        usb_to_output_fifo_depth = self.MAX_PACKET_SIZE // 2
        usb_to_output_fifo_level = Signal(range(usb_to_output_fifo_depth + 1))
        fifo_level_feedback      = Signal.like(usb_to_output_fifo_level)
        m.d.comb += fifo_level_feedback.eq(usb_to_output_fifo_level >> (usb_to_output_fifo_level.width - 7))

        adat_clock_usb = Signal()
        m.submodules.adat_clock_usb_sync = FFSynchronizer(ClockSignal("adat"), adat_clock_usb, o_domain="usb")
        m.submodules.adat_clock_usb_pulse = adat_clock_usb_pulse = DomainRenamer("usb")(EdgeToPulse())
        adat_clock_tick = Signal()
        m.d.usb += [
            adat_clock_usb_pulse.edge_in.eq(adat_clock_usb),
            adat_clock_tick.eq(adat_clock_usb_pulse.pulse_out),
        ]

        with m.If(adat_clock_tick):
            m.d.usb += adat_clock_counter.eq(adat_clock_counter + 1)

        with m.If(usb1.sof_detected):
            m.d.usb += sof_counter.eq(sof_counter + 1)

            with m.If(sof_counter == 0):
                # when feedbackValue == adat_clock_counter the
                # FIFO underflows slowly, but also when
                # feedbackValue == adat_clock_counter + 1
                # the FIFO slowly but surely fills to overflow.
                # since both of those feedback values are only one apart,
                # we need to start with the slowly overflowing value and
                # provide negative feedback proportional to the fill level
                # of the FIFO
                m.d.usb += [
                    feedbackValue.eq(adat_clock_counter - fifo_level_feedback),
                    adat_clock_counter.eq(0),
                ]

        m.d.comb += [
            usb1_ep1_in.bytes_in_frame.eq(4),
            bitPos.eq(usb1_ep1_in.address << 3),
            usb1_ep1_in.value.eq(0xff & (feedbackValue >> bitPos)),
        ]

        return (sof_counter, usb_to_output_fifo_level, usb_to_output_fifo_depth)

    def setup_ila(self, v):
        m                        = v['m']
        sof_counter              = v['sof_counter']
        usb1                     = v['usb1']
        usb1_ep1_out             = v['usb1_ep1_out']
        usb1_ep2_in              = v['usb1_ep2_in']
        audio_in_active          = v['audio_in_active']
        channels_to_usb_stream   = v['channels_to_usb_stream']
        usb_to_channel_stream    = v['usb_to_channel_stream']
        input_to_usb_fifo        = v['input_to_usb_fifo']
        usb_to_output_fifo       = v['usb_to_output_fifo']
        usb_to_output_fifo_level = v['usb_to_output_fifo_level']
        usb_to_output_fifo_depth = v['usb_to_output_fifo_depth']
        audio_in_frame_bytes     = v['audio_in_frame_bytes']
        min_fifo_level           = v['min_fifo_level']
        max_fifo_level           = v['max_fifo_level']
        adat_transmitters        = v['adat_transmitters']
        adat_receivers           = v['adat_receivers']
        bundle_demultiplexer     = v['bundle_demultiplexer']
        bundle_multiplexer       = v['bundle_multiplexer']

        adat_clock = Signal()
        m.d.comb += adat_clock.eq(ClockSignal("adat"))
        sof_wrap = Signal()
        m.d.comb += sof_wrap.eq(sof_counter == 0)

        usb_packet_counter = Signal(10)
        with m.If(usb1_ep1_out.stream.valid & usb1_ep1_out.stream.ready):
            m.d.usb += usb_packet_counter.eq(usb_packet_counter + 1)
            with m.If(usb1_ep1_out.stream.last):
                m.d.usb += usb_packet_counter.eq(0)

        weird_packet = Signal()
        m.d.comb += weird_packet.eq(usb1_ep1_out.stream.last & (
            usb_packet_counter[0:2] != Const(0b11, 2)
        ))

        strange_input = Signal()
        input_active  = Signal()
        output_active = Signal()
        input_or_output_active = Signal()

        m.d.comb += [
            input_active.eq (  channels_to_usb_stream.channel_stream_in.ready
                                & channels_to_usb_stream.channel_stream_in.valid),
            output_active.eq(  channels_to_usb_stream.usb_stream_out.ready
                                & channels_to_usb_stream.usb_stream_out.valid),
            input_or_output_active.eq(input_active | output_active),

            strange_input.eq(  (channels_to_usb_stream.channel_stream_in.payload != 0)
                                & (channels_to_usb_stream.channel_stream_in.channel_nr > 1)),
        ]

        channels_to_usb_input_frame = [
            usb1.sof_detected,
            #audio_in_active,
            input_to_usb_fifo.r_level,
            channels_to_usb_stream.channel_stream_in.channel_nr,
            channels_to_usb_stream.channel_stream_in.first,
            channels_to_usb_stream.channel_stream_in.last,
            input_active,
            #channels_to_usb_stream.channel_stream_in.payload,
        ]

        weird_frame_size = Signal()
        usb_outputting   = Signal()
        m.d.comb += usb_outputting.eq(usb1_ep1_out.stream.valid & usb1_ep1_out.stream.ready)

        usb_out_level_maxed = Signal()
        m.d.comb += usb_out_level_maxed.eq(usb_to_output_fifo_level >= (usb_to_output_fifo_depth - 1))

        m.d.comb += weird_frame_size.eq((audio_in_frame_bytes & 0b11) != 0)

        channels_to_usb_debug = [
            audio_in_frame_bytes,
            channels_to_usb_stream.current_channel,
            channels_to_usb_stream.feeder_state,
            channels_to_usb_stream.level,
            channels_to_usb_stream.fifo_full,
            channels_to_usb_stream.fifo_level_insufficient,
            channels_to_usb_stream.out_channel,
            channels_to_usb_stream.fifo_read,
            channels_to_usb_stream.usb_channel,
            channels_to_usb_stream.done,
            channels_to_usb_stream.usb_byte_pos,
            channels_to_usb_stream.skipping,
            channels_to_usb_stream.filling,
            usb1_ep2_in.data_requested,
            usb1_ep2_in.frame_finished,
        ]

        usb_out_debug = [
            usb_to_channel_stream.channel_stream_out.payload,
            usb_to_channel_stream.channel_stream_out.channel_nr,
            usb_to_channel_stream.channel_stream_out.first,
            usb_to_channel_stream.channel_stream_out.last,
            usb_to_output_fifo_level,
            usb_out_level_maxed
        ]

        usb_channel_outputting = Signal()
        m.d.comb += usb_channel_outputting.eq(
            usb_out_level_maxed |
            usb_to_channel_stream.channel_stream_out.first |
            usb_to_channel_stream.channel_stream_out.last  |
                ( usb_to_channel_stream.channel_stream_out.ready &
                    usb_to_channel_stream.channel_stream_out.valid)
            )

        ep1_out_fifo_debug = [
            audio_in_frame_bytes,
            min_fifo_level,
            usb_to_output_fifo_level,
            max_fifo_level,
            usb1.sof_detected,
        ]

        adat_nr = 0
        receiver_debug = [
            adat_receivers[adat_nr].sample_out,
            adat_receivers[adat_nr].addr_out,
            adat_receivers[adat_nr].output_enable,
        ]

        adat_first = Signal()
        m.d.comb += adat_first.eq(adat_receivers[adat_nr].output_enable & (adat_receivers[adat_nr].addr_out == 0))
        adat_clock = Signal()
        m.d.comb += adat_clock.eq(ClockSignal("adat"))

        adat_debug = [
            adat_clock,
            adat_transmitters[adat_nr].adat_out,
            adat_receivers[adat_nr].recovered_clock_out,
            adat_receivers[adat_nr].adat_in,
            adat_first,
            adat_receivers[adat_nr].output_enable,
        ]

        bundle0_active            = Signal()
        bundle3_active            = Signal()
        bundle_multiplexer_active = Signal()
        multiplexer_enable        = Signal()

        m.d.comb += [
            bundle0_active.eq((bundle_multiplexer.bundles_in[0].valid &
                               bundle_multiplexer.bundles_in[0].ready)),
            bundle3_active.eq((bundle_multiplexer.bundles_in[3].valid &
                               bundle_multiplexer.bundles_in[3].ready)),
            bundle_multiplexer_active.eq((bundle_multiplexer.channel_stream_out.valid &
                                          bundle_multiplexer.channel_stream_out.ready)),
            multiplexer_enable.eq(bundle0_active | bundle3_active | bundle_multiplexer_active),
        ]

        multiplexer_debug = [
            bundle_multiplexer.current_bundle,
            bundle_multiplexer.last_bundle,
            bundle0_active,
            #bundle_multiplexer.bundles_in[0].payload,
            bundle_multiplexer.bundles_in[0].channel_nr,
            bundle_multiplexer.bundles_in[0].last,
            bundle3_active,
            #bundle_multiplexer.bundles_in[3].payload,
            bundle_multiplexer.bundles_in[3].channel_nr,
            bundle_multiplexer.bundles_in[3].last,
            #bundle_multiplexer.channel_stream_out.payload,
            bundle_multiplexer_active,
            bundle_multiplexer.channel_stream_out.channel_nr,
            bundle_multiplexer.channel_stream_out.last,
            input_to_usb_fifo.w_level,
        ]

        demultiplexer_debug = [
            bundle_demultiplexer.channel_stream_in.ready,
            bundle_demultiplexer.channel_stream_in.valid,
            bundle_demultiplexer.channel_stream_in.channel_nr,
            #bundle_demultiplexer.channel_stream_in.payload,
            *[bundle_demultiplexer.bundles_out[i].ready for i in range(4)],
            *[bundle_demultiplexer.bundles_out[i].valid for i in range(4)],
            *[bundle_demultiplexer.bundles_out[i].channel_nr for i in range(4)],
        ]

        demultiplexer_enable = Signal()
        m.d.comb += demultiplexer_enable.eq(
            (bundle_demultiplexer.bundles_out[0].valid &
                bundle_demultiplexer.bundles_out[0].ready) |
            (bundle_demultiplexer.bundles_out[3].valid &
                bundle_demultiplexer.bundles_out[3].ready) |
            (bundle_demultiplexer.channel_stream_in.valid &
                bundle_demultiplexer.channel_stream_in.ready)
        )

        levels = [
            input_to_usb_fifo.r_level,
            channels_to_usb_stream.level,
        ]

        adat_receive_count   = Signal(8)
        adat_receive_frames  = Signal.like(adat_receive_count)
        adat_transmit_count  = Signal.like(adat_receive_count)
        adat_transmit_frames = Signal.like(adat_receive_count)

        with m.If(channels_to_usb_stream.channel_stream_in.last & channels_to_usb_stream.channel_stream_in.valid & channels_to_usb_stream.channel_stream_in.ready):
            m.d.usb += adat_receive_count.eq(adat_receive_count + 1)

        with m.If(usb_to_channel_stream.channel_stream_out.last & usb_to_channel_stream.channel_stream_out.valid & usb_to_channel_stream.channel_stream_out.ready):
            m.d.usb += adat_transmit_count.eq(adat_transmit_count + 1)

        with m.If(usb1.sof_detected):
            m.d.usb += [
                adat_receive_frames.eq(adat_receive_count),
                adat_receive_count.eq(0),
                adat_transmit_frames.eq(adat_transmit_count),
                adat_transmit_count.eq(0),
            ]

        adat_counts = [
            #adat_receive_count,
            #adat_transmit_count,
            adat_transmit_frames,
            adat_receive_frames,
        ]

        signals = adat_counts + [channels_to_usb_stream.skipping, channels_to_usb_stream.filling, usb1_ep2_in.frame_finished]
        #signals = channels_to_usb_input_frame + channels_to_usb_debug  #+ [channels_to_usb_stream.usb_stream_out.valid, channels_to_usb_stream.usb_stream_out.ready]

        signals_bits = sum([s.width for s in signals])
        m.submodules.ila = ila = \
            StreamILA(
                domain="usb", o_domain="usb",
                sample_rate=60e6, # usb domain
                #sample_rate=48e3 * 256 * 5, # sync domain
                #sample_rate=48e3 * 256 * 8, # fast domain
                signals=signals,
                sample_depth       = int(50 * 8 * 1024 / signals_bits),
                samples_pretrigger = 2, #int(98 * 8 * 1024 / signals_bits),
                with_enable=True)

        stream_ep = USBMultibyteStreamInEndpoint(
            endpoint_number=3, # EP 3 IN
            max_packet_size=self.ILA_MAX_PACKET_SIZE,
            byte_width=ila.bytes_per_sample
        )
        usb1.add_endpoint(stream_ep)

        garbage = Signal()

        m.d.comb += [
            stream_ep.stream.stream_eq(ila.stream),
            garbage.eq(channels_to_usb_stream.skipping | channels_to_usb_stream.filling),
            #ila.enable.eq(usb_outputting | weird_frame_size | usb1_ep1_out.stream.first | usb1_ep1_out.stream.last),
            #ila.enable.eq(usb_channel_outputting),
            #ila.enable.eq(input_or_output_active | garbage | usb1_ep2_in.data_requested | usb1_ep2_in.frame_finished),
            #ila.trigger.eq(audio_in_frame_bytes > 0xc0),
            #ila.enable.eq(bundle_multiplexer_active),
            ila.enable .eq(usb1.sof_detected | garbage | usb1_ep2_in.frame_finished),
            ila.trigger.eq(usb1.sof_detected) #adat_receive_frames > 0x7),
        ]

        ILACoreParameters(ila).pickle()

if __name__ == "__main__":
    os.environ["LUNA_PLATFORM"] = "qmtech_ep4ce_platform:ADATFacePlatform"
    #os.environ["LUNA_PLATFORM"] = "qmtech_10cl006_platform:ADATFacePlatform"
    top_level_cli(USB2AudioInterface)