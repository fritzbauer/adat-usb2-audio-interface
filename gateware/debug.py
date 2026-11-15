from amaranth import *
from amaranth.lib.cdc    import FFSynchronizer, PulseSynchronizer

from amlib.debug.ila     import StreamILA, ILACoreParameters
from amlib.stream        import StreamInterface
from amlib.io.led        import NumberToBitBar
from amlib.io.max7219    import SerialLEDArray

from luna.gateware.usb.usb2.endpoints.stream  import USBMultibyteStreamInEndpoint

def add_debug_led_array(v):
    self                      = v['self']
    m                         = v['m']
    platform                  = v['platform']
    channels_to_usb1_stream   = v['channels_to_usb1_stream']
    input_to_usb_fifo         = v['input_to_usb_fifo']
    usb1_to_output_fifo_level = v['usb1_to_output_fifo_level']
    usb1_to_output_fifo_depth = v['usb1_to_output_fifo_depth']
    usb2_to_usb1_fifo_level   = v['usb2_to_usb1_fifo_level']
    usb2_to_usb1_fifo_depth   = v['usb2_to_usb1_fifo_depth']
    channels_to_usb2_stream   = v['channels_to_usb2_stream']
    usb2_audio_out_active     = v['usb2_audio_out_active']
    usb2_audio_in_active      = v['usb2_audio_in_active']
    bundle_multiplexer        = v['bundle_multiplexer']
    adat_transmitters         = v['adat_transmitters']
    usb1_to_usb2_midi_fifo    = v['usb1_to_usb2_midi_fifo']
    usb2_to_usb1_midi_fifo    = v['usb2_to_usb1_midi_fifo']
    usb_midi_fifo_depth       = v['usb_midi_fifo_depth']


    adat1_underflow_count = Signal(16)
    with m.If(adat_transmitters[0].underflow_out):
        m.d.sync += adat1_underflow_count.eq(adat1_underflow_count + 1)

    spi = platform.request("spi")
    m.submodules.led_display  = led_display = SerialLEDArray(divisor=10, init_delay=24e6, no_modules=4)

    rx_level_bars = []
    for i in range(1, 5):
        rx_level_bar = NumberToBitBar(0, bundle_multiplexer.FIFO_DEPTH, 8)
        setattr(m.submodules, f"rx{i}_level_bar", rx_level_bar)
        m.d.comb += rx_level_bar.value_in.eq(bundle_multiplexer.levels[i - 1])
        rx_level_bars.append(rx_level_bar)

    m.submodules.in_bar       = in_to_usb_fifo_bar  = NumberToBitBar(0, self.INPUT_CDC_FIFO_DEPTH, 32)
    m.submodules.in_fifo_bar  = channels_to_usb_bar = NumberToBitBar(0, 2 * self.USB1_MAX_PACKET_SIZE, 32)
    m.submodules.out_fifo_bar = out_fifo_bar        = NumberToBitBar(0, usb1_to_output_fifo_depth, 32)

    m.d.comb += [
        # LED bar displays
        in_to_usb_fifo_bar.value_in.eq(input_to_usb_fifo.r_level),
        channels_to_usb_bar.value_in.eq(channels_to_usb1_stream.level >> 3),
        out_fifo_bar.value_in.eq(usb1_to_output_fifo_level >> 1),

        *[led_display.digits_in[i].eq(Cat(reversed(rx_level_bars[i].bitbar_out))) for i in range(4)],
        led_display.digits_in[4].eq(Cat(reversed(in_to_usb_fifo_bar.bitbar_out))),
        led_display.digits_in[5].eq(Cat(reversed(channels_to_usb_bar.bitbar_out))),
        led_display.digits_in[6].eq(Cat(reversed(out_fifo_bar.bitbar_out))),
        led_display.digits_in[7].eq(adat1_underflow_count),
    ]

    usb2 = lambda x: 8 + x

    m.submodules.usb2_output_fifo_bar = usb2_output_fifo_bar = NumberToBitBar(0, usb2_to_usb1_fifo_depth, 8)
    m.submodules.usb2_input_fifo_bar  = usb2_input_fifo_bar  = NumberToBitBar(0, channels_to_usb2_stream._fifo_depth, 8)
    m.submodules.usb2_to_usb1_bar     = usb2_to_usb1_bar     = NumberToBitBar(0, usb_midi_fifo_depth, 8)
    m.submodules.usb1_to_usb2_bar     = usb1_to_usb2_bar     = NumberToBitBar(0, usb_midi_fifo_depth, 8)

    m.d.comb += [
        usb2_output_fifo_bar.value_in.eq(usb2_to_usb1_fifo_level),
        usb2_input_fifo_bar.value_in.eq(channels_to_usb2_stream.level),
        led_display.digits_in[usb2(0)][0].eq(usb2_audio_out_active),
        led_display.digits_in[usb2(0)][7].eq(usb2_audio_in_active),
        led_display.digits_in[usb2(1)].eq(Cat(usb2_output_fifo_bar.bitbar_out)),
        led_display.digits_in[usb2(2)].eq(Cat(reversed(usb2_input_fifo_bar.bitbar_out))),
        led_display.digits_in[usb2(3)].eq(usb2_to_usb1_bar.bitbar_out),
        led_display.digits_in[usb2(4)].eq(usb1_to_usb2_bar.bitbar_out),
    ]

    m.d.comb += [
        *led_display.connect_to_resource(spi),
        led_display.valid_in.eq(1),
    ]



def setup_ila(v, ila_max_packet_size):
    """ILA in the fast domain (98.3 MHz) observing ADAT receiver 1
    (port 2 = the optical loopback) at full resolution, 2026-11-15.

    Captured: raw nrzi level, raw/qualified edges, the edge gap timer,
    the measured edge interval, the decoder FSM state, running, the
    decoded bit and its strobe. The StreamILA's CDC to the usb domain
    has backpressure (AsyncFIFOBuffered), so no samples are lost; the
    readout just takes longer than the capture.

    Trigger: the first qualified edge after reset (any edge starts the
    capture; 500 records of pretrigger).
    """
    m              = v['m']
    usb1           = v['usb1']
    adat_receivers = v['adat_receivers']

    rx = adat_receivers[1]

    s_nrzi    = Signal(name="nrzi")
    s_raw     = Signal(name="raw_edge")
    s_qual    = Signal(name="qual_edge")
    s_timer   = Signal(8, name="gap_timer")
    s_iv      = Signal(8, name="interval")
    s_sync    = Signal(name="dec_sync")
    s_run     = Signal(name="running")
    s_data    = Signal(name="data_out")
    s_dataen  = Signal(name="data_out_en")

    m.d.comb += [
        s_nrzi.eq(rx.dbg_nrzi),
        s_raw.eq(rx.dbg_raw_edge),
        s_qual.eq(rx.dbg_qualified_edge),
        s_timer.eq(rx.dbg_gap_timer),
        s_iv.eq(rx.dbg_interval),
        s_sync.eq(rx.dbg_sync_state),
        s_run.eq(rx.synced_out),
        s_data.eq(rx.sample_out[0]),   # 1 bit of the decoded sample as marker
        s_dataen.eq(rx.dbg_data_out_en),
    ]

    signals = [s_nrzi, s_raw, s_qual, s_timer, s_iv, s_sync, s_run, s_data, s_dataen]
    # pad the sample to 32 bits: the ILA capture memory showed field
    # corruption at odd widths (23, 50) -- M9K packing splits the memory
    # and the fields come back skewed. A clean 32-bit width avoids that.
    pad_total = 32 - sum([len(s) for s in signals])
    for k in range(pad_total):
        z = Signal(name=f"pad{k}")
        m.d.comb += z.eq(0)
        signals.append(z)
    signals_bits = sum([len(s) for s in signals])
    print(f"ILA: {len(signals)} signals, {signals_bits} bits/sample, "
          f"{ila_max_packet_size} max packet")
    m.submodules.ila = ila = \
        StreamILA(
            domain="fast", o_domain="usb",
            sample_rate=98304000,
            signals=signals,
            sample_depth       = int(50 * 8 * 1024 / signals_bits),
            samples_pretrigger = 256,
            with_enable=True)

    stream_ep = USBMultibyteStreamInEndpoint(
        endpoint_number=4,
        max_packet_size=ila_max_packet_size,
        byte_width=ila.bytes_per_sample
    )
    usb1.add_endpoint(stream_ep)

    m.d.comb += [
        stream_ep.stream.stream_eq(ila.stream),
        ila.trigger.eq(s_qual),
        ila.enable.eq(1),
    ]

    ILACoreParameters(ila).pickle()
