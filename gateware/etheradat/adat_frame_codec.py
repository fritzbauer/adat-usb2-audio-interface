"""Canonical 32-byte ADAT frame <-> audio samples conversion.

The AdatNet protocol transports the raw 256-bit ADAT frame as 32 bytes
(EtherADAT PROTOCOL.md §5, layout bit-exact to the PC-side reference in
etheradat/reference.py, which is ported from the verified
virtual_soundcard/test_veth/rp2350_sim.py):

    bits 0..9    sync, all '0'
    bit  10      '1' (user block start)
    bits 11..14  user bits (bit 11 = first on the wire)
    bit  15      '1' (user block end)
    channel c (0..7), nibble group n (0..5):
      bit 16+30c+5n       '1' (marker)
      bits +1..+4         sample bits 23-4n .. 20-4n (MSB first)

This layout is also bit-identical to the order the ADATTransmitter puts
on the wire (marker-then-data, MSB-first per nibble group), so frames
captured on the ADAT wire and frames built here agree byte-for-byte.
"""

from amaranth import Elaboratable, Module, Signal, Array, Memory, Cat

from etheradat.reference import EMPTY_USER_FRAME

# frame bit of sample bit (channel c, sample bit b): marker at
# 16+30c+5n, data at +1..+4 with n = (23-b)>>2, j = (23-b)&3.
def _sample_frame_bit(c, b):
    n = (23 - b) >> 2
    j = (23 - b) & 3
    return 16 + 30 * c + 5 * n + 1 + j


SAMPLE_BIT_MAP = [(c, b, _sample_frame_bit(c, b))
                  for c in range(8) for b in range(24)]


def _frame_sig_bit(frame_bit):
    """Map a canonical frame bit index to the bit index of the 256-bit
    signal convention used here: the signal is assembled from 32-bit
    words that are MSB-first packed (word w, bit 31 = frame bit 32w,
    matching int.from_bytes(frame[4w:4w+4], "big"))."""
    return 32 * (frame_bit >> 5) + 31 - (frame_bit & 31)


def connect_frame_to_samples(m, frame, samples, user):
    """Combinational deserialize: 256-bit frame signal -> 8 x 24-bit
    samples + user bits.

    samples must be a list of eight 24-bit signals. The sync/user marker
    bits are not validated here (the Ethernet path validated the frame
    header; frames from the ADAT wire have been validated by the
    ADATReceiver).
    """
    for c in range(8):
        value = 0
        for b in range(24):
            value = value | (frame[_frame_sig_bit(_sample_frame_bit(c, b))]
                             << b)
        m.d.comb += samples[c].eq(value)
    # user value convention of the PC-side adat-codec crate: value bit k
    # at frame bit 14-k (frame[1] >> 1 & 0xF on the byte level)
    m.d.comb += user.eq(
        frame[_frame_sig_bit(14)] | frame[_frame_sig_bit(13)] << 1 |
        frame[_frame_sig_bit(12)] << 2 | frame[_frame_sig_bit(11)] << 3)


def build_frame_from_registers(sample_regs, user_reg):
    """Return the list of 256 terms (Signals/Constants, LSB first) whose
    ``Cat(*terms)`` evaluates to the canonical frame over the given
    registers (list of eight 24-bit signals + 4-bit user signal), in
    the 32-bit-word MSB-first signal convention (see _frame_sig_bit)."""
    terms = [0] * 256
    for k in range(10):
        terms[_frame_sig_bit(k)] = 0        # sync zeros
    terms[_frame_sig_bit(10)] = 1
    for k in range(4):
        # user bits: value bit k at frame bit 14-k (adat-codec convention)
        terms[_frame_sig_bit(14 - k)] = user_reg[k]
    terms[_frame_sig_bit(15)] = 1
    for c in range(8):
        for n in range(6):
            # channel nibble-group markers ('1' before each nibble)
            terms[_frame_sig_bit(16 + 30 * c + 5 * n)] = 1
    for c, b, p in SAMPLE_BIT_MAP:
        terms[_frame_sig_bit(p)] = sample_regs[c][b]
    return terms


class InputFrameCollector(Elaboratable):
    """Collects the channel samples of one ADAT receiver into frames and
    writes the canonical 32-byte frame into a 15-slot ring buffer.

    One instance per ADAT input, in the receiver's clock domain (fast).
    On the last channel (addr 7) of a frame in which all eight channels
    were seen, the samples + user bits are latched into a shadow register
    and written into the ring as 8 x 32-bit words. While the block
    builder reads the ring it asserts ``pause_in`` and ring writes pause
    (a frame arriving during the pause replaces the shadow — the oldest
    samples repeat, never a torn frame; the pause is bounded by the copy
    time, far below one ADAT frame period). Ring write cycles stretch
    across receiver channel pulses the same way.

    Attributes
    ----------
    sample_in, addr_in, enable_in, user_in : ADATReceiver sample outputs
    synced_in             : ADATReceiver.synced_out
    pause_in              : block builder is reading our ring
    ring                  : Memory(32, 15*8), read by the block builder
    slot_pos_out          : ring slot being written next (0..14)
    slot_wrapped_out      : pulse when slot 14 was written (block pace)
    dbg_frame_done        : pulse: a complete frame was latched
    dbg_incomplete        : pulse: a frame ended with channels missing
    """

    RING_SLOTS = 15
    WORDS_PER_SLOT = 8

    def __init__(self, domain="sync"):
        self.domain = domain
        self.sample_in = Signal(24)
        self.addr_in = Signal(3)
        self.enable_in = Signal()
        self.user_in = Signal(4)
        self.synced_in = Signal()
        self.pause_in = Signal()

        # rings start filled with the silent EMPTY_USER_FRAME: an input
        # that has never seen a frame transmits silence (PROTOCOL.md §5.1)
        empty_words = [int.from_bytes(EMPTY_USER_FRAME[4*w:4*w+4], "big")
                       for w in range(8)] * self.RING_SLOTS
        self.ring = Memory(width=32, depth=self.RING_SLOTS * self.WORDS_PER_SLOT,
                           init=empty_words, name="adat_frame_ring")
        self.slot_pos_out = Signal(4)
        self.slot_wrapped_out = Signal()

        self.dbg_frame_done = Signal()
        self.dbg_incomplete = Signal()

    def elaborate(self, platform):
        m = Module()
        sync = m.d[self.domain]
        comb = m.d.comb

        # shadow: the complete frame currently being written to the ring
        shadow = [Signal(24, name=f"shadow{i}") for i in range(8)]
        shadow_user = Signal(4)
        # live: channels arriving from the receiver
        live = [Signal(24, name=f"live{i}") for i in range(8)]
        seen = Signal(8)

        write_pos = Signal(4)
        word_cnt = Signal(4)
        writing = Signal()

        write_port = self.ring.write_port(domain=self.domain)
        m.submodules += write_port

        terms = build_frame_from_registers(shadow, shadow_user)
        frame = Signal(256)
        comb += frame.eq(Cat(*terms))

        comb += [
            self.slot_pos_out.eq(write_pos),
            write_port.addr.eq(write_pos * self.WORDS_PER_SLOT + word_cnt),
            write_port.data.eq(frame.word_select(word_cnt, 32)),
            write_port.en.eq(writing & ~self.pause_in & ~self.enable_in),
        ]

        # default pulses (later conditional assignments override)
        sync += [
            self.slot_wrapped_out.eq(0),
            self.dbg_frame_done.eq(0),
            self.dbg_incomplete.eq(0),
        ]

        # ring write state machine: advances only on free cycles
        with m.If(writing & ~self.pause_in & ~self.enable_in):
            with m.If(word_cnt == self.WORDS_PER_SLOT - 1):
                sync += writing.eq(0)
                with m.If(write_pos == self.RING_SLOTS - 1):
                    # wrap to slot 0 directly — write_pos is 4 bits and
                    # a plain +1 would land on phantom slot 15, whose
                    # write addresses (120..127) fall outside the ring
                    # (harmlessly ignored in simulation, but real
                    # memory would wrap them onto slot 0)
                    sync += [
                        write_pos.eq(0),
                        self.slot_wrapped_out.eq(1),
                    ]
                with m.Else():
                    sync += write_pos.eq(write_pos + 1)
            with m.Else():
                sync += word_cnt.eq(word_cnt + 1)

        # collect channels from the receiver
        with m.If(self.enable_in & self.synced_in):
            sync += [
                Array(live)[self.addr_in].eq(self.sample_in),
                seen.eq(seen | (1 << self.addr_in)),
            ]
            with m.If(self.addr_in == 7):
                sync += self.dbg_frame_done.eq(1)
                # `seen` does not yet hold this edge's ch7 bit: the
                # frame is complete when channels 0..6 were all seen
                with m.If((seen & 0x7F) == 0x7F):
                    # frame complete: latch into shadow, start ring write.
                    # If a write is still in flight (pause), the shadow is
                    # replaced: the older frame repeats in the ring.
                    # ch7 arrives on THIS edge: live[7] still holds the
                    # previous frame's ch7 here, so take it from
                    # sample_in directly (the channels 0..6 latched into
                    # live[] on their earlier edges are current)
                    for c in range(7):
                        sync += shadow[c].eq(live[c])
                    sync += [
                        shadow[7].eq(self.sample_in),
                        shadow_user.eq(self.user_in),
                        writing.eq(1),
                        word_cnt.eq(0),
                        seen.eq(0),
                    ]
                with m.Else():
                    sync += [
                        seen.eq(0),
                        self.dbg_incomplete.eq(1),
                    ]

        return m


class FrameDeserializer(Elaboratable):
    """Reads one 32-byte slot (8 x 32-bit words) from a memory and
    combinationally exposes 8 x 24-bit samples + user bits.

    The owner asserts ``read_strobe_in`` (level) to start one read
    burst; the burst issues/captures 8 words over 16 cycles and
    ``samples_valid_out`` then pulses for one cycle with the decoded
    samples stable on the outputs (until the next burst).
    """

    def __init__(self, memory, domain="sync"):
        self.domain = domain
        self.mem = memory
        self.read_base_in = Signal(12)     # first word address of the slot
        self.read_strobe_in = Signal()
        self.sample_out = [Signal(24, name=f"deser{i}") for i in range(8)]
        self.user_out = Signal(4)
        self.samples_valid_out = Signal()
        # test observation point: the assembled 256-bit frame
        self.dbg_frame_out = Signal(256)

    def elaborate(self, platform):
        m = Module()
        sync = m.d[self.domain]
        comb = m.d.comb

        read_port = self.mem.read_port(domain=self.domain)
        m.submodules += read_port

        word_cnt = Signal(3)
        burst = Signal()
        phase = Signal()                   # 0: issue read, 1: capture
        frame = Signal(256)

        comb += [
            read_port.addr.eq(self.read_base_in + word_cnt),
            read_port.en.eq(burst & ~phase),
            self.dbg_frame_out.eq(frame),
        ]

        connect_frame_to_samples(m, frame, self.sample_out, self.user_out)

        sync += self.samples_valid_out.eq(0)
        with m.If(burst):
            with m.If(~phase):
                sync += phase.eq(1)
            with m.Else():
                sync += [
                    phase.eq(0),
                    frame.word_select(word_cnt, 32).eq(read_port.data),
                ]
                with m.If(word_cnt == 7):
                    sync += [
                        burst.eq(0),
                        word_cnt.eq(0),
                        self.samples_valid_out.eq(1),
                    ]
                with m.Else():
                    sync += word_cnt.eq(word_cnt + 1)
        with m.Else():
            with m.If(self.read_strobe_in):
                sync += burst.eq(1)

        return m
