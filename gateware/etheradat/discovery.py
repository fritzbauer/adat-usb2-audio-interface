"""Hello/discovery (EtherADAT PROTOCOL.md §12) — RMII clock domain.

Implements the device-side (role 1) half of the discovery protocol,
mirroring rp2350_sim.py and the PC kernel module's expectations:

- broadcast hello (dst ff:ff:ff:ff:ff:ff) at 4 Hz while no PC frame has
  been seen, 1 Hz unicast (to the learned PC MAC) afterwards;
- learn the PC MAC from the source address of any valid PC frame
  (hello with role 2, or an audio frame);
- keepalive: > 1 s without any PC frame -> forget the PC MAC, back to
  the 4 Hz broadcast rate;
- never send a hello as a reaction to a PC hello (no ping-pong).

The hello frame is padded to the Ethernet minimum by the MAC core
(60 bytes payload + FCS); this module emits the unpadded 34 bytes
(14 MAC/EtherType + 20 hello payload) on a byte stream.
"""

from amaranth import Elaboratable, Module, Signal, Mux

from etheradat.reference import (
    ETHERTYPE_HELLO, HELLO_LEN, HELLO_ROLE_DEVICE, N_SAMPLES)

HELLO_FRAME_BYTES = 14 + HELLO_LEN          # 34, padding added by MAC core

HELLO_PERIOD_DISCOVER_DEFAULT = 0.25        # s (rp2350_sim.py)
HELLO_PERIOD_LINKED_DEFAULT = 1.0           # s (§12.2)
KEEPALIVE_DEFAULT = 1.0                     # s (§12.2)


class Discovery(Elaboratable):
    """Device-side discovery state machine + hello frame source.

    Parameters
    ----------
    own_mac            : 48-bit integer, locally administered
    device_serial      : 32-bit integer for the hello payload
    session_nonce      : 32-bit integer, changes per boot (the PC resets
                         its tracker on change)
    rmii_clk_freq      : for the default timer periods
    hello_discover_cycles / hello_linked_cycles / keepalive_cycles :
                         cycle counts, defaulting to 0.25 s / 1 s / 1 s;
                         tests pass tiny values.

    Attributes (MAC-frame byte stream, hand to the TX arbiter)
    ----------------------------------------------------------
    hello_data/valid/ready/last : byte stream of the 34-byte hello frame
    hello_granted     : level while this module owns the TX path

    Attributes (RX side, driven by the frame parser)
    ------------------------------------------------
    rx_frame_strobe   : pulse on the last byte of a valid frame
    rx_is_hello_in    : frame is a hello (EtherType 0x88B6)
    rx_is_audio_in    : frame is an audio frame (EtherType 0x88B5)
    rx_role_in        : hello role field (valid with rx_frame_strobe +
                        rx_is_hello_in and hello_valid_in)
    rx_hello_valid_in : hello payload passed the parser's field checks
    rx_src_mac_in     : source MAC of the received frame

    Attributes (status)
    -------------------
    pc_mac_out        : learned PC MAC (48-bit int, big-endian bytes:
                        pc_mac_out[47:40] is the first byte on the wire)
    linked_out        : level: PC MAC known & keepalive not expired
    audio_active_in   : informational hello flag (any ADAT input synced)
    dbg_hello_due     : hello timer expired this cycle
    """

    def __init__(self, *, own_mac, device_serial=0, session_nonce=0,
                 rmii_clk_freq=50_000_000,
                 hello_discover_cycles=None, hello_linked_cycles=None,
                 keepalive_cycles=None):
        self.own_mac = own_mac & 0xFFFFFFFFFFFF
        self.device_serial = device_serial & 0xFFFFFFFF
        self.session_nonce = session_nonce & 0xFFFFFFFF
        self.rmii_clk_freq = rmii_clk_freq
        self.hello_discover_cycles = (
            hello_discover_cycles or int(rmii_clk_freq
                                         * HELLO_PERIOD_DISCOVER_DEFAULT))
        self.hello_linked_cycles = (
            hello_linked_cycles or int(rmii_clk_freq
                                       * HELLO_PERIOD_LINKED_DEFAULT))
        self.keepalive_cycles = (
            keepalive_cycles or int(rmii_clk_freq * KEEPALIVE_DEFAULT))

        self.hello_data = Signal(8)
        self.hello_valid = Signal()
        self.hello_ready = Signal()
        self.hello_last = Signal()
        self.hello_granted = Signal()

        self.rx_frame_strobe = Signal()
        self.rx_is_hello_in = Signal()
        self.rx_is_audio_in = Signal()
        self.rx_role_in = Signal(8)
        self.rx_hello_valid_in = Signal()
        self.rx_src_mac_in = Signal(48)

        self.audio_active_in = Signal()
        self.pc_mac_out = Signal(48)
        self.linked_out = Signal()
        self.dbg_hello_due = Signal()

        self.dbg_hello_tx = Signal()
        self.dbg_hello_rx = Signal()
        self.dbg_keepalive_timeout = Signal()

    def _frame_byte(self, index):
        """Comb term for hello frame byte `index` (0..33)."""
        own = self.own_mac
        mac_bc = 0xFFFFFFFFFFFF
        dst = Mux(self.linked_out, self.pc_mac_out, mac_bc)
        table = {
            0:  dst[40:48], 1: dst[32:40], 2: dst[24:32],
            3:  dst[16:24], 4: dst[8:16],  5: dst[0:8],
            6:  (own >> 40) & 0xFF, 7: (own >> 32) & 0xFF,
            8:  (own >> 24) & 0xFF, 9: (own >> 16) & 0xFF,
            10: (own >> 8) & 0xFF, 11: own & 0xFF,
            12: ETHERTYPE_HELLO >> 8, 13: ETHERTYPE_HELLO & 0xFF,
            14: 0x41, 15: 0x44, 16: 0x41, 17: 0x54,     # "ADAT"
            18: HELLO_ROLE_DEVICE,
        }
        if index in table:
            return table[index]
        if index == 19:
            return self.audio_active_in                 # flags bit 0
        if index == 20:
            return 1                                    # proto_ver LSB
        if index == 21:
            return 0                                    # proto_ver MSB
        if 22 <= index <= 25:
            return (self.device_serial >> (8 * (index - 22))) & 0xFF
        if 26 <= index <= 29:
            return (self.session_nonce >> (8 * (index - 26))) & 0xFF
        if index == 30:
            return N_SAMPLES & 0xFF
        if index == 31:
            return N_SAMPLES >> 8
        return 0                                        # reserved + pad

    def elaborate(self, platform):
        m = Module()
        sync = m.d.sync
        comb = m.d.comb

        pc_mac = Signal(48, reset=0)
        linked = Signal(reset=0)
        comb += [
            self.pc_mac_out.eq(pc_mac),
            self.linked_out.eq(linked),
        ]

        # default strobes (conditional assignments below override)
        sync += [
            self.dbg_hello_due.eq(0),
            self.dbg_hello_tx.eq(0),
            self.dbg_hello_rx.eq(0),
            self.dbg_keepalive_timeout.eq(0),
        ]

        # keepalive timer
        keepalive_width = max(2, self.keepalive_cycles.bit_length())
        keepalive = Signal(keepalive_width)
        with m.If(~linked):
            sync += keepalive.eq(0)
        with m.Elif(self.rx_frame_strobe):
            sync += keepalive.eq(0)
        with m.Else():
            with m.If(keepalive == self.keepalive_cycles - 1):
                sync += [
                    linked.eq(0),
                    self.dbg_keepalive_timeout.eq(1),
                ]
            with m.Else():
                sync += keepalive.eq(keepalive + 1)

        # MAC learning: any valid PC frame teaches us its source MAC
        with m.If(self.rx_frame_strobe):
            with m.If(self.rx_is_hello_in & self.rx_hello_valid_in
                      & (self.rx_role_in == 2)):
                sync += [
                    pc_mac.eq(self.rx_src_mac_in),
                    linked.eq(1),
                ]
                sync += self.dbg_hello_rx.eq(1)
            with m.Elif(self.rx_is_audio_in):
                # audio frames are only accepted from the PC once the
                # header checked out; the parser gates the strobe
                sync += [
                    pc_mac.eq(self.rx_src_mac_in),
                    linked.eq(1),
                ]

        # hello TX timer: 4 Hz unlinked, 1 Hz linked; never as a reaction
        hello_timer = Signal(max(self.hello_discover_cycles,
                                 self.hello_linked_cycles).bit_length())
        sync += self.dbg_hello_due.eq(0)
        with m.If(self.hello_granted):
            sync += hello_timer.eq(0)
        with m.Else():
            with m.If(hello_timer >= Mux(linked, self.hello_linked_cycles,
                                         self.hello_discover_cycles) - 1):
                sync += [
                    hello_timer.eq(0),
                    self.dbg_hello_due.eq(1),
                ]
            with m.Else():
                sync += hello_timer.eq(hello_timer + 1)

        # hello frame: 34 byte registers, rebuilt when inputs change
        frame = Signal(HELLO_FRAME_BYTES * 8)
        frame_fresh = Signal()
        with m.If(frame_fresh | self.dbg_hello_due):
            for i in range(HELLO_FRAME_BYTES):
                sync += frame[i * 8:(i + 1) * 8].eq(self._frame_byte(i))

        # serialize on demand; dbg_hello_due only triggers the rebuild,
        # the (rebuilt) frame becomes visible one cycle later
        byte_idx = Signal(6)
        comb += [
            self.hello_valid.eq(frame_fresh),
            self.hello_data.eq(frame.word_select(byte_idx, 8)),
            self.hello_last.eq(byte_idx == HELLO_FRAME_BYTES - 1),
        ]
        with m.If(self.hello_valid & self.hello_ready):
            sync += [
                frame_fresh.eq(~self.hello_last),
                byte_idx.eq(byte_idx + 1),
                self.dbg_hello_tx.eq(1),
            ]
        with m.If(self.dbg_hello_due):
            sync += [
                frame_fresh.eq(1),
                byte_idx.eq(0),
            ]

        return m
