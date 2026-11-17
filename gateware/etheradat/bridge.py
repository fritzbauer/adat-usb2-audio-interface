"""EtherADAT bridge core: AdatNet role-1 protocol over a LiteEth MAC
core's user streams.

Clock domains (real names, must exist; see RMIIClockDomain):
- "fast": ADAT receivers + InputFrameCollectors + AudioBlockBuilder
- "sync": ADATTransmitter sample inputs + AudioFeeder
- "rmii": this module's protocol FSMs, the LiteEth MAC core user
  streams, discovery and MDIO

The MAC boundary is the LiteEth MAC core user-stream contract
(generated Verilog module `liteeth_rmii_mac`, see liteeth/):
- TX sink: frame bytes WITHOUT preamble/SFD and WITHOUT FCS, `last` on
  the final payload byte; the core pads short frames to the Ethernet
  minimum, appends the CRC-32 FCS, prepends the preamble and enforces
  the inter-frame gap.
- RX source: frames WITHOUT preamble/SFD and WITHOUT FCS, `last` on
  the final payload byte, `error` set on the last byte of frames that
  failed a check.

Audio payloads (PROTOCOL.md §4, mode A):
- TX direction frames: 8-byte header + 15 samples x 3 inputs x 32-byte
  ADAT frames, slot (s*3+i), payload_len = 1448.
- RX direction frames: 8-byte header + 15 samples x 2 outputs x 32
  bytes, slot (s*2+o), payload_len = 968.
"""

from amaranth import (Elaboratable, Module, Signal, Array, Mux, Memory, Cat,
                      ClockDomain, ClockSignal, ResetSignal)
from amaranth.lib.cdc import FFSynchronizer, ResetSynchronizer

from etheradat.audio_payload import (AudioBlockBuilder, AudioFeeder,
                                     TX_BUFFER_BYTES, BLOCK_WORDS_RX,
                                     gray2bin2_terms)
from etheradat.discovery import Discovery, HELLO_LEN
from etheradat.mdio import MDIOMaster
from etheradat.reference import (
    ETHERTYPE_AUDIO, ETHERTYPE_HELLO, MAGIC, N_SAMPLES, PAYLOAD_LEN_TX,
    PAYLOAD_LEN_RX, VER_FLAGS_TX)

TX_FRAME_BYTES = 14 + PAYLOAD_LEN_TX        # 1462 (FCS added by the core)
TX_HDR_BYTES = 22                            # MAC + EtherType + payload hdr
RX_HDR_END = 14 + 8                          # MAC/Type + payload header
RX_AUDIO_BYTES = 15 * 2 * 32
BROADCAST_MAC = 0xFFFFFFFFFFFF

# RX parser states
RX_HEADER = 0            # bytes 0..13: dst, src, EtherType
RX_AUDIO_HDR = 1         # payload header bytes 0..7
RX_AUDIO_PL = 2          # audio payload bytes
RX_HELLO_PL = 3          # hello payload bytes 0..19
RX_TAIL = 4              # rest of the frame (padding or filtered bytes)

# tail kinds (what to do at the last byte of a TAIL-state frame)
TAIL_FILTERED = 0        # filtered frame: nothing
TAIL_HELLO = 1           # hello parsed: apply verdict at frame end
TAIL_AUDIO = 2           # padded audio frame: audio learn strobe


class RMIIClockDomain(Elaboratable):
    """Declares the "rmii" clock domain from an external clock net.

    The domain is held in reset for ``por_cycles`` after configuration
    (a power-on-reset counter in the rmii domain itself — the PLLs and
    the PHY need a moment to come up) and by the ``rst_in`` input.
    ``reset_out`` reflects the unsynchronized reset (for PHY nRST use
    the synchronized ResetSignal("rmii")).
    """

    def __init__(self, por_cycles=1 << 20):
        self.clk_in = Signal()
        self.rst_in = Signal()
        self.por_cycles = por_cycles
        self.reset_out = Signal()

    def elaborate(self, platform):
        m = Module()
        m.domains.rmii = ClockDomain()
        m.d.comb += ClockSignal("rmii").eq(self.clk_in)

        por_cnt = Signal(max(1, self.por_cycles.bit_length()))
        por = por_cnt < self.por_cycles
        with m.If(por):
            m.d.rmii += por_cnt.eq(por_cnt + 1)

        m.d.comb += self.reset_out.eq(por | self.rst_in)
        m.submodules.reset = ResetSynchronizer(self.reset_out,
                                               domain="rmii")
        return m


class EtherADATBridge(Elaboratable):
    """AdatNet role-1 bridge: 3 ADAT inputs <-> raw Ethernet (RMII).

    Parameters
    ----------
    own_mac, device_serial, session_nonce : hello/discovery parameters
    fast_clk_freq    : ADAT receiver clock (free-run timer)
    rmii_clk_freq    : unused by the RTL itself (the discovery/mdio
                       parameters carry the timer periods); kept for
                       reference/tests
    hello_discover_cycles, hello_linked_cycles, keepalive_cycles :
        discovery timer overrides (tests)
    mdio_poll_cycles : MDIO poll period override (tests)

    The "rmii", "fast" and "sync" clock domains must exist.
    """

    def __init__(self, *, own_mac, device_serial=0, session_nonce=0,
                 fast_clk_freq=98_304_000, rmii_clk_freq=50_000_000,
                 hello_discover_cycles=None, hello_linked_cycles=None,
                 keepalive_cycles=None, mdio_poll_cycles=None):
        self.own_mac = own_mac & 0xFFFFFFFFFFFF
        self.fast_clk_freq = fast_clk_freq
        self.rmii_clk_freq = rmii_clk_freq

        # MAC core user-stream boundary
        self.mac_tx_data = Signal(8)
        self.mac_tx_valid = Signal()
        self.mac_tx_ready = Signal()
        self.mac_tx_last = Signal()
        # (the core's sink.be/sink.error are tied: be=1, error=0)

        self.mac_rx_data = Signal(8)
        self.mac_rx_valid = Signal()
        self.mac_rx_ready = Signal()
        self.mac_rx_last = Signal()
        self.mac_rx_error = Signal()

        # audio-side builder/feeder interfaces are exposed as attributes
        self.builder = AudioBlockBuilder(fast_clk_freq)
        self.feeder = AudioFeeder()

        self.discovery = Discovery(
            own_mac=self.own_mac, device_serial=device_serial,
            session_nonce=session_nonce, rmii_clk_freq=rmii_clk_freq,
            hello_discover_cycles=hello_discover_cycles,
            hello_linked_cycles=hello_linked_cycles,
            keepalive_cycles=keepalive_cycles)
        self.mdio = MDIOMaster(
            poll_cycles=(mdio_poll_cycles if mdio_poll_cycles is not None
                         else 1 << 22))

        # status
        self.link_up = Signal()          # MDIO BMSR link bit
        self.pc_linked = Signal()        # discovery: PC MAC learned
        self.free_run = Signal()         # no ADAT input synced
        self.stats_tx_frames = Signal(32)
        self.stats_tx_drops = Signal(32)
        self.stats_rx_frames = Signal(32)
        self.stats_rx_errors = Signal(32)   # bad FCS / header / short
        self.stats_rx_drops = Signal(32)    # dropped: buffers busy
        self.stats_rx_gaps = Signal(32)
        self.stats_rx_dups = Signal(32)
        self.stats_hello_tx = Signal(32)
        self.stats_hello_rx = Signal(32)
        self.stats_filtered = Signal(32)    # wrong dst / own MAC echo
        # test observation points (the header-decision conditions)
        self.dbg_dst_ok = Signal()
        self.dbg_self_echo = Signal()
        self.dbg_type_audio = Signal()
        self.dbg_type_hello = Signal()
        self.dbg_src_ok = Signal()
        self.dbg_hdr_valid = Signal()
        self.dbg_rx_state = Signal(3)
        self.dbg_rx_cnt = Signal(12)
        self.dbg_rx_eth_type = Signal(16)

        # the feeder reads the rx payload memory created below
        self.rx_ram = Memory(width=32, depth=2 * BLOCK_WORDS_RX,
                             name="audio_rx_block")

    def elaborate(self, platform):
        m = Module()
        rmii = m.d.rmii
        comb = m.d.comb

        builder = self.builder
        feeder = self.feeder.attach(self.rx_ram)
        discovery = self.discovery
        mdio = self.mdio
        m.submodules.builder = builder
        m.submodules.feeder = feeder
        m.submodules.discovery = discovery
        m.submodules.mdio = mdio

        #
        # gray-counter CDCs between the three domains
        #
        tx_prod_seen = Signal(2)         # fast -> rmii (builder produced)
        tx_cons_gray = Signal(2)         # rmii -> fast (we consumed)
        rx_prod_gray = Signal(2)         # rmii -> sync (we produced)
        rx_cons_seen = Signal(2)         # sync -> rmii (feeder consumed)
        m.submodules += [
            FFSynchronizer(builder.prod_cnt_gray_out, tx_prod_seen,
                           o_domain="rmii"),
            FFSynchronizer(tx_cons_gray, builder.cons_cnt_gray_in,
                           o_domain="fast"),
            FFSynchronizer(rx_prod_gray, feeder.prod_cnt_gray_in,
                           o_domain="sync"),
            FFSynchronizer(feeder.cons_cnt_gray_out, rx_cons_seen,
                           o_domain="rmii"),
        ]
        # status level crossing fast -> rmii (hello flags bit): must not
        # reach the discovery module's rmii registers combinationally
        audio_active_rmii = Signal()
        m.submodules += FFSynchronizer(
            (builder.synced_in[0] | builder.synced_in[1]
             | builder.synced_in[2]),
            audio_active_rmii, o_domain="rmii")
        # drop strobe crossing fast -> rmii (the builder stretches it
        # to 4 fast cycles): edge-detect and count in this domain
        tx_drops_seen = Signal()
        tx_drops_prev = Signal()
        m.submodules += FFSynchronizer(builder.tx_drops_out,
                                       tx_drops_seen, o_domain="rmii")

        #
        # status
        #
        comb += [
            self.link_up.eq(mdio.link_up_out),
            self.pc_linked.eq(discovery.linked_out),
            self.free_run.eq(builder.free_run_out),
            discovery.audio_active_in.eq(audio_active_rmii),
        ]

        # ==================================================================
        # TX path: block payload -> MAC sink (audio frames), arbitered
        # against the discovery module's hello frames.
        # ==================================================================
        tx_prod_b1, tx_prod_b0 = gray2bin2_terms(tx_prod_seen[1],
                                                 tx_prod_seen[0])
        tx_prod_val = Signal(2)
        tx_cons_cnt = Signal(2)
        have_tx_work = Signal()
        comb += [
            tx_prod_val.eq(Cat(tx_prod_b0, tx_prod_b1)),
            have_tx_work.eq(tx_prod_val != tx_cons_cnt),
        ]

        tx_ram_read = builder.block_ram.read_port(domain="rmii")
        m.submodules += tx_ram_read

        tx_sending = Signal()
        tx_buf = Signal(1)
        tx_byte = Signal(11)             # 0..TX_FRAME_BYTES-1
        sample_cntr = Signal(32)

        # MAC header bytes 0..21
        own = self.own_mac
        dst = Mux(discovery.linked_out, discovery.pc_mac_out,
                  BROADCAST_MAC)
        hdr_terms = [
            dst[40:48], dst[32:40], dst[24:32],
            dst[16:24], dst[8:16], dst[0:8],
            (own >> 40) & 0xFF, (own >> 32) & 0xFF, (own >> 24) & 0xFF,
            (own >> 16) & 0xFF, (own >> 8) & 0xFF, own & 0xFF,
            ETHERTYPE_AUDIO >> 8, ETHERTYPE_AUDIO & 0xFF,
            MAGIC, VER_FLAGS_TX,
        ]
        for shift in (0, 8, 16, 24):     # sample_cntr, little-endian
            hdr_terms.append((sample_cntr >> shift) & 0xFF)
        hdr_terms += [PAYLOAD_LEN_TX & 0xFF, PAYLOAD_LEN_TX >> 8]
        hdr_byte = Array(hdr_terms)
        hdr_index = Mux(tx_byte < TX_HDR_BYTES, tx_byte, 0)

        tx_pres = tx_byte - TX_HDR_BYTES  # payload byte index (>= 22)

        # Payload presentation: the sync read port's registered output
        # IS the byte currently being presented (addr always runs one
        # byte ahead, so during cycle C the port outputs the byte that
        # this cycle presents). The first payload byte's read is issued
        # during the header's last byte (addr = byte 0 there).
        next_pres = tx_pres + 1
        comb += [
            tx_ram_read.addr.eq(tx_buf * TX_BUFFER_BYTES
                                + Mux(next_pres < 1440, next_pres, 1439)),
            tx_ram_read.en.eq(tx_sending & (tx_byte >= TX_HDR_BYTES - 1)
                              & (tx_byte < TX_FRAME_BYTES - 1)),
        ]

        # byte currently presented by the audio TX FSM
        tx_fsm_valid = tx_sending
        tx_fsm_data = Mux(tx_byte < TX_HDR_BYTES, hdr_byte[hdr_index],
                          tx_ram_read.data)
        tx_fsm_last = tx_byte == TX_FRAME_BYTES - 1

        # TX arbiter: audio frames first, then hello
        hello_grant = ~tx_sending & discovery.hello_valid
        comb += [
            self.mac_tx_valid.eq(tx_fsm_valid | hello_grant),
            self.mac_tx_data.eq(Mux(tx_sending, tx_fsm_data,
                                    discovery.hello_data)),
            self.mac_tx_last.eq(Mux(tx_sending, tx_fsm_last,
                                    discovery.hello_last)),
            discovery.hello_ready.eq(self.mac_tx_ready & hello_grant),
            discovery.hello_granted.eq(hello_grant),
        ]

        tx_consume = tx_fsm_valid & self.mac_tx_ready
        with m.If(tx_sending):
            with m.If(tx_consume):
                rmii += tx_byte.eq(tx_byte + 1)
                with m.If(tx_fsm_last):
                    rmii += [
                        tx_sending.eq(0),
                        tx_cons_cnt.eq(tx_cons_cnt + 1),
                        # gray of the NEW count (G(b+1): g1 = b1^b0,
                        # g0 = ~b1 for 2 bits; Cat is LSB-first)
                        tx_cons_gray.eq(Cat(
                            ~tx_cons_cnt[1],
                            tx_cons_cnt[1] ^ tx_cons_cnt[0])),
                        sample_cntr.eq(sample_cntr + N_SAMPLES),
                        self.stats_tx_frames.eq(self.stats_tx_frames + 1),
                    ]
        with m.If(~tx_sending & have_tx_work):
            rmii += [
                tx_sending.eq(1),
                tx_buf.eq(tx_cons_cnt[0]),
                tx_byte.eq(0),
            ]

        # count builder block drops (stretched fast strobe, edge here)
        rmii += tx_drops_prev.eq(tx_drops_seen)
        with m.If(tx_drops_seen & ~tx_drops_prev):
            rmii += self.stats_tx_drops.eq(self.stats_tx_drops + 1)

        # ==================================================================
        # RX path: MAC source -> payload buffers (never backpressures)
        # ==================================================================
        rx_ram_write = self.rx_ram.write_port(domain="rmii")
        m.submodules += rx_ram_write

        rx_state = Signal(3, reset=RX_HEADER)
        rx_cnt = Signal(12)              # frame byte counter
        rx_dst_mac = Signal(48)
        rx_src_mac = Signal(48)
        rx_eth_type = Signal(16)
        # audio payload header fields (little-endian assembly)
        rx_magic = Signal(8)
        rx_vflags = Signal(8)
        rx_cntr = Signal(32)
        rx_plen = Signal(16)
        rx_word = Signal(32)             # 4-byte big-endian accumulator
        rx_hello_sh = Signal(160)
        rx_wr_buf = Signal(1)
        # latched hello verdict, applied at frame end (FCS is known then)
        rx_hel_ok = Signal()
        rx_hel_role = Signal(8)
        rx_tail_kind = Signal(2)
        # received-direction sample counter tracking (gap detection)
        last_rx_cntr = Signal(32)
        rx_seen_cntr = Signal()

        rx_prod_cnt = Signal(2)
        rx_prod_b1, rx_prod_b0 = gray2bin2_terms(rx_prod_cnt[1],
                                                 rx_prod_cnt[0])
        rx_cons_b1, rx_cons_b0 = gray2bin2_terms(rx_cons_seen[1],
                                                 rx_cons_seen[0])
        rx_in_flight = Signal(2)
        comb += [
            rx_prod_gray.eq(Cat(rx_prod_b0, rx_prod_b1)),
            rx_in_flight.eq(rx_prod_cnt - Cat(rx_cons_b0, rx_cons_b1)),
            self.mac_rx_ready.eq(1),
        ]

        dst_ok = (rx_dst_mac == self.own_mac) | \
                 (rx_dst_mac == BROADCAST_MAC)
        self_echo = rx_src_mac == self.own_mac
        # EtherType / payload_len, completed combinationally with the
        # byte arriving this cycle (decisions happen on the last byte).
        # rx_eth_type shifts left per byte, so byte 12 (the high byte
        # on the wire) sits in the register's LOW byte; the arriving
        # byte 13 is the low byte of the type.
        type_now = Cat(self.mac_rx_data, rx_eth_type[0:8])
        type_audio = type_now == ETHERTYPE_AUDIO
        type_hello = type_now == ETHERTYPE_HELLO
        src_ok = ~discovery.linked_out | (rx_src_mac
                                          == discovery.pc_mac_out)
        plen_now = (rx_plen >> 8) | (self.mac_rx_data << 8)
        hdr_valid = (rx_magic == MAGIC) & (rx_vflags == 0x08) & \
                    (plen_now == PAYLOAD_LEN_RX)
        comb += [
            self.dbg_dst_ok.eq(dst_ok),
            self.dbg_self_echo.eq(self_echo),
            self.dbg_type_audio.eq(type_audio),
            self.dbg_type_hello.eq(type_hello),
            self.dbg_src_ok.eq(src_ok),
            self.dbg_hdr_valid.eq(hdr_valid),
            self.dbg_rx_state.eq(rx_state),
            self.dbg_rx_cnt.eq(rx_cnt),
            self.dbg_rx_eth_type.eq(rx_eth_type),
        ]

        # audio payload byte index within the frame (valid in RX_AUDIO_PL)
        rx_p = rx_cnt - RX_HDR_END

        # hello payload fields. At the verdict latch (rx_cnt == 33, the
        # 20th payload byte arriving) the shift register holds payload
        # bytes 0..18: byte j sits at [(18-j)*8 : (19-j)*8). Byte 19
        # (the second reserved byte) is on mac_rx_data at that edge.
        hel_id_ok = (rx_hello_sh[144:152] == 0x41) & \
                    (rx_hello_sh[136:144] == 0x44) & \
                    (rx_hello_sh[128:136] == 0x41) & \
                    (rx_hello_sh[120:128] == 0x54)
        hel_ver_ok = (rx_hello_sh[96:104] == 1) & \
                     (rx_hello_sh[88:96] == 0)
        hel_res_ok = rx_hello_sh[0:8] == 0
        hel_ok_live = hel_id_ok & hel_ver_ok & hel_res_ok & \
                      (self.mac_rx_data == 0)

        # audio payload write port: a word every 4th payload byte
        comb += [
            rx_ram_write.en.eq((rx_state == RX_AUDIO_PL)
                               & self.mac_rx_valid
                               & (rx_p[0:2] == 3)
                               & (rx_p < RX_AUDIO_BYTES)),
            rx_ram_write.addr.eq(rx_wr_buf * BLOCK_WORDS_RX
                                 + (rx_p >> 2)),
            rx_ram_write.data.eq((rx_word << 8) | self.mac_rx_data),
        ]

        # discovery strobes (combinationally derived from the frame end)
        rx_last_edge = self.mac_rx_valid & self.mac_rx_last
        audio_accept = (rx_state == RX_AUDIO_PL) & rx_last_edge & \
            (rx_p == RX_AUDIO_BYTES - 1) & ~self.mac_rx_error
        tail_accept = (rx_state == RX_TAIL) & rx_last_edge & \
            ~self.mac_rx_error
        comb += [
            discovery.rx_src_mac_in.eq(rx_src_mac),
            discovery.rx_role_in.eq(rx_hel_role),
            discovery.rx_frame_strobe.eq(0),
            discovery.rx_is_hello_in.eq(0),
            discovery.rx_is_audio_in.eq(0),
            discovery.rx_hello_valid_in.eq(0),
        ]
        with m.If(audio_accept | (tail_accept & (rx_tail_kind == TAIL_AUDIO))):
            comb += [
                discovery.rx_frame_strobe.eq(1),
                discovery.rx_is_audio_in.eq(1),
            ]
        with m.If(tail_accept & (rx_tail_kind == TAIL_HELLO)):
            comb += [
                discovery.rx_frame_strobe.eq(1),
                discovery.rx_is_hello_in.eq(1),
                discovery.rx_hello_valid_in.eq(rx_hel_ok),
            ]
        # unpadded hello: the verdict is applied directly at the last
        # payload byte (the latched one would never be consumed)
        hello_last_byte = (rx_state == RX_HELLO_PL) & rx_last_edge & \
                          (rx_cnt == 14 + HELLO_LEN - 1)
        with m.If(hello_last_byte):
            comb += [
                discovery.rx_frame_strobe.eq(1),
                discovery.rx_is_hello_in.eq(1),
                discovery.rx_hello_valid_in.eq(hel_ok_live),
            ]

        # ------------------------------------------------------------------
        # RX parser state machine (one byte per valid cycle; ready == 1)
        # ------------------------------------------------------------------
        with m.If(self.mac_rx_valid):
            with m.Switch(rx_state):
                with m.Case(RX_HEADER):
                    with m.If(rx_cnt < 6):
                        rmii += rx_dst_mac.eq(
                            (rx_dst_mac << 8) | self.mac_rx_data)
                    with m.Elif(rx_cnt < 12):
                        rmii += rx_src_mac.eq(
                            (rx_src_mac << 8) | self.mac_rx_data)
                    with m.Else():
                        rmii += rx_eth_type.eq(
                            (rx_eth_type << 8) | self.mac_rx_data)
                    with m.If(rx_cnt == 13):
                        with m.If(~dst_ok | self_echo
                                  | ~(type_audio | type_hello)):
                            rmii += [
                                rx_state.eq(RX_TAIL),
                                rx_tail_kind.eq(TAIL_FILTERED),
                                rx_hel_ok.eq(0),
                                self.stats_filtered.eq(
                                    self.stats_filtered + 1),
                            ]
                        with m.Elif(type_hello):
                            rmii += [
                                rx_state.eq(RX_HELLO_PL),
                                rx_hel_ok.eq(0),
                            ]
                        with m.Elif(~src_ok):
                            rmii += [
                                rx_state.eq(RX_TAIL),
                                rx_tail_kind.eq(TAIL_FILTERED),
                                rx_hel_ok.eq(0),
                                self.stats_filtered.eq(
                                    self.stats_filtered + 1),
                            ]
                        with m.Else():
                            rmii += rx_state.eq(RX_AUDIO_HDR)

                with m.Case(RX_AUDIO_HDR):
                    with m.If(rx_cnt == RX_HDR_END - 8):
                        rmii += rx_magic.eq(self.mac_rx_data)
                    with m.Elif(rx_cnt == RX_HDR_END - 7):
                        rmii += rx_vflags.eq(self.mac_rx_data)
                    with m.If((rx_cnt >= RX_HDR_END - 6)
                              & (rx_cnt < RX_HDR_END - 2)):
                        # little-endian: byte 16 is the LSB; shift right
                        rmii += rx_cntr.eq(
                            (rx_cntr >> 8) | (self.mac_rx_data << 24))
                    with m.If(rx_cnt >= RX_HDR_END - 2):
                        rmii += rx_plen.eq(
                            (rx_plen >> 8) | (self.mac_rx_data << 8))
                    with m.If(rx_cnt == RX_HDR_END - 1):
                        # header complete: validate + counter tracking
                        with m.If(~hdr_valid):
                            rmii += [
                                rx_state.eq(RX_TAIL),
                                rx_tail_kind.eq(TAIL_FILTERED),
                                self.stats_rx_errors.eq(
                                    self.stats_rx_errors + 1),
                            ]
                        with m.Else():
                            # sample_cntr gap/dup bookkeeping (stats only)
                            diff = rx_cntr - last_rx_cntr
                            with m.If(rx_seen_cntr):
                                with m.If((diff == N_SAMPLES) | (diff == 0)):
                                    with m.If(diff == 0):
                                        rmii += self.stats_rx_dups.eq(
                                            self.stats_rx_dups + 1)
                                with m.Elif(diff < 0x80000000):
                                    rmii += self.stats_rx_gaps.eq(
                                        self.stats_rx_gaps + 1)
                                with m.Else():
                                    rmii += self.stats_rx_dups.eq(
                                        self.stats_rx_dups + 1)
                            rmii += [
                                last_rx_cntr.eq(rx_cntr),
                                rx_seen_cntr.eq(1),
                            ]
                            # buffer availability
                            with m.If(rx_in_flight == 2):
                                rmii += [
                                    rx_state.eq(RX_TAIL),
                                    rx_tail_kind.eq(TAIL_FILTERED),
                                    self.stats_rx_drops.eq(
                                        self.stats_rx_drops + 1),
                                ]
                            with m.Else():
                                rmii += [
                                    rx_state.eq(RX_AUDIO_PL),
                                    rx_wr_buf.eq(rx_prod_cnt[0]),
                                    rx_word.eq(0),
                                ]

                with m.Case(RX_AUDIO_PL):
                    rmii += rx_word.eq((rx_word << 8) | self.mac_rx_data)
                    with m.If(self.mac_rx_last):
                        with m.If((rx_p == RX_AUDIO_BYTES - 1)
                                  & ~self.mac_rx_error):
                            rmii += [
                                rx_state.eq(RX_HEADER),
                                rx_prod_cnt.eq(rx_prod_cnt + 1),
                                self.stats_rx_frames.eq(
                                    self.stats_rx_frames + 1),
                            ]
                        with m.Else():
                            rmii += [
                                rx_state.eq(RX_HEADER),
                                self.stats_rx_errors.eq(
                                    self.stats_rx_errors + 1),
                            ]
                    with m.Elif(rx_p == RX_AUDIO_BYTES):
                        # padded audio frame: the PC never pads these
                        rmii += [
                            rx_state.eq(RX_TAIL),
                            rx_tail_kind.eq(TAIL_FILTERED),
                            self.stats_rx_errors.eq(
                                self.stats_rx_errors + 1),
                        ]

                with m.Case(RX_HELLO_PL):
                    rmii += rx_hello_sh.eq(
                        (rx_hello_sh << 8) | self.mac_rx_data)
                    with m.If(rx_cnt == 14 + HELLO_LEN - 1):
                        # latch the verdict for the (usual) padded case
                        rmii += [
                            rx_hel_ok.eq(hel_ok_live),
                            rx_hel_role.eq(rx_hello_sh[112:120]),
                            rx_tail_kind.eq(TAIL_HELLO),
                        ]
                        with m.If(self.mac_rx_last):
                            # unpadded hello: the frame ends here;
                            # apply the verdict directly (the latched
                            # one would never be consumed)
                            rmii += rx_state.eq(RX_HEADER)
                        with m.Else():
                            rmii += rx_state.eq(RX_TAIL)
                    with m.Elif(self.mac_rx_last):
                        # hello payload ended before 20 bytes
                        rmii += [
                            rx_state.eq(RX_HEADER),
                            self.stats_rx_errors.eq(
                                self.stats_rx_errors + 1),
                        ]

                with m.Case(RX_TAIL):
                    with m.If(self.mac_rx_last):
                        with m.If(self.mac_rx_error
                                  & (rx_tail_kind != TAIL_FILTERED)):
                            rmii += self.stats_rx_errors.eq(
                                self.stats_rx_errors + 1)
                        rmii += rx_state.eq(RX_HEADER)

            rmii += rx_cnt.eq(rx_cnt + 1)
            with m.If(self.mac_rx_last):
                rmii += rx_cnt.eq(0)

        # hello accept counter (valid PC hello)
        with m.If(tail_accept & (rx_tail_kind == TAIL_HELLO) & rx_hel_ok):
            rmii += self.stats_hello_rx.eq(self.stats_hello_rx + 1)

        return m
