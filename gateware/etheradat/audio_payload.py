"""Audio block builder (ADAT -> Ethernet payload) and feeder (Ethernet
payload -> ADAT), the two audio-side datapaths of the EtherADAT bridge.

TX direction (device -> PC, PROTOCOL.md §4.1): the ADAT receivers are
collected into 15-slot rings of canonical 32-byte frames by
InputFrameCollectors; AudioBlockBuilder snapshots the rings into one of
two ping-pong block buffers (slot order: sample-major, then input-major,
i.e. slot (s*3+i)) and hands the buffer to the RMII-domain sender via
gray-coded produce/consume counters. The block pace is the wordclock of
the first synced ADAT input; with no input synced, a free-run timer at
exactly 3200 blocks/s (48 kHz / 15) paces silent EMPTY_USER_FRAME
blocks (PROTOCOL.md §6: unsynced inputs go out as silence).

RX direction (PC -> device, §4.2): AudioFeeder consumes one of two
ping-pong payload buffers written by the RMII-domain receiver
(payload = 15 samples x 2 outputs of 32-byte frames, slot (s*2+o)) and
feeds the two ADATTransmitters (ch0..ch7 in order, user bits on last).
It is paced by the transmitters' ready handshake, i.e. by the local
ADAT transmit clock; a stale buffer is re-fed (repeat) until a fresh
one arrives. Overrun is dropped by the RMII-domain receiver
(PROTOCOL.md §7 adapter semantics).

Buffer hand-off both ways uses the same mailbox: a 2-bit produce counter
(gray-coded towards the other domain) and a 2-bit consume counter
(gray-coded back). With two buffers, "(produce - consume) mod 4" is the
number of buffers in flight (0..2) and the parity selects the buffer —
no multi-bit level signals cross a clock domain.

All modules take their clock domain name as a parameter; the memories
created inside a module use that domain for their ports.
"""

from amaranth import Elaboratable, Module, Signal, Array, Mux, Memory, Cat

from etheradat.adat_frame_codec import InputFrameCollector, FrameDeserializer
from etheradat.reference import EMPTY_USER_FRAME

TX_BUFFER_BYTES = 1440       # 15 samples x 3 inputs x 32 bytes
RX_BUFFER_BYTES = 960        # 15 samples x 2 outputs x 32 bytes
BLOCK_WORDS_TX = TX_BUFFER_BYTES          # tx block RAM is byte-wide
BLOCK_WORDS_RX = RX_BUFFER_BYTES // 4     # rx block RAM is 32-bit wide
BLOCK_SLOTS_RX = 30                       # 15 samples x 2 outputs


def bin2gray2_terms(b1, b0):
    """Gray code terms (g1, g0) of a 2-bit binary value."""
    return [b1, b1 ^ b0]


def gray2bin2_terms(g1, g0):
    """Binary terms (b1, b0) of a 2-bit gray value."""
    return [g1, g1 ^ g0]


class AudioBlockBuilder(Elaboratable):
    """Collects 3 ADAT inputs and builds TX-direction block payloads.

    Runs in the ADAT receiver clock domain ("fast" by default).
    Instantiates three InputFrameCollectors (their receiver-facing
    signals are exposed as lists on this module) and the two-block
    ping-pong payload RAM (byte-wide; the RMII-domain sender reads it
    byte-serially through its own read port).

    Attributes
    ----------
    sample_in[i], addr_in[i], enable_in[i], user_in[i], synced_in[i]
                         : receiver-facing inputs of collector i
    block_ram            : Memory(8, 2*TX_BUFFER_BYTES); the RMII-domain
                           sender adds its own read port
    cons_cnt_gray_in     : sender's consume counter (gray), already
                           synchronized into this domain by the owner
    prod_cnt_gray_out    : our produce counter (gray), to be synchronized
                           into the RMII domain by the owner
    tx_drops_out         : strobe: a completed block was dropped (both
                           buffers still in flight)
    free_run_out         : level: no ADAT input synced (free-running)
    dbg_prod_cnt_out     : produce counter (binary, for tests/ILA)
    clk_freq             : receiver clock frequency, for the free-run
                           timer (clk_freq * 15 / 48000 cycles per block)
    """

    N_INPUTS = 3

    def __init__(self, clk_freq, domain="fast"):
        self.clk_freq = clk_freq
        self.domain = domain
        self.sample_in = [Signal(24, name=f"bb_sample{i}")
                          for i in range(self.N_INPUTS)]
        self.addr_in = [Signal(3, name=f"bb_addr{i}")
                        for i in range(self.N_INPUTS)]
        self.enable_in = [Signal(name=f"bb_enable{i}")
                          for i in range(self.N_INPUTS)]
        self.user_in = [Signal(4, name=f"bb_user{i}")
                        for i in range(self.N_INPUTS)]
        self.synced_in = [Signal(name=f"bb_synced{i}")
                          for i in range(self.N_INPUTS)]

        self.block_ram = Memory(width=8, depth=2 * TX_BUFFER_BYTES,
                                name="audio_tx_block")
        self.cons_cnt_gray_in = Signal(2)
        self.prod_cnt_gray_out = Signal(2)
        self.tx_drops_out = Signal()
        self.free_run_out = Signal()
        self.dbg_prod_cnt_out = Signal(2)

    def elaborate(self, platform):
        m = Module()
        sync = m.d[self.domain]
        comb = m.d.comb

        collectors = []
        for i in range(self.N_INPUTS):
            collector = InputFrameCollector(domain=self.domain)
            collector.sample_in = self.sample_in[i]
            collector.addr_in = self.addr_in[i]
            collector.enable_in = self.enable_in[i]
            collector.user_in = self.user_in[i]
            collector.synced_in = self.synced_in[i]
            m.submodules[f"collector{i}"] = collector
            collectors.append(collector)
        read_ports = [collector.ring.read_port(domain=self.domain)
                      for collector in collectors]
        for rp in read_ports:
            m.submodules += rp
        rp_data = Array([rp.data for rp in read_ports])

        write_port = self.block_ram.write_port(domain=self.domain)
        m.submodules += write_port

        block_cycles = self.clk_freq * 15 // 48000

        # produce/consume mailbox (all in this domain; the gray signals
        # cross domains through FFSynchronizers owned by the bridge)
        prod_cnt = Signal(2)
        prod_gray = Signal(2)
        cons_gray_seen = Signal(2)
        cons_b1, cons_b0 = gray2bin2_terms(cons_gray_seen[1],
                                           cons_gray_seen[0])
        in_flight = Signal(2)
        comb += [
            # our view of the sender's consume counter (gray), synced
            # into this domain by the owner through cons_cnt_gray_in
            cons_gray_seen.eq(self.cons_cnt_gray_in),
            self.prod_cnt_gray_out.eq(prod_gray),
            self.dbg_prod_cnt_out.eq(prod_cnt),
            in_flight.eq(prod_cnt - Cat(cons_b0, cons_b1)),
        ]

        # master input: the first synced one (bogus if none synced; the
        # free-run timer takes over the block pace then)
        master = Signal(2)
        wrapped_i = Array([c.slot_wrapped_out for c in collectors])
        comb += [
            master.eq(Mux(self.synced_in[0], 0,
                          Mux(self.synced_in[1], 1, 2))),
            self.free_run_out.eq(~(self.synced_in[0] | self.synced_in[1]
                                   | self.synced_in[2])),
        ]

        # copy state: s/i select the payload slot (s*3+i), w the ring
        # word, lane the byte within the 32-bit ring word
        copying = Signal()
        buf = Signal(1)
        s_cnt = Signal(4)        # sample index 0..14
        i_cnt = Signal(2)        # input index 0..2
        w_cnt = Signal(4)        # ring word 0..7
        lane = Signal(2)         # byte lane within the ring word
        got_word = Signal()      # ring word read, byte writes pending
        # running byte address: the copy walks the buffer strictly in
        # ascending order, so a single counter replaces the
        # buf*1440 + (s*3+i)*32 + w*4 + lane adder tree (which did not
        # close timing at 98.304 MHz)
        wr_byte = Signal(12)

        # pause a collector's ring writes while its ring is being read
        for i in range(self.N_INPUTS):
            comb += collectors[i].pause_in.eq(copying & (i_cnt == i)
                                              & ~got_word)

        # ring read: issue in the first cycle of a ring word
        for j in range(self.N_INPUTS):
            comb += [
                read_ports[j].en.eq(copying & ~got_word & (i_cnt == j)),
                read_ports[j].addr.eq(s_cnt * 8 + w_cnt),
            ]

        # block ram write: 4 byte lanes per ring word
        # byte lane order within the 32-bit ring word: MSB first
        byte_sel = Array([3, 2, 1, 0])[lane]
        comb += [
            write_port.en.eq(copying & got_word),
            write_port.addr.eq(wr_byte),
            write_port.data.eq(rp_data[i_cnt].word_select(byte_sel, 8)),
        ]

        timer = Signal(16)
        trigger = Signal()
        # drop strobe stretch: a 1-cycle fast pulse can fall entirely
        # between two rmii sampling edges; hold drops for 4 cycles so
        # the bridge's edge detector cannot miss them
        drop_hold = Signal(2)

        # default strobes (conditional assignments below override)
        sync += [
            trigger.eq(0),
        ]

        # drop strobe stretch: the drop site sets drop_hold for one
        # cycle; holding it for 4 fast cycles guarantees the bridge's
        # rmii edge detector sees every drop
        with m.If(drop_hold != 0):
            sync += drop_hold.eq(drop_hold - 1)
        comb += self.tx_drops_out.eq(drop_hold != 0)

        # free-run timer
        with m.If(trigger | (timer == block_cycles - 1)):
            sync += timer.eq(0)
        with m.Else():
            sync += timer.eq(timer + 1)

        # block trigger: master input's ring wrapped, or free-run period
        with m.If(~copying):
            with m.If((~self.free_run_out & wrapped_i[master])
                      | (self.free_run_out & (timer == block_cycles - 1))):
                with m.If(in_flight == 2):
                    # both buffers still unconsumed: drop the newest
                    sync += drop_hold.eq(3)
                with m.Else():
                    sync += [
                        copying.eq(1),
                        buf.eq(prod_cnt[0]),
                        wr_byte.eq(prod_cnt[0] * TX_BUFFER_BYTES),
                        s_cnt.eq(0),
                        i_cnt.eq(0),
                        w_cnt.eq(0),
                        lane.eq(0),
                        got_word.eq(0),
                    ]

        # copy FSM: one ring read, then 4 byte writes per ring word
        with m.If(copying):
            with m.If(~got_word):
                sync += got_word.eq(1)
            with m.Else():
                sync += [
                    lane.eq(lane + 1),
                    wr_byte.eq(wr_byte + 1),
                ]
                with m.If(lane == 3):
                    sync += lane.eq(0)
                    with m.If(w_cnt == 7):
                        sync += w_cnt.eq(0)
                        with m.If(i_cnt == 2):
                            sync += i_cnt.eq(0)
                            with m.If(s_cnt == 14):
                                # block complete: publish the buffer
                                # (gray of the NEW count, written as a
                                # function of the old one: G(b+1) has
                                # g1 = b1 ^ b0, g0 = ~b1 for 2 bits)
                                sync += [
                                    copying.eq(0),
                                    got_word.eq(0),
                                    prod_cnt.eq(prod_cnt + 1),
                                    prod_gray.eq(Cat(
                                        ~prod_cnt[1],
                                        prod_cnt[1] ^ prod_cnt[0])),
                                ]
                            with m.Else():
                                sync += [
                                    s_cnt.eq(s_cnt + 1),
                                    got_word.eq(0),
                                ]
                        with m.Else():
                            sync += [
                                i_cnt.eq(i_cnt + 1),
                                got_word.eq(0),
                            ]
                    with m.Else():
                        sync += [
                            w_cnt.eq(w_cnt + 1),
                            got_word.eq(0),
                        ]

        return m


class AudioFeeder(Elaboratable):
    """Feeds the two ADATTransmitters from RX-direction payload buffers.

    Runs in the ADAT transmitters' sample-input clock domain ("sync" by
    default). The owner provides the ping-pong payload Memory (written
    by the RMII-domain receiver: 2 buffers x 240 words) and
    synchronizes the gray counters.

    Attributes
    ----------
    payload_ram          : Memory(32, 2*240), set via attach()
    prod_cnt_gray_in     : receiver's produce counter (gray), already
                           synchronized into this domain by the owner
    cons_cnt_gray_out    : our consume counter (gray), to be synchronized
                           into the RMII domain by the owner
    sample_out[o], addr_out[o], valid_out[o], last_out[o], user_out[o],
    ready_in[o]          : per-ADATTransmitter commit interface; the
                           transmitter commits when valid_out & ready_in
    dbg_cons_cnt_out     : consume counter (binary, for tests/ILA)
    """

    N_OUTPUTS = 2
    WORDS_PER_SLOT = 8

    def __init__(self, domain="sync"):
        self.domain = domain
        self.payload_ram = None            # attach() before elaborate
        self.prod_cnt_gray_in = Signal(2)
        self.cons_cnt_gray_out = Signal(2)
        self.sample_out = [Signal(24, name=f"fd_sample{o}")
                           for o in range(self.N_OUTPUTS)]
        self.addr_out = [Signal(3, name=f"fd_addr{o}")
                         for o in range(self.N_OUTPUTS)]
        self.valid_out = [Signal(name=f"fd_valid{o}")
                          for o in range(self.N_OUTPUTS)]
        self.last_out = [Signal(name=f"fd_last{o}")
                         for o in range(self.N_OUTPUTS)]
        self.user_out = [Signal(4, name=f"fd_user{o}")
                         for o in range(self.N_OUTPUTS)]
        self.ready_in = [Signal(name=f"fd_ready{o}")
                         for o in range(self.N_OUTPUTS)]
        self.dbg_cons_cnt_out = Signal(2)
        # test observation points (driven in elaborate)
        self.dbg_active = Signal()
        self.dbg_burst = Signal()
        self.dbg_committing = Signal()
        self.dbg_slot = Signal(5)
        self.dbg_deserializer_valid = Signal()
        self.dbg_have_work = Signal()

    def attach(self, payload_ram):
        self.payload_ram = payload_ram
        return self

    def elaborate(self, platform):
        assert self.payload_ram is not None, "call attach() first"
        m = Module()
        sync = m.d[self.domain]
        comb = m.d.comb

        deserializer = FrameDeserializer(self.payload_ram,
                                         domain=self.domain)
        m.submodules.deserializer = deserializer

        prod_seen = Signal(2)
        prod_b1, prod_b0 = gray2bin2_terms(prod_seen[1], prod_seen[0])
        prod_bin = Signal(2)
        cons_cnt = Signal(2)
        cons_gray = Signal(2)
        have_work = Signal()
        in_flight = Signal(2)
        comb += [
            # the owner synchronizes the producer's gray counter into
            # this domain and drives prod_cnt_gray_in
            prod_seen.eq(self.prod_cnt_gray_in),
            prod_bin.eq(Cat(prod_b0, prod_b1)),
            self.cons_cnt_gray_out.eq(cons_gray),
            self.dbg_cons_cnt_out.eq(cons_cnt),
            have_work.eq(prod_bin != cons_cnt),
            in_flight.eq(prod_bin - cons_cnt),
            self.dbg_deserializer_valid.eq(
                deserializer.samples_valid_out),
            self.dbg_have_work.eq(have_work),
        ]

        # buffer currently being read
        cur_buf = Signal(1)

        # slot/channel state
        slot = Signal(5)         # payload slot 0..29 (s*2+o)
        chn = Signal(3)          # channel 0..7 within the slot
        active = Signal()        # a buffer is being fed
        burst = Signal()         # deserializer read burst running
        rd_cnt = Signal(4)       # burst cycle counter 0..7
        committing = Signal()    # handing samples to the transmitter

        cur_out = Signal(1)      # output index of the current slot
        ready_arr = Array(self.ready_in)
        deser_samples = Array(deserializer.sample_out)
        comb += [
            cur_out.eq(slot[0]),
            self.dbg_active.eq(active),
            self.dbg_burst.eq(burst),
            self.dbg_committing.eq(committing),
            self.dbg_slot.eq(slot),
        ]

        # deserializer interface: one 8-word burst per slot
        comb += [
            deserializer.read_base_in.eq(
                cur_buf * BLOCK_WORDS_RX + slot * 8),
            deserializer.read_strobe_in.eq(burst),
        ]

        # transmitter-facing outputs (active only for the current output)
        for o in range(self.N_OUTPUTS):
            comb += [
                self.sample_out[o].eq(deser_samples[chn]),
                self.addr_out[o].eq(chn),
                self.valid_out[o].eq(committing & (cur_out == o)),
                self.last_out[o].eq((chn == 7)),
                self.user_out[o].eq(deserializer.user_out),
            ]

        with m.If(~active):
            with m.If(have_work):
                sync += [
                    active.eq(1),
                    cur_buf.eq(cons_cnt[0]),
                    slot.eq(0),
                    chn.eq(0),
                    burst.eq(1),
                    rd_cnt.eq(0),
                    committing.eq(0),
                ]
        with m.Else():
            with m.If(burst):
                with m.If(rd_cnt == 7):
                    sync += [
                        burst.eq(0),
                        rd_cnt.eq(0),
                    ]
                with m.Else():
                    sync += rd_cnt.eq(rd_cnt + 1)
            with m.Elif(deserializer.samples_valid_out):
                sync += committing.eq(1)
            with m.Elif(committing):
                with m.If(ready_arr[cur_out]):
                    # committed the sample at chn on this edge
                    with m.If(chn == 7):
                        # slot complete: next slot / next buffer
                        sync += [
                            committing.eq(0),
                            chn.eq(0),
                        ]
                        with m.If((slot == BLOCK_SLOTS_RX - 1)
                                  & (in_flight == 2)):
                            # last slot and a fresh buffer is waiting:
                            # consume it (the consume counter only moves
                            # on a real buffer change — the mailbox
                            # invariant in_flight = prod - cons <= 2
                            # depends on it)
                            sync += [
                                slot.eq(0),
                                cur_buf.eq(~cons_cnt[0]),
                                cons_cnt.eq(cons_cnt + 1),
                                # gray of the NEW count (see builder)
                                cons_gray.eq(Cat(
                                    ~cons_cnt[1],
                                    cons_cnt[1] ^ cons_cnt[0])),
                            ]
                        with m.Elif(slot == BLOCK_SLOTS_RX - 1):
                            # repeat the same buffer from the start
                            sync += slot.eq(0)
                        with m.Else():
                            sync += slot.eq(slot + 1)
                        # all three branches continue at (possibly the
                        # same) slot's first channel: re-arm the
                        # deserializer burst — without it a wrapped
                        # feeder sits idle with active=1, burst=0,
                        # committing=0 forever
                        sync += [
                            burst.eq(1),
                            rd_cnt.eq(0),
                        ]
                    with m.Else():
                        sync += chn.eq(chn + 1)

        return m
