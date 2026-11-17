"""End-to-end tests of the EtherADAT bridge at the MAC user-stream
boundary (the LiteEth MAC core contract).

The harness loops the ADAT side in simulation: three ADATTransmitters
drive the three ADATReceivers (the bridge's inputs), and the bridge's
two outputs feed two more ADATTransmitters whose commits are observed.

The MAC boundary is driven exactly per the LiteEth MAC core contract:
- TX: the bridge's byte stream is captured (frames without FCS; the
  core would pad/FCS/preamble/IFG them);
- RX: frames are played byte-serially (without preamble/FCS), with the
  error flag on the last byte for bad-FCS frames.

Run:  python -m etheradat.tests.test_bridge
"""

import os
import random
import unittest

from amaranth import *
from amaranth.sim import Simulator, Tick

from adat import ADATTransmitter, ADATReceiver

from etheradat import reference as ref
from etheradat.bridge import EtherADATBridge

FAST_CLK = 98_304_000
SYNC_CLK = 61_440_000
ADAT_CLK = 12_288_000
RMII_CLK = 50_000_000

OWN_MAC = 0x02AD6E657401
PC_MAC = 0x112233445566

# huge discovery timers = hello disabled for the audio tests
HELLO_OFF = 1 << 26


def int_to_mac(v):
    return v.to_bytes(6, "big")


class BridgeHarness(Elaboratable):
    """The bridge + in-sim ADAT loopbacks.

    - adat_in_tx[i] drives adat_rx[i] (the bridge's input i)
    - the bridge's outputs commit into adat_out_tx[o]
    - the MAC user streams are exposed for the bench
    """

    def __init__(self, *, hello_discover=HELLO_OFF, hello_linked=HELLO_OFF,
                 keepalive=HELLO_OFF):
        self.bridge = EtherADATBridge(
            own_mac=OWN_MAC, device_serial=0x42, session_nonce=0x77,
            fast_clk_freq=FAST_CLK, rmii_clk_freq=RMII_CLK,
            hello_discover_cycles=hello_discover,
            hello_linked_cycles=hello_linked,
            keepalive_cycles=keepalive,
            mdio_poll_cycles=1 << 26)
        self.adat_rx = [ADATReceiver(FAST_CLK) for _ in range(3)]
        self.adat_in_tx = [ADATTransmitter(fifo_depth=8) for _ in range(3)]
        self.adat_out_tx = [ADATTransmitter(fifo_depth=8) for _ in range(2)]

    def elaborate(self, platform):
        m = Module()
        m.submodules.bridge = self.bridge
        for i in range(3):
            m.submodules[f"adat_rx{i}"] = DomainRenamer("fast")(
                self.adat_rx[i])
            m.submodules[f"adat_in_tx{i}"] = self.adat_in_tx[i]
            m.d.comb += [
                self.adat_rx[i].adat_in.eq(self.adat_in_tx[i].adat_out),
                self.bridge.builder.sample_in[i].eq(
                    self.adat_rx[i].sample_out),
                self.bridge.builder.addr_in[i].eq(self.adat_rx[i].addr_out),
                self.bridge.builder.enable_in[i].eq(
                    self.adat_rx[i].output_enable),
                self.bridge.builder.user_in[i].eq(
                    self.adat_rx[i].user_data_out),
                self.bridge.builder.synced_in[i].eq(
                    self.adat_rx[i].synced_out),
            ]
        for o in range(2):
            m.submodules[f"adat_out_tx{o}"] = self.adat_out_tx[o]
            m.d.comb += [
                self.adat_out_tx[o].sample_in.eq(
                    self.bridge.feeder.sample_out[o]),
                self.adat_out_tx[o].addr_in.eq(
                    self.bridge.feeder.addr_out[o]),
                self.adat_out_tx[o].valid_in.eq(
                    self.bridge.feeder.valid_out[o]),
                self.adat_out_tx[o].last_in.eq(
                    self.bridge.feeder.last_out[o]),
                self.adat_out_tx[o].user_data_in.eq(
                    self.bridge.feeder.user_out[o]),
                self.bridge.feeder.ready_in[o].eq(
                    self.adat_out_tx[o].ready_out),
            ]
        return m


def run_harness(harness, processes, vcd="etheradat_bridge.vcd"):
    sim = Simulator(harness)
    sim.add_clock(1 / SYNC_CLK)
    sim.add_clock(1 / ADAT_CLK, domain="adat")
    sim.add_clock(1 / FAST_CLK, domain="fast")
    sim.add_clock(1 / RMII_CLK, domain="rmii")
    for proc, domain in processes:
        sim.add_sync_process(proc, domain=domain)
    if os.getenv("GENERATE_VCDS"):
        with sim.write_vcd(vcd):
            sim.run()
    else:
        sim.run()


def commit_frames(dut_tx, frames_with_user):
    """Commit frames (lists of 8 samples + user) into a transmitter."""
    for samples, user in frames_with_user:
        for _ in range(10):
            yield Tick("sync")
        for ch in range(8):
            yield dut_tx.sample_in.eq(samples[ch])
            yield dut_tx.addr_in.eq(ch)
            yield dut_tx.valid_in.eq(1)
            yield dut_tx.last_in.eq(ch == 7)
            yield dut_tx.user_data_in.eq(user)
            while not (yield dut_tx.ready_out):
                yield Tick("sync")
            yield Tick("sync")
        yield dut_tx.valid_in.eq(0)
        yield dut_tx.last_in.eq(0)


def play_rx_frame(harness, frame, error=0):  # noqa: E301
    """Play one frame into the bridge's MAC RX stream (no FCS: the
    core would have stripped it); error flag on the last byte."""
    for i, byte in enumerate(frame):
        yield harness.bridge.mac_rx_data.eq(byte)
        yield harness.bridge.mac_rx_valid.eq(1)
        yield harness.bridge.mac_rx_last.eq(i == len(frame) - 1)
        yield harness.bridge.mac_rx_error.eq(
            error if i == len(frame) - 1 else 0)
        yield Tick("rmii")
    yield harness.bridge.mac_rx_valid.eq(0)
    yield harness.bridge.mac_rx_last.eq(0)
    yield harness.bridge.mac_rx_error.eq(0)
    for _ in range(50):      # inter-frame gap
        yield Tick("rmii")


def make_input_blocks(n_blocks, seed_base):
    """n_blocks x 3 inputs x 15 frames of recognizable samples."""
    blocks = []
    for b in range(n_blocks):
        per_input = []
        for i in range(3):
            frames = []
            for s in range(15):
                samples = [(0xA0 + seed_base + b + i) << 16
                           | (s << 8) | (i + 1) for _ in range(8)]
                frames.append((samples, (b + i) & 0xF))
            per_input.append(frames)
        blocks.append(per_input)
    return blocks


class TestBridgeTXDirection(unittest.TestCase):
    def test_tx_frames_match_reference_layout(self):
        """Three inputs' frames must come out as reference-conformant
        TX-direction Ethernet frames (header + input/sample-major slots),
        with sample_cntr advancing by 15 per frame. The driver
        interleaves the three inputs frame-by-frame (they stream
        concurrently, like three live ADAT sources); the last two
        captured blocks are asserted completely — earlier blocks may
        still contain silence for inputs whose frames were not yet
        delivered when the master input's ring wrapped (PROTOCOL.md §6
        sends unsynced inputs as silence)."""
        harness = BridgeHarness()

        n_blocks = 5
        block_frames = make_input_blocks(n_blocks, 0)

        captured = []

        def adat_driver():
            for b in range(n_blocks):
                for s in range(15):
                    for i in range(3):
                        yield from commit_frames(
                            harness.adat_in_tx[i],
                            [block_frames[b][i][s]])
            for _ in range(120_000):
                yield Tick("sync")

        def tx_capture():
            yield harness.bridge.mac_tx_ready.eq(1)
            frame = bytearray()
            for _ in range(250_000):
                yield Tick("rmii")
                if (yield harness.bridge.mac_tx_valid):
                    frame.append((yield harness.bridge.mac_tx_data))
                    if (yield harness.bridge.mac_tx_last):
                        captured.append(bytes(frame))
                        frame = bytearray()
                if len(captured) >= n_blocks:
                    return

        run_harness(harness, [(adat_driver, "sync"),
                              (tx_capture, "rmii")])

        self.assertEqual(len(captured), n_blocks)
        for b, frame in enumerate(captured):
            self.assertEqual(frame[0:6], b"\xff" * 6,
                             "no PC seen: frames must be broadcast")
            self.assertEqual(frame[6:12], int_to_mac(OWN_MAC))
            self.assertEqual(frame[12:14], b"\x88\xb5")
            hdr = ref.parse_audio_header(frame[14:22])
            self.assertIsNotNone(hdr)
            self.assertEqual(hdr["direction"], 0)
            self.assertEqual(hdr["payload_len"], ref.PAYLOAD_LEN_TX)
            self.assertEqual(hdr["sample_cntr"], b * 15)
        for b in (n_blocks - 2, n_blocks - 1):
            frame = captured[b]
            expect = bytearray()
            for s in range(15):
                for i in range(3):
                    samples, user = block_frames[b][i][s]
                    expect += ref.adat_encode_frame(samples, user)
            self.assertEqual(frame[22:], bytes(expect),
                             f"block {b}: payload slots differ")

    def test_tx_learns_pc_mac_from_audio(self):
        """After an RX-direction frame from the PC, TX frames are
        unicast to the learned MAC."""
        harness = BridgeHarness()
        captured = []

        def adat_driver():
            frames = []
            for s in range(15 * 5):
                samples = [0x1000 + s for _ in range(8)]
                frames.append((samples, 0))
            yield from commit_frames(harness.adat_in_tx[0], frames)
            for _ in range(120_000):
                yield Tick("sync")

        def pc_player():
            for _ in range(30_000):
                yield Tick("rmii")
            slots = [ref.EMPTY_USER_FRAME] * 30
            frame = ref.build_audio_frame(1, 0, slots, int_to_mac(PC_MAC),
                                          int_to_mac(OWN_MAC), fcs=False)
            yield from play_rx_frame(harness, frame)
            for _ in range(3):
                for _ in range(30_000):
                    yield Tick("rmii")
                yield from play_rx_frame(harness, frame)

        def tx_capture():
            yield harness.bridge.mac_tx_ready.eq(1)
            frame = bytearray()
            for _ in range(250_000):
                yield Tick("rmii")
                if (yield harness.bridge.mac_tx_valid):
                    frame.append((yield harness.bridge.mac_tx_data))
                    if (yield harness.bridge.mac_tx_last):
                        captured.append(bytes(frame))
                        frame = bytearray()
                if len(captured) >= 3:
                    return

        run_harness(harness, [(adat_driver, "sync"),
                              (pc_player, "rmii"),
                              (tx_capture, "rmii")])

        self.assertEqual(len(captured), 3)
        self.assertEqual(captured[0][0:6], b"\xff" * 6, "not yet learned")
        self.assertEqual(captured[1][0:6], int_to_mac(PC_MAC))
        self.assertEqual(captured[2][0:6], int_to_mac(PC_MAC))


class TestBridgeRXDirection(unittest.TestCase):
    def test_rx_frames_reach_adat_outputs(self):
        """RX-direction frames' slots must be fed to the two outputs,
        channel 0..7 in order, paced by the transmitters."""
        harness = BridgeHarness()

        blocks = []
        for b in range(2):
            per_out = []
            for o in range(2):
                frames = []
                for s in range(15):
                    samples = [(0x30 + b) << 16 | (o << 12) | (s << 4) + 7
                               for _ in range(8)]
                    frames.append(samples)
                per_out.append(frames)
            blocks.append(per_out)

        slots_bytes = []
        for b in range(2):
            slots = [ref.adat_encode_frame(blocks[b][o][s], 0)
                     for s in range(15) for o in range(2)]
            slots_bytes.append(slots)

        got_commits = {0: [], 1: []}   # per output: (addr, sample) in order
        done = []

        def rx_player():
            for _ in range(20_000):
                yield Tick("rmii")
            for b in range(2):
                frame = ref.build_audio_frame(1, b * 15, slots_bytes[b],
                                              int_to_mac(PC_MAC),
                                              int_to_mac(OWN_MAC), fcs=False)
                yield from play_rx_frame(harness, frame)
            for _ in range(150_000):
                yield Tick("rmii")

        def out_collector(o):
            for _ in range(300_000):
                yield Tick("sync")
                if (yield harness.bridge.feeder.valid_out[o]) & \
                        (yield harness.bridge.feeder.ready_in[o]):
                    # the feeder repeats the newest buffer until a fresh
                    # one arrives (PROTOCOL.md §7), so commits keep
                    # flowing after both buffers were delivered once —
                    # only the first two buffers' commits are checked
                    if len(got_commits[o]) < 240:
                        addr = (yield harness.bridge.feeder.addr_out[o])
                        sample = (yield harness.bridge.feeder.sample_out[o])
                        got_commits[o].append((addr, sample))
                if len(got_commits[0]) >= 240 and len(got_commits[1]) >= 240:
                    done.append(1)
                    return

        def collector0():
            yield from out_collector(0)

        def collector1():
            yield from out_collector(1)

        run_harness(harness, [(rx_player, "rmii"),
                              (collector0, "sync"),
                              (collector1, "sync")])

        for o in range(2):
            commits = got_commits[o]
            self.assertEqual(len(commits), 240,
                             f"output {o}: expected 2 buffers x 15 x 8 commits")
            for buf in range(2):
                for s in range(15):
                    for ch in range(8):
                        addr, sample = commits[buf * 120 + s * 8 + ch]
                        self.assertEqual(addr, ch,
                                         f"out {o} buf {buf} sample {s}")
                        self.assertEqual(
                            sample, blocks[buf][o][s][ch],
                            f"out {o} buf {buf} sample {s} ch {ch}")

    def test_rx_rejects_bad_frames(self):
        """Bad FCS, wrong magic and wrong payload_len frames must be
        dropped (no output commits)."""
        harness = BridgeHarness()

        good_slots = [ref.EMPTY_USER_FRAME] * 30
        good = ref.build_audio_frame(1, 0, good_slots, int_to_mac(PC_MAC),
                                     int_to_mac(OWN_MAC), fcs=False)

        bad_fcs = bytearray(good)
        bad_fcs[-1] ^= 0x55
        wrong_magic = bytearray(good)
        wrong_magic[14] = 0x5A
        wrong_len = bytearray(good)
        wrong_len[20] = 0x00
        wrong_len[21] = 0x05

        n_commits = [0]

        def rx_player():
            for _ in range(20_000):
                yield Tick("rmii")
            yield from play_rx_frame(harness, bytes(bad_fcs), error=1)
            yield from play_rx_frame(harness, bytes(wrong_magic))
            yield from play_rx_frame(harness, bytes(wrong_len))
            for _ in range(80_000):
                yield Tick("rmii")

        def out_collector(o):
            for _ in range(120_000):
                yield Tick("sync")
                if (yield harness.bridge.feeder.valid_out[o]) & \
                        (yield harness.bridge.feeder.ready_in[o]):
                    n_commits[0] += 1

        def collector0():
            yield from out_collector(0)

        def collector1():
            yield from out_collector(1)

        run_harness(harness, [(rx_player, "rmii"),
                              (collector0, "sync"),
                              (collector1, "sync")])

        self.assertEqual(n_commits[0], 0,
                         "bad frames must not reach the outputs")

    def test_rx_counts_cntr_gaps(self):
        """A skipped sample_cntr must be counted as a gap."""
        harness = BridgeHarness()
        good_slots = [ref.EMPTY_USER_FRAME] * 30
        gaps_seen = []

        def rx_player():
            for _ in range(20_000):
                yield Tick("rmii")
            for cntr in (0, 15, 60):     # 30..44 missing
                frame = ref.build_audio_frame(1, cntr, good_slots,
                                              int_to_mac(PC_MAC),
                                              int_to_mac(OWN_MAC), fcs=False)
                yield from play_rx_frame(harness, frame)
            for _ in range(60_000):
                yield Tick("rmii")

        def gap_watcher():
            for _ in range(150_000):
                yield Tick("rmii")
            gaps_seen.append((yield harness.bridge.stats_rx_gaps))

        run_harness(harness, [(rx_player, "rmii"),
                              (gap_watcher, "rmii")])

        self.assertGreater(gaps_seen[0], 0, "the 30..44 gap was not counted")


class TestBridgeFreeRun(unittest.TestCase):
    def test_free_run_silent_blocks(self):
        """With no ADAT input synced, the bridge transmits silent
        EMPTY_USER_FRAME blocks at exactly 3200/s (30720 fast cycles),
        and sample_cntr advances per sent frame."""
        harness = BridgeHarness()
        self.assertTrue(harness.bridge.free_run is not None)

        captured = []
        last_capture_time = []

        def tx_capture():
            yield harness.bridge.mac_tx_ready.eq(1)
            frame = bytearray()
            cycles_since_last = 0
            prev_rmii_cycle = 0
            now = 0
            for _ in range(70_000):
                yield Tick("rmii")
                now += 1
                if (yield harness.bridge.mac_tx_valid):
                    frame.append((yield harness.bridge.mac_tx_data))
                    if (yield harness.bridge.mac_tx_last):
                        captured.append((bytes(frame), now))
                        frame = bytearray()
                if len(captured) >= 3:
                    return

        def free_run_watcher():
            for _ in range(70_000):
                yield Tick("rmii")
            # never asserted below, collected via captured

        run_harness(harness, [(tx_capture, "rmii"),
                              (free_run_watcher, "rmii")])

        self.assertEqual(len(captured), 3)
        for b, (frame, _t) in enumerate(captured):
            hdr = ref.parse_audio_header(frame[14:22])
            self.assertIsNotNone(hdr)
            self.assertEqual(hdr["sample_cntr"], b * 15)
            # payload: 45 EMPTY_USER_FRAME slots
            audio = frame[22:]
            self.assertEqual(len(audio), 1440)
            for slot in range(45):
                self.assertEqual(audio[slot * 32:(slot + 1) * 32],
                                 ref.EMPTY_USER_FRAME)
        # block pace: ~15625 rmii cycles between frames
        for b in range(1, 3):
            delta = captured[b][1] - captured[b - 1][1]
            self.assertAlmostEqual(delta, 15625, delta=40,
                                   msg=f"block pace {delta}")


class TestBridgeFullLoop(unittest.TestCase):
    def test_adat_in_to_ethernet_to_adat_out(self):
        """Full round trip: one block of audio in through input 0,
        captured as a TX-direction frame, played back as an
        RX-direction frame (PC side), and committed to output 0 —
        the samples must arrive intact and in order."""
        harness = BridgeHarness()
        rng = random.Random(0x10CA1)

        in_frames = []
        for s in range(15):
            samples = [(0x5EED << 8) | s for _ in range(8)]
            in_frames.append((samples, 0b1010))
        captured_tx = []

        def adat_driver():
            yield from commit_frames(harness.adat_in_tx[0], in_frames)
            for _ in range(120_000):
                yield Tick("sync")

        def loop():
            # catch one TX frame (like the PC NIC would), then play it
            # back as an RX-direction frame
            yield harness.bridge.mac_tx_ready.eq(1)
            frame = bytearray()
            for _ in range(120_000):
                yield Tick("rmii")
                if (yield harness.bridge.mac_tx_valid):
                    frame.append((yield harness.bridge.mac_tx_data))
                    if (yield harness.bridge.mac_tx_last):
                        break
            self.assertEqual(len(frame), 14 + 8 + 1440)

            # PC side: decode slots (input 0 only is real), re-encode as
            # an RX-direction frame for outputs 0/1 (loop input 0 back)
            slots = []
            for s in range(15):
                for o in range(2):
                    slot = frame[22 + (s * 3 + o) * 32:
                                 22 + (s * 3 + o) * 32 + 32]
                    slots.append(slot)
            rx_frame = ref.build_audio_frame(1, 0, slots,
                                             int_to_mac(PC_MAC),
                                             int_to_mac(OWN_MAC),
                                             fcs=False)
            captured_tx.append(rx_frame)
            yield from play_rx_frame(harness, rx_frame)
            for _ in range(150_000):
                yield Tick("rmii")

        got = {0: [], 1: []}

        def out_collector(o):
            for _ in range(250_000):
                yield Tick("sync")
                if (yield harness.bridge.feeder.valid_out[o]) & \
                        (yield harness.bridge.feeder.ready_in[o]):
                    # the feeder repeats the newest buffer until a fresh
                    # one arrives (PROTOCOL.md §7) — cap at the first
                    # buffer's 120 commits
                    if len(got[o]) < 120:
                        addr = (yield harness.bridge.feeder.addr_out[o])
                        sample = (yield harness.bridge.feeder.sample_out[o])
                        user = (yield harness.bridge.feeder.user_out[o])
                        last = (yield harness.bridge.feeder.last_out[o])
                        got[o].append((addr, sample, user, last))
                    if len(got[0]) >= 120 and len(got[1]) >= 120:
                        return

        def collector0():
            yield from out_collector(0)

        def collector1():
            yield from out_collector(1)

        run_harness(harness, [(adat_driver, "sync"),
                              (loop, "rmii"),
                              (collector0, "sync"),
                              (collector1, "sync")])

        self.assertEqual(len(captured_tx), 1)
        # the loop maps TX slots (s*3+o) = inputs 0/1 onto RX slots
        # (s*2+o) = outputs 0/1: output 0 carries input 0's audio,
        # output 1 carries input 1 (undriven = silence)
        for o in range(2):
            commits = got[o]
            self.assertEqual(len(commits), 120, f"output {o}")
            for s in range(15):
                for ch in range(8):
                    addr, sample, user, last = commits[s * 8 + ch]
                    self.assertEqual(addr, ch)
                    want_user = 0b1010 if o == 0 else 0
                    self.assertEqual(user, want_user)
                    if ch == 7:
                        self.assertTrue(last)
                    self.assertEqual(
                        sample,
                        (0x5EED << 8) | s if o == 0 else 0,
                        f"out {o} sample {s} ch {ch}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
