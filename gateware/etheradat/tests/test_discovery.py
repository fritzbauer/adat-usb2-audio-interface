"""Tests for the discovery module (hello frames, MAC learning,
keepalive) against the reference model.

Run:  python -m etheradat.tests.test_discovery
"""

import os
import unittest

from amaranth import *
from amaranth.sim import Simulator, Tick

from etheradat import reference as ref
from etheradat.discovery import Discovery, HELLO_FRAME_BYTES

RMII_CLK = 50_000_000
OWN_MAC = 0x02AD6E657401


def mac_str(v):
    return ":".join(f"{(v >> s) & 0xFF:02x}" for s in (40, 32, 24, 16, 8, 0))


class TestDiscovery(unittest.TestCase):
    def run_discovery(self, cycles, stimulus, timers):
        """Run the Discovery module; stimulus(drut) drives the RX-side
        inputs as a generator. Returns (hellos, state_samples)."""
        dut = Discovery(own_mac=OWN_MAC, device_serial=0xDEADBEef,
                        session_nonce=0x1234,
                        hello_discover_cycles=timers["discover"],
                        hello_linked_cycles=timers["linked"],
                        keepalive_cycles=timers["keepalive"])
        hellos = []

        def hello_collector():
            # ready tied high: the module consumes one byte per cycle;
            # sample what is presented each cycle
            yield dut.hello_ready.eq(1)
            frame = bytearray()
            for _ in range(cycles):
                yield Tick("sync")
                if (yield dut.hello_valid):
                    frame.append((yield dut.hello_data))
                    if (yield dut.hello_last):
                        hellos.append(bytes(frame))
                        frame = bytearray()

        sim = Simulator(dut)
        sim.add_clock(1 / RMII_CLK)
        sim.add_sync_process(hello_collector)
        if stimulus is not None:
            sim.add_sync_process(stimulus)
        if os.getenv("GENERATE_VCDS"):
            with sim.write_vcd("etheradat_discovery.vcd"):
                sim.run()
        else:
            sim.run()
        return hellos

    def test_hello_frame_matches_reference(self):
        """The emitted hello frame must match the reference builder."""
        timers = {"discover": 500, "linked": 1000, "keepalive": 10_000}
        hellos = self.run_discovery(3000, None, timers)
        self.assertGreaterEqual(len(hellos), 2)
        src = OWN_MAC.to_bytes(6, "big")
        expected = ref.build_hello_frame(0xDEADBEef, 0x1234, src)
        # our module emits the unpadded 34 bytes; the reference pads
        for hello in hellos:
            self.assertEqual(len(hello), 34)
            self.assertEqual(hello, expected[:34],
                             "hello frame differs from the reference")

    def test_discovery_learn_and_keepalive(self):
        """PC hello teaches the MAC (unicast hello afterwards), audio
        frames keep the link alive, silence times it out."""
        timers = {"discover": 400, "linked": 2000, "keepalive": 3000}
        dut = Discovery(own_mac=OWN_MAC, device_serial=1, session_nonce=2,
                        hello_discover_cycles=timers["discover"],
                        hello_linked_cycles=timers["linked"],
                        keepalive_cycles=timers["keepalive"])

        events = []

        def stimulus():
            pc_mac = 0x112233445566
            # let one broadcast hello go out first
            for _ in range(600):
                yield Tick("sync")
            # a valid PC hello arrives
            yield dut.rx_frame_strobe.eq(1)
            yield dut.rx_is_hello_in.eq(1)
            yield dut.rx_hello_valid_in.eq(1)
            yield dut.rx_role_in.eq(2)
            yield dut.rx_src_mac_in.eq(pc_mac)
            yield Tick("sync")
            yield dut.rx_frame_strobe.eq(0)
            yield dut.rx_is_hello_in.eq(0)
            events.append(("learned", (yield dut.linked_out)))
            for _ in range(4):
                yield Tick("sync")
            events.append(("linked after hello", (yield dut.linked_out),
                           mac_str((yield dut.pc_mac_out))))
            # audio frames (any src while unlinked... now linked: the
            # keepalive is refreshed by them)
            for _ in range(2000):
                yield Tick("sync")
                if _ % 900 == 0:
                    yield dut.rx_frame_strobe.eq(1)
                    yield dut.rx_is_audio_in.eq(1)
                    yield dut.rx_src_mac_in.eq(pc_mac)
                    yield Tick("sync")
                    yield dut.rx_frame_strobe.eq(0)
                    yield dut.rx_is_audio_in.eq(0)
            events.append(("linked after audio keepalive",
                           (yield dut.linked_out)))
            # silence: the keepalive must time out
            for _ in range(3200):
                yield Tick("sync")
            events.append(("linked after silence", (yield dut.linked_out)))
            # back to the fast discovery rate: two hellos within a
            # short window
            hellos = 0
            for _ in range(1000):
                yield Tick("sync")
                if (yield dut.hello_valid):
                    hellos += 1
                    for _ in range(40):
                        yield Tick("sync")
            events.append(("hello after timeout", hellos))

        def hello_collector():
            yield dut.hello_ready.eq(1)
            for _ in range(12_000):
                yield Tick("sync")
                yield

        sim = Simulator(dut)
        sim.add_clock(1 / RMII_CLK)
        sim.add_sync_process(stimulus)
        sim.add_sync_process(hello_collector)
        if os.getenv("GENERATE_VCDS"):
            with sim.write_vcd("etheradat_discovery2.vcd"):
                sim.run()
        else:
            sim.run()

        self.assertEqual(events[0], ("learned", 0))  # registered, next cycle
        self.assertEqual(events[1][1], 1)
        self.assertEqual(events[1][2], mac_str(0x112233445566))
        self.assertEqual(events[2][1], 1)   # audio kept it alive
        self.assertEqual(events[3][1], 0)   # timed out
        self.assertGreaterEqual(events[4][1], 2)  # 4 Hz rate again

    def test_invalid_hello_does_not_teach(self):
        """A hello with a bad version/role must not set the link."""
        timers = {"discover": 300, "linked": 600, "keepalive": 2000}
        dut = Discovery(own_mac=OWN_MAC, device_serial=1, session_nonce=2,
                        hello_discover_cycles=timers["discover"],
                        hello_linked_cycles=timers["linked"],
                        keepalive_cycles=timers["keepalive"])

        def stimulus():
            for _ in range(100):
                yield Tick("sync")
            yield dut.rx_frame_strobe.eq(1)
            yield dut.rx_is_hello_in.eq(1)
            yield dut.rx_hello_valid_in.eq(0)     # parser said: invalid
            yield dut.rx_role_in.eq(2)
            yield dut.rx_src_mac_in.eq(0xAABBCCDDEEFF)
            yield Tick("sync")
            yield dut.rx_frame_strobe.eq(0)
            for _ in range(50):
                yield Tick("sync")
            self.assertEqual((yield dut.linked_out), 0)

        def hello_collector():
            yield dut.hello_ready.eq(1)
            for _ in range(600):
                yield Tick("sync")
                yield

        sim = Simulator(dut)
        sim.add_clock(1 / RMII_CLK)
        sim.add_sync_process(stimulus)
        sim.add_sync_process(hello_collector)
        if os.getenv("GENERATE_VCDS"):
            with sim.write_vcd("etheradat_discovery3.vcd"):
                sim.run()
        else:
            sim.run()


if __name__ == "__main__":
    unittest.main(verbosity=2)
