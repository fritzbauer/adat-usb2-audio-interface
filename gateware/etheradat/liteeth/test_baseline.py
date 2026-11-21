#!/usr/bin/env python3
"""End-to-end migen sim of the Baseline design (gen_baseline.py): the
stock LiteEth IP/UDP stack around the RMII shim. Hand-plays an Ethernet
frame as RMII dibits into crs_dv/rxd and decodes whatever the design
transmits on txd/tx_en.

Test 1: ARP request  -> expect an ARP reply from 02:AD:6E:65:74:01.
Test 2: UDP datagram -> port 1234 -> expect the UDP echo.

Run with the scratch venv (migen/litex/liteeth):
  /tmp/etheradat-scratch/liteenv/bin/python test_baseline.py

Versions (21.11.2026): migen @ e19524c963a8, litex @ 6438a25a060b,
liteeth @ a54c730c323f (git master, needed for native `be` payload
fields — PyPI's liteeth 2024.12 still has last_be and breaks the shim).
migen master needs gateware/patches/migen-e19524c-sim-write-only-port.patch:
its simulator crashes on the write-only memory ports of its own SyncFIFO.
"""

import os
import struct
import sys
import unittest

from migen import *
from migen.sim.core import Simulator

from liteeth.common import *

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_baseline import Baseline, UDP_ECHO_PORT


def eth_fcs(frame):
    crc = 0xFFFFFFFF
    for b in frame:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1))
    return struct.pack("<I", crc ^ 0xFFFFFFFF)


PREAMBLE = bytes(7*[0x55]) + bytes([0xD5])   # 7x 0x55 + SFD
PREAMBLE_LEN = len(PREAMBLE)


def arp_request(who="192.168.42.2", tell="192.168.42.1"):
    frame = (
        bytes.fromhex("ffffffffffff") + bytes.fromhex("f8e43b907775") +
        struct.pack(">H", 0x0806) +
        struct.pack(">HHBBH", 1, 0x0800, 6, 4, 1) +       # hw, proto, hlen, plen, request
        bytes.fromhex("f8e43b907775") +
        bytes(map(int, tell.split("."))) +
        bytes(6) +                                        # target MAC (empty)
        bytes(map(int, who.split(".")))
    )
    frame += b"\x00" * max(0, 60 - len(frame))
    return frame + eth_fcs(frame)


def arp_reply():
    """PC answers an earlier ARP request from the design:
    192.168.42.1 is-at f8e43b907775. This primes the design's ARP table
    so the UDP echo can send its reply without a fresh ARP exchange
    (the echo goes to the datagram's SOURCE address, 192.168.42.1)."""
    frame = (
        bytes.fromhex("02ad6e657401") + bytes.fromhex("f8e43b907775") +
        struct.pack(">H", 0x0806) +
        struct.pack(">HHBBH", 1, 0x0800, 6, 4, 2) +       # reply
        bytes.fromhex("f8e43b907775") +                   # sender MAC = PC
        bytes(map(int, "192.168.42.1".split("."))) +      # sender IP = PC
        bytes.fromhex("02ad6e657401") +                   # target MAC = design
        bytes(map(int, "192.168.42.2".split(".")))        # target IP = design
    )
    frame += b"\x00" * max(0, 60 - len(frame))
    return frame + eth_fcs(frame)


def udp_packet(src_ip="192.168.42.1", dst_ip="192.168.42.2", dst_port=1234,
               payload=b"echo-test"):
    udp_len = 8 + len(payload)
    udp = struct.pack(">HHHH", 7, dst_port, udp_len, 0) + payload
    # UDP checksum over pseudo header
    ph = (bytes(map(int, src_ip.split("."))) +
          bytes(map(int, dst_ip.split("."))) +
          struct.pack(">BBH", 0, 17, udp_len))
    data = udp + (b"\x00" if len(udp) % 2 else b"")
    s = 0
    for i in range(0, len(data), 2):
        s += (data[i] << 8) | data[i + 1]
    for i in range(0, len(ph), 2):
        s += (ph[i] << 8) | ph[i + 1]
    while s >> 16:
        s = (s & 0xFFFF) + (s >> 16)
    ck = ~s & 0xFFFF or 0xFFFF
    # header is 8 bytes (0..7), checksum replaces bytes 6..7, payload starts at 8
    udp = udp[:6] + struct.pack(">H", ck) + udp[8:]
    total = 20 + udp_len
    ip = struct.pack(">BBHHHBBH", 0x45, 0, total, 0, 0, 64, 17, 0) + \
        bytes(map(int, src_ip.split("."))) + \
        bytes(map(int, dst_ip.split(".")))
    # IP header checksum
    s = 0
    for i in range(0, 20, 2):
        s += (ip[i] << 8) | ip[i + 1]
    while s >> 16:
        s = (s & 0xFFFF) + (s >> 16)
    ip = ip[:10] + struct.pack(">H", ~s & 0xFFFF) + ip[12:]

    frame = (bytes.fromhex("02ad6e657401") +
             bytes.fromhex("f8e43b907775") +
             struct.pack(">H", 0x0800) + ip + udp)
    frame += b"\x00" * max(0, 60 - len(frame))
    return frame + eth_fcs(frame)


class TestBaselineEndToEnd(unittest.TestCase):
    def play_frame(self, frame):
        """Feed one complete MAC frame as RMII dibits into crs_dv/rxd."""
        dibits = []
        for b in frame:
            for pair in range(4):
                dibits.append((b >> (2 * pair)) & 0x3)
        yield self.dut.rmii_crs_dv.eq(0)
        yield self.dut.rmii_rxd.eq(0)
        yield
        for d in dibits:
            yield self.dut.rmii_crs_dv.eq(1)
            yield self.dut.rmii_rxd.eq(d)
            yield
        # end of frame: two idle cycles
        yield self.dut.rmii_crs_dv.eq(0)
        yield self.dut.rmii_rxd.eq(0)
        yield

    def run_sim(self, frames, settle=6000, gap=1200):
        frames = [PREAMBLE + f for f in frames]
        self.dut = Baseline()
        rx_frames = []
        got = {"tx_en": 0}
        sent = bytearray()

        def rx_player():
            yield self.dut.rmii_crs_dv.eq(0)
            yield self.dut.rmii_rxd.eq(0)
            yield
            yield
            for frame in frames:
                yield from self.play_frame(frame)
                for _ in range(gap):
                    yield

        def tx_logger():
            prev_en = 0
            for _ in range(settle + 40 * 100):
                en = yield self.dut.rmii_tx_en
                d = yield self.dut.rmii_txd
                if en:
                    if not prev_en:
                        sent.clear()
                    sent.append(d)   # one dibit per cycle, LSB pair first
                    rx_frames.append(None)  # marker: saw TX activity
                prev_en = en
                yield

        sim = Simulator(self.dut, {"sys": [rx_player(), tx_logger()]},
                        clocks={"sys": 20, "eth_tx": 20, "eth_rx": 20})
        with sim:
            sim.run()
        return sent

    def decode_dibits(self, dibits, skip=PREAMBLE_LEN):
        """RMII dibit stream -> bytes (pair 0 = LSB pair of the byte),
        skipping the preamble+SFD the design prepends on TX."""
        out = bytearray()
        dibits = dibits[skip * 4:]
        for i in range(0, len(dibits) - 3, 4):
            b = 0
            for pair in range(4):
                b |= dibits[i + pair] << (2 * pair)
            out.append(b)
        return bytes(out)

    def test_arp_reply(self):
        sent = self.run_sim([arp_request()], settle=20000)
        self.assertTrue(sent, "design never transmitted anything")
        frame = self.decode_dibits(list(sent))
        # strip padding, find reply content
        self.assertEqual(frame[6:12], bytes.fromhex("02ad6e657401"),
                         f"sender MAC wrong: {frame[:20].hex()}")
        self.assertEqual(frame[12:14], b"\x08\x06",
                         f"ethertype wrong: {frame[:14].hex()}")
        self.assertEqual(frame[14:20],
                         bytes.fromhex("0001") + b"\x08\x00\x06\x04",
                         f"ARP header wrong: {frame[10:26].hex()}")
        self.assertEqual(frame[20:22], b"\x00\x02",
                         f"ARP opcode wrong: {frame[18:26].hex()}")
        self.assertEqual(frame[22:28], bytes.fromhex("02ad6e657401"),
                         f"sender MAC wrong: {frame[18:32].hex()}")
        self.assertEqual(frame[32:38], bytes.fromhex("f8e43b907775"),
                         f"target MAC wrong: {frame[28:42].hex()}")

    def test_udp_echo(self):
        # The design's ARP table only accepts replies while an ARP
        # request of its own is pending, so an unsolicited reply is
        # ignored: (1) try to prime (ignored), (2) send the datagram,
        # which makes the design ARP for 192.168.42.1, (3) answer that
        # request, (4) repeat the datagram -> the echo can now go out.
        # sent holds only the LAST TX frame (cleared at every TX start).
        sent = self.run_sim(
            [arp_reply(), udp_packet(), arp_reply(), udp_packet()],
            settle=30000)
        self.assertTrue(sent, "design never transmitted anything")
        frame = self.decode_dibits(list(sent))
        payload_seen = b"echo-test" in frame
        self.assertTrue(payload_seen, f"no UDP echo in TX: {frame[:60].hex()}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
