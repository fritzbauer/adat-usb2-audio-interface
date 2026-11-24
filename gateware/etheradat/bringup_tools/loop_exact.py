#!/usr/bin/env python3
"""Bit-exact hardware loopback test (ADAT OUT -> optical cable -> ADAT IN).

Unlike `loop_sniff_all.py` / `loop_full.py`, which only measure whether *a*
tone arrives (RMS + FFT peak), this tool proves that the samples which come
back are *exactly* the samples that were sent.

It bypasses ALSA and the kernel driver completely:

- it writes AdatNet RX-direction frames (EtherType 0x88B5, dir=1) straight
  onto the NIC with AF_PACKET.  OUT2 slots carry a self-describing 24-bit
  codeword per channel::

      bits 23..16  0xA5 marker
      bits 15..12  frame index f (0..15)
      bits 11..8   sample-in-block index s (0..14)
      bits  7..5   channel ch (0..7)
      bits  4..0   0b10101 constant tail

  13 of the 24 bits are fixed/structural, so a bit error almost always
  produces an invalid codeword (and the f/s/ch progression is checked too).

- it sniffs the FPGA TX-direction frames (dir=0) and takes input0 (IN1,
  slot s*3+0, the OUT2 loopback), decoding them with the reference codec.

Every returned 32-byte slot must be byte-identical to a sent OUT2 slot and
every decoded sample must equal the codeword for its (f, s, channel).  Any
difference is reported with the offending slot and codeword fields.

Requires root (AF_PACKET).  Run from gateware/ with the project venv:

    sudo venv/bin/python etheradat/bringup_tools/loop_exact.py [seconds] [rate]

The FPGA keeps running if the send stops; it just repeats the last buffer.
"""

import os
import socket
import struct
import sys
import threading
import time

GATEWARE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, GATEWARE)

from etheradat import reference as ref            # noqa: E402

IF = os.environ.get("ADATNET_IF", "enxf8e43b907775")
PC_MAC = bytes.fromhex(os.environ.get("ADATNET_PC_MAC", "f8e43b907775"))
FPGA_MAC = bytes.fromhex(os.environ.get("ADATNET_FPGA_MAC", "02ad6e657401"))
ETHERTYPE_AUDIO = b"\x88\xb5"

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
RATE = int(sys.argv[2]) if len(sys.argv) > 2 else 3200     # blocks/s
NFRAMES = 16                    # distinct (f, s, ch) codeword frames

MARKER = 0xA5
TAIL = 0b10101


def make_sample(f, s, ch):
    """Self-describing 24-bit test sample (see module docstring)."""
    return ((MARKER & 0xFF) << 16 | (f & 0xF) << 12
            | (s & 0xF) << 8 | (ch & 0x7) << 5 | TAIL)


def parse_sample(value):
    """Return (f, s, ch) if `value` is a valid codeword else None."""
    if ((value >> 16) & 0xFF) != MARKER:
        return None
    if (value & 0x1F) != TAIL:
        return None
    return ((value >> 12) & 0xF, (value >> 8) & 0xF, (value >> 5) & 0x7)


def build_frames():
    """NFRAMES RX-direction frames; returns (frames, expected_slot_map)."""
    frames = []
    expected = {}
    for f in range(NFRAMES):
        slots = []
        for s in range(15):
            # o=0 (OUT1) stays silent; o=1 (OUT2) is looped to IN1.
            out0 = ref.EMPTY_USER_FRAME
            samples = [make_sample(f, s, ch) for ch in range(8)]
            out1 = ref.adat_encode_frame(samples, 0)
            slots += [out0, out1]
            expected[out1] = (f, s)
        frames.append(ref.build_audio_frame(
            1, f * 15, slots, PC_MAC, FPGA_MAC, fcs=False, pad=False))
    return frames, expected


class Sniffer(threading.Thread):
    """Collect input0 (IN1) 32-byte slots from FPGA TX-direction frames."""

    def __init__(self, sock):
        super().__init__(daemon=True)
        self.sock = sock
        self.stop_event = threading.Event()
        self.slots = []          # list of (sample_cntr, slot_index, slot)
        self.frames = 0
        self.outgoing = 0
        # inputs 1/2 are unsynced in the OUT2->IN1 loop and are sent as the
        # constant silence frame; counting them separates a corruption in
        # the TX payload path from one in the ADAT input path.
        self.silence_ok = 0
        self.silence_bad = 0

    def run(self):
        self.sock.settimeout(0.2)
        while not self.stop_event.is_set():
            try:
                d, a = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            if a[2] == socket.PACKET_OUTGOING:
                self.outgoing += 1
                continue
            if (len(d) != 1462 or d[12:14] != ETHERTYPE_AUDIO
                    or d[6:12] != FPGA_MAC or d[15] != 0):
                continue
            cntr = struct.unpack_from("<I", d, 16)[0]
            payload = d[22:]
            self.frames += 1
            for k in range(45):
                slot = payload[k * 32:(k + 1) * 32]
                if k % 3 == 0:           # input0 (IN1) slot s*3+0
                    self.slots.append((cntr, k // 3, slot))
                elif slot == ref.EMPTY_USER_FRAME:
                    self.silence_ok += 1
                else:
                    self.silence_bad += 1


def main():
    frames, expected = build_frames()

    s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3))
    s.bind((IF, 0))

    sniffer = Sniffer(s)
    sniffer.start()

    n = max(1, int(SECONDS * RATE))
    period = 1.0 / RATE
    constant = os.environ.get("LOOP_EXACT_CONST") == "1"
    sent = 0
    t0 = time.perf_counter()
    try:
        for i in range(n):
            target = t0 + i * period
            while True:
                now = time.perf_counter()
                if now >= target:
                    break
            s.send(frames[0 if constant else i % NFRAMES])
            sent += 1
    except (BrokenPipeError, OSError) as exc:
        print("send stopped early:", exc)
    # let the pipeline drain (about one block period)
    time.sleep(0.1)
    sniffer.stop_event.set()
    sniffer.join(timeout=1.0)

    print(f"sent {sent} RX frames ({len(frames[0])} B) at "
          f"{RATE}/s for {SECONDS:.1f}s")
    print(f"sniffed {sniffer.frames} FPGA TX frames, "
          f"{len(sniffer.slots)} input0 slots")
    print(f"  unsynced input slots (must be exact EMPTY): "
          f"{sniffer.silence_ok} ok, {sniffer.silence_bad} corrupted")

    if not sniffer.slots:
        print("NO input0 slots captured -> is the OUT2->IN1 cable in place "
              "and input0 synced (uart status0 sync0=1)?")
        return 2

    def next_key(key):
        f, s = key
        return (f, s + 1) if s < 14 else ((f + 1) % NFRAMES, 0)

    def prev_key(key):
        f, s = key
        return (f, s - 1) if s > 0 else ((f - 1) % NFRAMES, 14)

    def expected_slot(key):
        f, s = key
        return ref.adat_encode_frame(
            [make_sample(f, s, ch) for ch in range(8)], 0)

    # group the captured input0 slots per FPGA TX frame
    by_frame = {}
    for cntr, k, slot in sniffer.slots:
        by_frame.setdefault(cntr, {})[k] = slot

    exact = 0
    empty = 0
    unknown = 0
    bad_samples = 0
    seq = []
    bad_positions = {}
    shown = 0
    for cntr in sorted(by_frame):
        slots = by_frame[cntr]
        # infer the expected (f,s) of every slot from exact neighbours
        keys = {}
        for k in range(15):
            if k in slots and slots[k] in expected:
                keys[k] = expected[slots[k]]
        for k in range(15):                 # forward fill
            if k not in keys and (k - 1) in keys:
                keys[k] = next_key(keys[k - 1])
        for k in range(14, -1, -1):         # backward fill
            if k not in keys and (k + 1) in keys:
                keys[k] = prev_key(keys[k + 1])

        for k in range(15):
            slot = slots.get(k)
            if slot is None:
                continue
            if slot in expected:
                exact += 1
                seq.append(expected[slot])
                continue
            if slot == ref.EMPTY_USER_FRAME:
                empty += 1
                continue
            unknown += 1
            bad_positions[k] = bad_positions.get(k, 0) + 1
            key = keys.get(k)
            samples, user = ref.adat_decode_frame(slot)
            if key is not None:
                want_slot = expected_slot(key)
                xor = bytes(a ^ b for a, b in zip(slot, want_slot))
                want_samples = [make_sample(key[0], key[1], ch)
                                for ch in range(8)]
                for ch, v in enumerate(samples):
                    if v != want_samples[ch]:
                        bad_samples += 1
                if shown < 6:
                    shown += 1
                    print(f"  BAD frame={cntr} slot={k} expected f,s={key}")
                    print(f"    got  {slot.hex(' ')}")
                    print(f"    want {want_slot.hex(' ')}")
                    print(f"    xor  {xor.hex(' ')}")
                    print("    samples got  " +
                          " ".join(f"{v:06x}" for v in samples))
                    print("    samples want " +
                          " ".join(f"{v:06x}" for v in want_samples))
            else:
                for v in samples:
                    if parse_sample(v) is None:
                        bad_samples += 1
                if shown < 6:
                    shown += 1
                    print(f"  BAD frame={cntr} slot={k} (no reference)")
                    print(f"    got  {slot.hex(' ')}")

    # dump the first frame that contains a bad slot (for offline analysis)
    dump_path = os.environ.get("LOOP_EXACT_DUMP", "/tmp/loop_exact_bad.txt")
    if unknown and dump_path:
        for cntr in sorted(by_frame):
            slots = by_frame[cntr]
            if any(s not in expected and s != ref.EMPTY_USER_FRAME
                   for s in slots.values() if s is not None):
                with open(dump_path, "w") as fh:
                    for k in range(15):
                        sl = slots.get(k)
                        if sl is not None:
                            # raw TX-frame slot index + bytes; map to the
                            # sent stream offline (the send pattern cycles
                            # through NFRAMES frames)
                            fh.write(f"{k} {sl.hex()}\n")
                print(f"  bad frame {cntr} dumped to {dump_path}")
                break

    total = len(sniffer.slots)
    print(f"  exact sent slots : {exact}/{total}")
    print(f"  EMPTY (silence)  : {empty}")
    print(f"  not byte-exact   : {unknown}")
    print(f"  bad samples      : {bad_samples}")
    if bad_positions:
        print("  bad slot position within the 15-slot block:",
              dict(sorted(bad_positions.items())))

    if seq:
        deltas = [(b[0] * 15 + b[1]) - (a[0] * 15 + a[1])
                  for a, b in zip(seq, seq[1:])]
        nz = [d for d in deltas if d != 0]
        print(f"  sequence: {len(seq)} slots, "
              f"{sum(1 for d in deltas if d == 0)} repeats, "
              f"{sum(1 for d in nz if d in (1, 1 - NFRAMES * 15))} +1 steps")
        print(f"  first/last (f,s): {seq[0]} .. {seq[-1]}")

    ok = unknown == 0 and bad_samples == 0
    print("RESULT:", "BIT-EXACT" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
