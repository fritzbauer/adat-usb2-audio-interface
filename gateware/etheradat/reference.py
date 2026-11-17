"""Golden reference model of the AdatNet wire protocol (testbench use).

Ported from the EtherADAT project (role-1 side):

- PROTOCOL.md v1.0-draft: frame layout (§2), 8-byte header (§3),
  payload layouts (§4.1/§4.2), ADAT bitstream reference (§5) and the
  hello frame (§12.1).
- virtual_soundcard/test_veth/rp2350_sim.py: EMPTY_USER_FRAME constant,
  adat_encode_frame() bit placement and the TX/RX frame builders. The
  semantics of this file mirror that simulator, which is the
  PC-kernel-module-verified behavioral reference.

The FPGA gateware in this directory must be bit-exact against the
functions in this file.
"""

import struct
import zlib

# Protocol constants (PROTOCOL.md §2/§3/§4).
ETHERTYPE_AUDIO = 0x88B5                 # audio frames
ETHERTYPE_HELLO = 0x88B6                 # hello/discovery frames
MAGIC = 0xA5                             # header magic
N_SAMPLES = 15                           # samples per frame, both directions
VER_FLAGS_TX = 0x00                      # dir = TX (device -> PC), raw mode
VER_FLAGS_RX = 0x08                      # dir = RX (PC -> device), raw mode

AUDIO_TX_PAYLOAD = 15 * 3 * 32           # 1440
AUDIO_RX_PAYLOAD = 15 * 2 * 32           # 960
HEADER_LEN = 8
PAYLOAD_LEN_TX = HEADER_LEN + AUDIO_TX_PAYLOAD   # 1448
PAYLOAD_LEN_RX = HEADER_LEN + AUDIO_RX_PAYLOAD   # 968

HELLO_LEN = 20                           # hello payload length (§12.1)
HELLO_ROLE_DEVICE = 1
HELLO_ROLE_PC = 2

BROADCAST_MAC = b"\xff" * 6

# Empty ADAT frame (all inputs silent): 10 sync '0' + user-block '1's +
# 48 channel-marker '1' bits, all 192 data bits zero. Verbatim from
# rp2350_sim.py — a real ADAT receiver sees exactly this on a silent
# channel, and all-zero-except-byte1 would be a protocol violation.
EMPTY_USER_FRAME = bytes.fromhex(
    "0021842108421084210842108421084210842108421084210842108421084210")

FRAME_BYTES = 32                          # one 256-bit ADAT frame in bytes


def adat_encode_frame(samples, user=0):
    """One 32-byte ADAT frame from 8 24-bit samples + 4 user bits (§5).

    Bit layout (cyclic; byte 0 starts at the sync trough):
      bits 0..9    sync, all '0'
      bit  10      '1' (start of user block)
      bits 11..14  user bits
      bit  15      '1' (end of user block)
      per channel c (0..7), 6 nibble groups n (0..5):
        bit  16+30c+5n      '1' (marker)
        bits +1..+4         sample bits 23-4n .. 20-4n (MSB first)

    The user value convention follows the PC-side adat-codec crate
    (etheradat/virtual_soundcard/rust_encoder/src/frame.rs):
    user = (frame[1] >> 1) & 0xF, i.e. user value bit k sits at frame
    bit 14-k.
    """
    assert len(samples) == 8
    f = bytearray(32)
    f[1] = 0x21                                   # sync + user block markers
    for c in range(8):
        s = samples[c] & 0xFFFFFF
        for n in range(6):
            field = 0x10 | ((s >> (20 - 4 * n)) & 0xF)
            base = 16 + 30 * c + 5 * n
            b, sh = base >> 3, base & 7
            if b < 31:
                v = (f[b] << 8) | f[b + 1]
                v |= field << (11 - sh)
                f[b] = v >> 8
                f[b + 1] = v & 0xFF
            else:
                # base = 251: the whole field is byte 31's low 5 bits
                f[31] |= field
    # user value bits at frame bits 14..11 (bit 0 at 14)
    f[1] |= (user & 0xF) << 1
    return bytes(f)


def adat_decode_frame(frame):
    """Inverse of adat_encode_frame: 32 bytes -> (8 samples, user bits)."""
    assert len(frame) == 32
    reg = int.from_bytes(frame, "big")            # frame bit k = reg bit 255-k

    def bit(k):
        return (reg >> (255 - k)) & 1

    samples = []
    for c in range(8):
        s = 0
        for n in range(6):
            base = 16 + 30 * c + 5 * n
            for j in range(4):
                s = (s << 1) | bit(base + 1 + j)   # MSB first: bits 23-4n .. 20-4n
        samples.append(s)
    user = (frame[1] >> 1) & 0xF
    return samples, user


def frame_crc32(frame):
    """Ethernet FCS value (CRC-32, reflected) over the frame bytes."""
    return zlib.crc32(frame) & 0xFFFFFFFF


def ethernet_fcs(frame):
    """The 4 FCS bytes appended after the payload (little-endian)."""
    return struct.pack("<I", frame_crc32(frame))


def build_audio_frame(direction, sample_cntr, slots, src_mac, dst_mac,
                      fcs=True, pad=True):
    """Build one audio Ethernet frame (§2/§3/§4).

    direction: 0 = TX (device -> PC, 3 slots/sample), 1 = RX (PC -> device,
    2 slots/sample). slots: flat bytes, slot s*n_slots+i ... in wire order.
    Returns the frame WITHOUT preamble/SFD (like LiteEth's user streams
    see it after stripping, resp. before insertion).
    """
    n_slots = 3 if direction == 0 else 2
    audio = b"".join(slots)
    assert len(audio) == N_SAMPLES * n_slots * FRAME_BYTES
    ver_flags = VER_FLAGS_TX if direction == 0 else VER_FLAGS_RX
    payload_len = PAYLOAD_LEN_TX if direction == 0 else PAYLOAD_LEN_RX
    hdr = struct.pack("<BBIH", MAGIC, ver_flags, sample_cntr & 0xFFFFFFFF,
                      payload_len)
    frame = dst_mac + src_mac + struct.pack(">H", ETHERTYPE_AUDIO) + hdr + audio
    if pad and len(frame) < 60:
        frame += b"\x00" * (60 - len(frame))
    if fcs:
        frame += ethernet_fcs(frame)
    return frame


def audio_tx_slots(frames):
    """TX-direction slot order (§4.1): sample-major, then input-major.

    frames: per input, a list of N_SAMPLES 32-byte frames.
    Returns the flat 1440-byte payload audio part.
    """
    assert len(frames) == 3
    out = bytearray()
    for s in range(N_SAMPLES):
        for i in range(3):
            out += frames[i][s]
    assert len(out) == AUDIO_TX_PAYLOAD
    return bytes(out)


def audio_rx_slots(frames):
    """RX-direction slot order (§4.2): sample-major, then output-major.

    frames: per output, a list of N_SAMPLES 32-byte frames.
    """
    assert len(frames) == 2
    out = bytearray()
    for s in range(N_SAMPLES):
        for o in range(2):
            out += frames[o][s]
    assert len(out) == AUDIO_RX_PAYLOAD
    return bytes(out)


def build_hello_frame(serial, nonce, src_mac, dst_mac=BROADCAST_MAC,
                      role=HELLO_ROLE_DEVICE, flags=0, fcs=True, pad=True):
    """Build one hello frame (EtherType 0x88B6, §12.1 layout)."""
    pl = b"ADAT" + bytes([role, flags]) + struct.pack("<HIIHH",
        1,           # proto_ver
        serial & 0xFFFFFFFF,
        nonce & 0xFFFFFFFF,
        N_SAMPLES,   # block_frames
        0)           # reserved
    assert len(pl) == HELLO_LEN
    frame = dst_mac + src_mac + struct.pack(">H", ETHERTYPE_HELLO) + pl
    if pad and len(frame) < 60:
        frame += b"\x00" * (60 - len(frame))
    if fcs:
        frame += ethernet_fcs(frame)
    return frame


def parse_hello_payload(payload):
    """Parse a hello payload -> dict or None (§12.1 parser rules)."""
    if len(payload) < HELLO_LEN or payload[0:4] != b"ADAT":
        return None
    role, flags = payload[4], payload[5]
    ver, serial, nonce, block, reserved = struct.unpack_from("<HIIHH", payload, 6)
    if ver != 1 or reserved != 0:
        return None
    return dict(role=role, flags=flags, serial=serial, nonce=nonce,
                block_frames=block)


def parse_audio_header(hdr):
    """Parse the 8-byte header -> dict or None (strict like the PC module)."""
    if len(hdr) < HEADER_LEN:
        return None
    magic, ver_flags, sample_cntr, payload_len = struct.unpack(
        "<BBIH", hdr[:HEADER_LEN])
    if magic != MAGIC:
        return None
    if (ver_flags >> 4) != 0:            # version 0
        return None
    if ver_flags & 0x3:                  # reserved bits must be 0
        return None
    return dict(ver_flags=ver_flags,
                direction=(ver_flags >> 3) & 1,
                mode=(ver_flags >> 2) & 1,
                sample_cntr=sample_cntr,
                payload_len=payload_len)
