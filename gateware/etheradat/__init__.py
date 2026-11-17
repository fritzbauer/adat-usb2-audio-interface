"""EtherADAT bridge gateware.

Bridges 3 ADAT inputs / 2 ADAT outputs to raw-Ethernet (Layer 2) frames
carrying the AdatNet protocol (see /mnt/Audio/etheradat/PROTOCOL.md), via
an RMII-attached LAN8720 PHY. This is the FPGA implementation of what the
EtherADAT project's RP2350 firmware does (role 1 / device side).
"""

from etheradat.reference import (
    ETHERTYPE_AUDIO,
    ETHERTYPE_HELLO,
    MAGIC,
    N_SAMPLES,
    PAYLOAD_LEN_RX,
    PAYLOAD_LEN_TX,
    VER_FLAGS_RX,
    VER_FLAGS_TX,
    EMPTY_USER_FRAME,
    adat_decode_frame,
    adat_encode_frame,
    build_audio_frame,
    build_hello_frame,
    ethernet_fcs,
    frame_crc32,
)
