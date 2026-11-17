"""Sanity checks of the golden reference model itself (no RTL).

Run:  python -m etheradat.tests.test_reference
"""
import random
import unittest

from etheradat import reference as ref


class TestReference(unittest.TestCase):
    def test_empty_frame_constant(self):
        self.assertEqual(
            ref.adat_encode_frame([0] * 8, user=0), ref.EMPTY_USER_FRAME)

    def test_round_trip(self):
        rng = random.Random(0xADC0FFEE)
        for _ in range(200):
            samples = [rng.getrandbits(24) for _ in range(8)]
            user = rng.getrandbits(4)
            frame = ref.adat_encode_frame(samples, user)
            self.assertEqual(len(frame), 32)
            s2, u2 = ref.adat_decode_frame(frame)
            self.assertEqual(s2, samples)
            self.assertEqual(u2, user)

    def test_decode_empty(self):
        samples, user = ref.adat_decode_frame(ref.EMPTY_USER_FRAME)
        self.assertEqual(samples, [0] * 8)
        self.assertEqual(user, 0)

    def test_marker_bits_always_one(self):
        rng = random.Random(1)
        for _ in range(50):
            frame = ref.adat_encode_frame(
                [rng.getrandbits(24) for _ in range(8)], rng.getrandbits(4))
            reg = int.from_bytes(frame, "big")

            def bit(k):
                return (reg >> (255 - k)) & 1
            # sync trough: 10 zeros
            for k in range(10):
                self.assertEqual(bit(k), 0)
            # user block markers + 48 channel markers
            self.assertEqual(bit(10), 1)
            self.assertEqual(bit(15), 1)
            for c in range(8):
                for n in range(6):
                    self.assertEqual(bit(16 + 30 * c + 5 * n), 1)

    def test_fcs_is_zlib_crc32_le(self):
        frame = bytes(range(64))
        self.assertEqual(ref.ethernet_fcs(frame),
                         ref.frame_crc32(frame).to_bytes(4, "little"))
        # known value cross-check: zlib.crc32 semantics
        import zlib
        self.assertEqual(ref.frame_crc32(frame),
                         zlib.crc32(frame) & 0xFFFFFFFF)

    def test_hello_round_trip(self):
        src = b"\x02\xad\x6e\x65\x74\x01"
        dst = b"\x11" * 6
        frame = ref.build_hello_frame(0xDEADBEEF, 0x12345678, src, dst)
        # 14 + 20 = 34 -> padded to 60 + 4 FCS
        self.assertEqual(len(frame), 64)
        self.assertEqual(frame[:6], dst)
        self.assertEqual(frame[6:12], src)
        self.assertEqual(frame[12:14], b"\x88\xb6")
        payload = frame[14:60]
        parsed = ref.parse_hello_payload(payload)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["role"], ref.HELLO_ROLE_DEVICE)
        self.assertEqual(parsed["serial"], 0xDEADBEEF)
        self.assertEqual(parsed["nonce"], 0x12345678)
        self.assertEqual(parsed["block_frames"], 15)
        # FCS must verify
        body, fcs = frame[:-4], frame[-4:]
        self.assertEqual(ref.ethernet_fcs(body), fcs)

    def test_audio_header_parse(self):
        hdr = ref.build_audio_frame(
            0, 12345, [ref.EMPTY_USER_FRAME] * 45,
            b"\x02" * 6, ref.BROADCAST_MAC, fcs=False, pad=False)
        parsed = ref.parse_audio_header(hdr[14:22])
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["direction"], 0)
        self.assertEqual(parsed["mode"], 0)
        self.assertEqual(parsed["sample_cntr"], 12345)
        self.assertEqual(parsed["payload_len"], ref.PAYLOAD_LEN_TX)


if __name__ == "__main__":
    unittest.main(verbosity=2)
