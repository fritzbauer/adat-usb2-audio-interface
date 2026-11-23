"""Guard the ADAT receiver's clock-ratio assumption.

adat-core's ADATReceiver/NRZIDecoder hardcodes ``bit_time = 8`` fast-clock
cycles per ADAT bit (12.288 Mbit/s at 48 kHz). The platform must therefore
run the "fast" domain at exactly 8 x that rate, i.e. 98.304 MHz.

This is a regression test for the ADAT-input bug fixed on 2026-11-23: the
Cyclone IV platforms declared fast_multiplier = 9, so car.py's fast PLL
actually produced 110.592 MHz. The receiver still "synced" (the sync-trough
lock window is wide enough) but lost/added bits, and the ADAT input decoded
as the seeded EMPTY frame (silence) or garbage. Both projects (EtherADAT
and adat_usb2_audio_interface) share this platform and the same receiver.
"""

import unittest

from adat.nrzidecoder import NRZIDecoder

import platforms

ADAT_BITRATE = NRZIDecoder.adat_freq(48000)     # 12_288_000 bit/s
FAST_CYCLES_PER_BIT = 8                          # hardcoded in NRZIDecoder


class TestFastClockRatio(unittest.TestCase):
    def _check(self, platform_cls):
        freq = platform_cls.fast_domain_clock_freq
        self.assertEqual(
            freq, FAST_CYCLES_PER_BIT * ADAT_BITRATE,
            f"{platform_cls.__name__}: fast clock {freq/1e6:.3f} MHz is not "
            f"{FAST_CYCLES_PER_BIT}x the {ADAT_BITRATE/1e6:.3f} MHz ADAT bit "
            "rate; ADATReceiver's bit_time=8 will mis-sample the input")

    def test_cyclone_iv_ep4cgx150_is_8x(self):
        self._check(platforms.ADATFaceCycloneIV_EP4CGX150)

    def test_cyclone_iv_ep4ce_is_8x(self):
        self._check(platforms.ADATFaceCycloneIV)

    def test_cyclone_v_is_8x(self):
        # declaration only; this generator's actual PLL output is fixed at
        # 98.304 MHz regardless of fast_multiplier
        self._check(platforms.ADATFaceCycloneV)


if __name__ == "__main__":
    unittest.main(verbosity=2)
