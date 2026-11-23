#!/usr/bin/env python3
"""Quartus files for the hardcoded ADAT output test."""

import argparse
import os
import sys

os.environ.setdefault("LUNA_PLATFORM",
                      "platforms:ADATFaceCycloneIV_EP4CGX150")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from luna.gateware.platform import get_appropriate_platform

from etheradat.adat_pattern_test import AdatPatternTest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", nargs="?", default="build_adatpat")
    args = parser.parse_args()

    platform = get_appropriate_platform()
    plan = platform.build(AdatPatternTest(), do_build=False)
    plan.execute_local(args.build_dir, run_script=False)
    print(f"Wrote files to {args.build_dir}/")


if __name__ == "__main__":
    main()
