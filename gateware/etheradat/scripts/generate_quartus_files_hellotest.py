#!/usr/bin/env python3
"""Quartus files for the PHY reset probe (etheradat/phy_reset_probe.py)."""

import argparse
import os
import sys

os.environ.setdefault("LUNA_PLATFORM",
                      "platforms:ADATFaceCycloneIV_EP4CGX150")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from luna.gateware.platform import get_appropriate_platform

from etheradat.mac_hello_test import MacHelloTest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", nargs="?", default=None)
    args = parser.parse_args()

    platform = get_appropriate_platform()
    build_dir = args.build_dir or "build_hellotest"
    plan = platform.build(MacHelloTest(), do_build=False)
    plan.execute_local(build_dir, run_script=False)

    with open(os.path.join(build_dir, "top.sdc"), "a") as f:
        f.write('\ncreate_clock -name "clk50_0__io" -period 20.0 '
                '[get_ports "clk50_0__io"]\n')
    print(f"Wrote files to {build_dir}/")


if __name__ == "__main__":
    main()
