#!/usr/bin/env python3
"""Generate the Quartus project files for the EtherADAT bridge top
without running the synthesis toolchain (same flow as
scripts/generate_quartus_files.py, but for etheradat/).

Usage (from the gateware directory, project venv activated):
    ./venv/bin/python etheradat/scripts/generate_quartus_files_etheradat.py [BUILD_DIR]

The FPGA platform is selected with the LUNA_PLATFORM environment
variable; default: platforms:ADATFaceCycloneV.

Afterwards, on the synthesis host: cd BUILD_DIR && ./build_top.sh
"""

import argparse
import os
import sys

# Must be set before luna/amaranth are imported.
os.environ.setdefault("LUNA_PLATFORM", "platforms:ADATFaceCycloneV")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from luna.gateware.platform import get_appropriate_platform

from etheradat.etheradat_interface import EtherADATInterface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", nargs="?", default=None)
    parser.add_argument("--ref-clk-mode", default="output",
                        choices=["input", "output"],
                        help="who sources the 50 MHz RMII reference "
                             "(default: the FPGA)")
    args = parser.parse_args()

    platform = get_appropriate_platform()
    print(f"Platform: {platform.__class__.__name__} "
          f"(device {getattr(platform, 'device', '?')})")

    build_dir = args.build_dir
    if build_dir is None:
        plat_name = platform.__class__.__name__.lower()
        build_dir = f"build_etheradat_{plat_name}"

    top = EtherADATInterface(ref_clk_mode=args.ref_clk_mode)
    plan = platform.build(top, do_build=False)
    plan.execute_local(build_dir, run_script=False)

    print(f"Wrote {len(plan.files)} files to {build_dir}/:")
    for filename in plan.files:
        print(f"  {filename}")
    print("\nNext step (on the synthesis host):")
    print(f"  cd {build_dir} && ./build_top.sh")


if __name__ == "__main__":
    main()
