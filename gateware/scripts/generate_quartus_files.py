#!/usr/bin/env python3
"""Generate the Quartus project files for the ADATface gateware without
running the synthesis toolchain.

This runs the same elaboration as `adat_usb2_audio_interface.py` and uses
Amaranth's `platform.build(..., do_build=False)` to produce the complete
build plan, then writes it out without invoking Quartus. The resulting
directory contains top.v, top.qsf, top.sdc, top.qpf and build_top.sh and
can be synthesized on the host with ./build_top.sh.

Usage (from the gateware directory, with the project venv activated):
    ./venv/bin/python scripts/generate_quartus_files.py [BUILD_DIR]

The FPGA platform is selected with the LUNA_PLATFORM environment variable
exactly like in adat_usb2_audio_interface.py; it defaults here to
"platforms:ADATFaceCycloneV" (QMTech 5CEFA2F23C8 daughterboard).
"""

import argparse
import os
import sys

# Must be set before luna/amaranth are imported.
os.environ.setdefault("LUNA_PLATFORM", "platforms:ADATFaceCycloneV")

# Make the gateware modules (platforms.py, stereo_convolution_mac.py, ...)
# importable; this script is meant to be run from the gateware directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from luna.gateware.platform import get_appropriate_platform

from adat_usb2_audio_interface import USB2AudioInterface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", nargs="?", default=None,
                        help="Directory to write the Quartus project into "
                             "(default: build_<platform>, e.g. build_cycloneV)")
    args = parser.parse_args()

    platform = get_appropriate_platform()
    print(f"Platform: {platform.__class__.__name__} "
          f"(device {getattr(platform, 'device', '?')})")

    build_dir = args.build_dir
    if build_dir is None:
        plat_name = platform.__class__.__name__.lower()
        build_dir = f"build_{plat_name}"

    top = USB2AudioInterface()
    # do_build=False: elaborate everything and produce the file templates,
    # but do not look for or run the Quartus toolchain.
    plan = platform.build(top, do_build=False)
    plan.execute_local(build_dir, run_script=False)

    print(f"Wrote {len(plan.files)} files to {build_dir}/:")
    for filename in plan.files:
        print(f"  {filename}")
    print(f"\nNext step (on the host, not in this VM):")
    print(f"  cd {build_dir} && ./build_top.sh")


if __name__ == "__main__":
    main()
