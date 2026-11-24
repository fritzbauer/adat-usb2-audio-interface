#!/usr/bin/env python3
"""Generate the Quartus project files for the full EtherADAT bridge top
without running the synthesis toolchain.

Usage (from the gateware directory, project venv activated):
    LUNA_PLATFORM=platforms:ADATFaceCycloneIV_EP4CGX150 \
        ./venv/bin/python etheradat/scripts/generate_quartus_files_etheradat.py \
        [BUILD_DIR] [--ref-clk-mode input|output]

Afterwards, on the synthesis host: ~/quartus_build_host.sh (the build
directory must be gateware/build).
"""

import argparse
import os
import sys

# Must be set before luna/amaranth are imported.
os.environ.setdefault("LUNA_PLATFORM",
                      "platforms:ADATFaceCycloneIV_EP4CGX150")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from luna.gateware.platform import get_appropriate_platform

from etheradat.etheradat_interface import EtherADATInterface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", nargs="?", default=None)
    parser.add_argument("--ref-clk-mode", default="input",
                        choices=["input", "output"],
                        help="who sources the 50 MHz RMII reference "
                             "(default: the LAN8720 module, REF_CLK-OUT)")
    parser.add_argument("--no-audio-tx", action="store_true",
                        help="debug: suppress TX audio frames (hello-only)")
    parser.add_argument("--mon-source", default="ref",
                        choices=["ref", "clk50", "none"],
                        help="REF_CLK integrity monitor source: ref (the "
                             "REF_CLK pin), clk50 (control: the board "
                             "oscillator that also feeds the monitor PLL) "
                             "or none (no monitor PLL; Cyclone V PLL budget)")
    parser.add_argument("--mon-pll-src", default="clk50",
                        choices=["clk50", "ref"],
                        help="monitor PLL reference: clk50 (default) or "
                             "ref (the REF_CLK pin; capability probe for a "
                             "REF_CLK cleanup PLL)")
    parser.add_argument("--no-debug-taps", action="store_true",
                        help="drop the 16-byte TX capture/payload UART "
                             "taps (saves the per-byte CDC registers; "
                             "needed to fit the Cyclone V A2)")
    parser.add_argument("--tx-negedge", action="store_true",
                        help="drive RMII TX on the falling REF_CLK edge "
                             "(default: rising edge / same clock as the "
                             "MAC; the inverted ~REF_CLK clock was found "
                             "NOT to work on hardware 23.11.2026 — frames "
                             "lost their header)")
    args = parser.parse_args()

    platform = get_appropriate_platform()
    print(f"Platform: {platform.__class__.__name__} "
          f"(device {getattr(platform, 'device', '?')})")

    build_dir = args.build_dir
    if build_dir is None:
        plat_name = platform.__class__.__name__.lower()
        build_dir = f"build_etheradat_{plat_name}"

    top = EtherADATInterface(ref_clk_mode=args.ref_clk_mode,
                             audio_tx_enable=not args.no_audio_tx,
                             tx_negedge=args.tx_negedge,
                             mon_source=args.mon_source,
                             debug_taps=not args.no_debug_taps,
                             mon_pll_src=args.mon_pll_src)
    plan = platform.build(top, do_build=False)
    plan.execute_local(build_dir, run_script=False)

    # The RMII REF_CLK is not a Clock() resource: constrain it here. The
    # name must match the platform's set_clock_groups filter
    # (rmii_0__ref_clk*) so the asynchronous CDC groups are real. The
    # constraint MUST precede set_clock_groups: SDC is evaluated in
    # order, and the group filter silently matches nothing otherwise
    # (observed as negative setup slack on every cross-domain path).
    sdc_path = os.path.join(build_dir, "top.sdc")
    with open(sdc_path) as f:
        sdc = f.read()
    clock_line = ('create_clock -name "rmii_0__ref_clk" -period 20.0 '
                  '[get_ports "rmii_0__ref_clk__io"]\n')
    marker = "set_clock_groups"
    idx = sdc.find(marker)
    if idx < 0:
        sdc = sdc + "\n" + clock_line
    else:
        sdc = sdc[:idx] + clock_line + "\n" + sdc[idx:]
    with open(sdc_path, "w") as f:
        f.write(sdc)

    print(f"Wrote {len(plan.files)} files to {build_dir}/:")
    for filename in plan.files:
        print(f"  {filename}")
    print("\nNext step: copy to gateware/build and run "
          "~/quartus_build_host.sh")


if __name__ == "__main__":
    main()
