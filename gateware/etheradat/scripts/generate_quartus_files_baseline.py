#!/usr/bin/env python3
"""Generate the Quartus files for the EtherADAT bring-up baseline
(stock LiteEth IP/UDP stack, see etheradat/etheradat_baseline.py and
liteeth/gen_baseline.py). Same host build flow as the etheradat
generator; appends the 50 MHz create_clock to the SDC (no PLLs here,
so the audio-PLL SDC additions match nothing).

Usage: python etheradat/scripts/generate_quartus_files_baseline.py [dir]
"""

import argparse
import os
import sys

# Must be set before luna/amaranth are imported.
os.environ.setdefault("LUNA_PLATFORM", "platforms:ADATFaceCycloneIV_EP4CGX150")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from luna.gateware.platform import get_appropriate_platform

from etheradat.etheradat_baseline import EtherADATBaseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", nargs="?", default=None)
    parser.add_argument("--no-tx-swap", action="store_true",
                        help="TXD0/TXD1 unswapped (repeater-verified: "
                             "swapped is correct)")
    parser.add_argument("--tx-probe", default=None,
                        choices=[None, "float", "tx_en", "txd0", "txd1"],
                        help="drive only this TX pin (bring-up probe)")
    parser.add_argument("--ref-clk-mode", default="output",
                        choices=["input", "output"],
                        help="who sources the RMII reference clock "
                             "(default: the FPGA, with strap forcing)")
    args = parser.parse_args()

    platform = get_appropriate_platform()
    print(f"Platform: {platform.__class__.__name__} "
          f"(device {getattr(platform, 'device', '?')})")

    build_dir = args.build_dir
    if build_dir is None:
        plat_name = platform.__class__.__name__.lower()
        build_dir = f"build_etheradat_baseline_{plat_name}"

    top = EtherADATBaseline(ref_clk_mode=args.ref_clk_mode,
                            tx_probe=args.tx_probe,
                            tx_swap=not args.no_tx_swap)
    plan = platform.build(top, do_build=False)
    plan.execute_local(build_dir, run_script=False)

    # no PLLs in the baseline: constrain the oscillator directly
    sdc_path = os.path.join(build_dir, "top.sdc")
    with open(sdc_path, "a") as f:
        f.write('\ncreate_clock -name "clk50_0__io" -period 20.0 '
                '[get_ports "clk50_0__io"]\n')
        f.write('create_clock -name "rmii_ref_clk" -period 20.0 '
                '[get_ports "rmii_0__ref_clk__io"]\n')

    print(f"Wrote {len(plan.files)} files to {build_dir}/")
    print("\nNext step (on the synthesis host):")
    print(f"  cd {build_dir} && ./build_top.sh")


if __name__ == "__main__":
    main()
