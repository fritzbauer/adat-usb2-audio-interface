#!/usr/bin/env python3
"""Generate the UART RMII probe (etheradat/rmii_uartprobe.py)."""
import os, sys
os.environ.setdefault("LUNA_PLATFORM", "platforms:ADATFaceCycloneIV_EP4CGX150")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--tx-swap", action="store_true")
parser.add_argument("--tx-no-negedge", action="store_true")
args = parser.parse_args()

from luna.gateware.platform import get_appropriate_platform
from etheradat.rmii_uartprobe import RMIIUARTProbe
platform = get_appropriate_platform()
plan = platform.build(RMIIUARTProbe(tx_swap=args.tx_swap,
                                    tx_negedge=not args.tx_no_negedge),
                      do_build=False)
plan.execute_local("build_etheradat_uartprobe", run_script=False)
with open("build_etheradat_uartprobe/top.sdc", "a") as f:
    f.write('\ncreate_clock -name "clk50_0__io" -period 20.0 '
            '[get_ports "clk50_0__io"]\n')
print("Wrote build_etheradat_uartprobe/")
