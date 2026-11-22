#!/usr/bin/env python3
import os, sys
os.environ.setdefault("LUNA_PLATFORM", "platforms:ADATFaceCycloneIV_EP4CGX150")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from luna.gateware.platform import get_appropriate_platform
from etheradat.txtest import TXTestTop
plan = get_appropriate_platform().build(TXTestTop(), do_build=False)
plan.execute_local("build_txtest", run_script=False)
with open("build_txtest/top.sdc", "a") as f:
    f.write('\ncreate_clock -name "clk50_0__io" -period 20.0 [get_ports "clk50_0__io"]\n')
print("Wrote build_txtest/")
