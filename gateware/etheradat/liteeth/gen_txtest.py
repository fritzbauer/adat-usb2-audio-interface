#!/usr/bin/env python3
"""Standalone TX shim test: sends a fixed valid Ethernet frame every 100 ms."""
import os, sys
from migen import *
from migen.fhdl.verilog import convert
from litex.gen import LiteXModule
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmii_shim import RMIITXShim
from gen_uartprobe import SlowTXGenerator

class TXTest(LiteXModule):
    def __init__(self):
        self.tx_en = Signal()
        self.txd = Signal(2)
        self.clock_domains.cd_eth_tx = ClockDomain()
        gen = SlowTXGenerator()
        shim = RMIITXShim(self.tx_en, self.txd)
        self.submodules.gen = ClockDomainsRenamer("eth_tx")(gen)
        self.submodules.shim = ClockDomainsRenamer("eth_tx")(shim)
        self.comb += gen.sink.connect(shim.sink)

if __name__ == "__main__":
    dut = TXTest()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "liteeth_txtest.v")
    convert(dut, ios={dut.tx_en, dut.txd, dut.cd_eth_tx.clk, dut.cd_eth_tx.rst},
            name="liteeth_txtest").write(out)
    print("wrote", out)
