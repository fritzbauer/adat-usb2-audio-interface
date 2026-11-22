#!/usr/bin/env python3
from amaranth import *
from luna import top_level_cli

class TXTestTop(Elaboratable):
    def elaborate(self, platform):
        m = Module()
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        platform.add_file("liteeth_txtest.v",
                          open(os.path.join(here, "liteeth", "liteeth_txtest.v"), "rb"))
        rmii = platform.request("rmii", 0, dir={"ref_clk": "i"})
        m.submodules.txtest = Instance("liteeth_txtest",
            i_eth_tx_clk=rmii.ref_clk.i, i_eth_tx_rst=0,
            o_tx_en=rmii.tx_en.o, o_txd=Cat(rmii.txd0.o, rmii.txd1.o))
        return m

if __name__ == "__main__":
    import os
    os.environ.setdefault("LUNA_PLATFORM", "platforms:ADATFaceCycloneIV_EP4CGX150")
    top_level_cli(TXTestTop)
