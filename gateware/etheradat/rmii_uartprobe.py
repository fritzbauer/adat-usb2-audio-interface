#!/usr/bin/env python3
#
# UART RMII probe top: two RX shims (plain + inverted REF_CLK edge)
# with UART dumps of every captured frame. See liteeth/gen_uartprobe.py
# for the protocol (0xAA <marker> <len> <bytes>, heartbeat 0x7E).
#
# UART: J2:34 = FPGA TX (to the adapter's RX), J2:36 = FPGA RX (unused
# here), 115200 8N1.

from amaranth            import Elaboratable, Module, Signal, Instance, \
                                Cat, ClockSignal, ClockDomain
from amaranth.build      import Attrs, Pins, Resource, Subsignal

from luna import top_level_cli

_rmii = Resource("rmii", 0,
    Subsignal("txd0",   Pins("J_2:8",  dir="io")),
    Subsignal("txd1",   Pins("J_2:10", dir="io")),
    Subsignal("tx_en",  Pins("J_2:12", dir="io")),
    Subsignal("rxd0",   Pins("J_2:14", dir="io")),
    Subsignal("rxd1",   Pins("J_2:16", dir="io")),
    Subsignal("crs_dv", Pins("J_2:18", dir="io")),
    Subsignal("ref_clk", Pins("J_2:20", dir="io")),
    Subsignal("mdc",    Pins("J_2:24", dir="io")),
    Subsignal("mdio",   Pins("J_2:26", dir="io"),
              Attrs(io_standard="3.3-V LVCMOS")),
    Attrs(io_standard="3.3-V LVCMOS"))

_uart = Resource("uart_dbg", 0,
    Subsignal("tx", Pins("J_2:34", dir="o")),
    Subsignal("rx", Pins("J_2:36", dir="i"),
              Attrs(io_standard="3.3-V LVCMOS")),
    Attrs(io_standard="3.3-V LVCMOS"))


class RMIIUARTProbe(Elaboratable):
    def __init__(self, tx_swap=False, tx_negedge=True):
        self.tx_swap = tx_swap
        # TX on the falling REF_CLK edge: the module samples TXD on the
        # rising edge, and on hardware the plain-edge drive produced
        # systematic bit errors in the upper dibit pairs (UART probe
        # 21.11.2026). Same convention as the repeater.
        self.tx_negedge = tx_negedge

    def elaborate(self, platform):
        m = Module()

        import os as _os
        _v = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                           "liteeth", "uart_probe.v")
        platform.add_file("uart_probe.v", open(_v, "rb"))
        platform.add_resources([_uart])

        rmii = platform.request("rmii", 0, dir={"ref_clk": "i"})
        uart = platform.request("uart_dbg", 0)
        leds = platform.request("leds")

        m.d.comb += [
            leds.host.eq(rmii.crs_dv.i),
            leds.usb1.eq(uart.tx),
        ]

        m.submodules.uart_probe = Instance("uart_probe",
            i_rmii_crs_dv = rmii.crs_dv.i,
            i_rmii_rxd    = Cat(rmii.rxd0.i, rmii.rxd1.i),
            o_uart_txd    = uart.tx,
            o_rmii_tx_en  = rmii.tx_en.o,
            # tx_swap: the module's TXD0/TXD1 may be crossed on the
            # wire; swap which of our pads carries which dibit lane
            o_rmii_txd    = (Cat(rmii.txd1.o, rmii.txd0.o)
                             if self.tx_swap else
                             Cat(rmii.txd0.o, rmii.txd1.o)),
            i_sys_clk     = rmii.ref_clk.i,
            i_sys_rst     = 0,
            i_eth_rx_clk  = rmii.ref_clk.i,
            i_eth_rx_rst  = 0,
            i_negrx_clk   = (~rmii.ref_clk.i),
            i_negrx_rst   = 0,
            i_eth_tx_clk  = (~rmii.ref_clk.i) if self.tx_negedge \
                            else rmii.ref_clk.i,
            i_eth_tx_rst  = 0,
        )

        return m


if __name__ == "__main__":
    import os
    os.environ.setdefault("LUNA_PLATFORM",
                          "platforms:ADATFaceCycloneIV_EP4CGX150")
    top_level_cli(RMIIUARTProbe)
