#!/usr/bin/env python3
#
# EtherADAT bring-up baseline: the stock LiteEth IP/UDP stack (liteeth/
# liteeth_baseline.v, UDP echo on port 1234, ping-able at 192.168.42.2)
# around the proven RMII shim — NO EtherADAT bridge RTL. Purpose: prove
# the RMII wiring and the LAN8720 module independently of our gateware.
#
# FPGA: MAC 02:AD:6E:65:74:01, IP 192.168.42.2/24 (static).
# PC:   static 192.168.42.1/24 on the USB-dock interface, then
#       ping 192.168.42.2  and
#       python -c 'import socket; s=socket.socket(socket.AF_INET,
#                  socket.SOCK_DGRAM); s.settimeout(2); s.sendto(b"hello",
#                  ("192.168.42.2",1234)); print(s.recvfrom(100))'
#
# Build like the etheradat design:
#   python etheradat/scripts/generate_quartus_files_baseline.py <dir>

from amaranth            import *
from amaranth.build      import Attrs, Pins, Resource, Subsignal

from luna import top_level_cli

_uart_dbg = Resource("uart_dbg", 0,
    Subsignal("tx", Pins("J_2:34", dir="o"),
              Attrs(io_standard="3.3-V LVCMOS")))


class EtherADATBaseline(Elaboratable):
    def __init__(self, ref_clk_mode="input", tx_probe=None, tx_swap=True,
                 tx_negedge=False, rx_negedge=False):
        assert ref_clk_mode in ("input", "output")
        self.ref_clk_mode = ref_clk_mode
        # bring-up: drive only ONE tx pin (to 0), leave the others
        # floating, to find which driven pin kills the PHY link
        assert tx_probe in (None, "float", "tx_en", "txd0", "txd1")
        self.tx_probe = tx_probe
        self.tx_swap = tx_swap
        # TX on the falling REF_CLK edge (repeater-verified: the module
        # samples TXD on the rising edge; driving on the same edge gives
        # the bits zero setup time). Same convention as the repeater.
        self.tx_negedge = tx_negedge
        # RX on the plain edge (UART-probe-verified 21.11.2026: the
        # module changes RXD/CRS_DV on the FALLING edge, so plain-edge
        # sampling has a half cycle of setup — the inverted edge lands
        # exactly on the transition and produces broken frames).
        self.rx_negedge = rx_negedge

    def elaborate(self, platform):
        m = Module()

        import os as _os
        _v = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                           "liteeth", "liteeth_baseline.v")
        platform.add_file("liteeth_baseline.v", open(_v, "rb"))

        # The RMII reference clock source is configurable. With
        # ref_clk_mode="output" the FPGA drives the module's clock input
        # (the module latches the nINTSEL strap from this pin at ITS
        # power-on; a HIGH level forces REF_CLK-IN). To make the strap
        # latch deterministic, REF_CLK is driven constant-HIGH for 30 s
        # after FPGA configuration, then switches to the 50 MHz clock.
        # The MODE straps (TX_EN/RXD0/RXD1) default to all-capable via
        # the PHY's internal pull-ups when floating, so no forcing there.
        clk_pin = platform.request(platform.default_clk)

        if self.ref_clk_mode == "output":
            rmii = platform.request("rmii", 0, dir={"ref_clk": "o"})
            m.d.comb += [
                rmii.ref_clk.o.eq(clk_pin.i),
            ]
            rmii_clk = clk_pin.i
        else:
            rmii = platform.request("rmii", 0, dir={"ref_clk": "i"})
            rmii_clk = rmii.ref_clk.i
        platform.add_resources([_uart_dbg])
        uart = platform.request("uart_dbg", 0)
        leds = platform.request("leds")

        # Debug outputs: pulses are stretched in the core's own domain
        # before crossing to the asynchronous LED clock.
        dbg_names = ("crs_dv", "rx_udp", "rx_mac", "tx_mac", "pre_err",
                     "crc_err", "arp_rx", "arp_req", "arp_rep", "mac_in",
                     "mac_out", "led_shim_last", "led_cdc_last",
                     "led_mac_last", "led_arp_req", "led_arp_rep",
                     "led_tx_last", "led_rx_err")
        dbg = {_n: Signal(name="dbg_" + _n) for _n in dbg_names}

        ports = {
            "i_sys_clk":    rmii_clk,
            "i_sys_rst":    0,
            "i_eth_tx_clk": (~rmii_clk) if self.tx_negedge else rmii_clk,
            "i_eth_tx_rst": 0,
            "i_eth_rx_clk": (~rmii_clk) if self.rx_negedge else rmii_clk,
            "i_eth_rx_rst": 0,
            "o_rmii_tx_en": rmii.tx_en.o,
            "o_rmii_txd":   Cat(rmii.txd0.o, rmii.txd1.o),
            "i_rmii_crs_dv": rmii.crs_dv.i,
            "i_rmii_rxd":   Cat(rmii.rxd0.i, rmii.rxd1.i),
        }
        for _name in dbg_names:
            ports["o_dbg_" + _name] = dbg[_name]
        ports["o_uart_txd"] = uart.tx

        m.submodules.liteeth_baseline = Instance("liteeth_baseline", **ports)
        # MDC/MDIO unused: the PHY runs on strap defaults, no MDIO here.

        # The dbg_* signals are us-short pulses (one MAC byte stream
        # window per frame) — an LED cannot show them directly. Hold
        # each one for ~0.7 s after its rising edge (50 MHz osc domain)
        # so every received frame lights its LED for exactly one blip.
        HOLD = (1 << 25)   # ~0.67 s at 50 MHz
        m.domains.led = ClockDomain()
        m.d.comb += ClockSignal("led").eq(clk_pin.i)
        for _sig, _led in (
            (dbg["led_shim_last"], leds.usb2),    # RX shim frame-end
            (dbg["led_cdc_last"], leds.sync1),    # after same-clock CDC
            (dbg["led_mac_last"], leds.sync2),    # MAC depacketizer input
            (dbg["led_arp_req"], leds.sync3),     # ARP request recognized
            (dbg["led_arp_rep"], leds.sync4),     # ARP reply emitted
            (dbg["led_tx_last"], leds.host),      # TX frame from MAC
            (dbg["crs_dv"], leds.usb1),           # raw carrier activity
            (dbg["pre_err"], leds.active1),
            (dbg["crc_err"], leds.active2),
        ):
            _prev = Signal(reset=0)
            _held = Signal(reset=0)
            _hcnt = Signal(25, reset=0)
            m.d.led += _prev.eq(_sig)
            with m.If(_sig & ~_prev):
                m.d.led += _held.eq(1), _hcnt.eq(0)
            with m.Elif(_held):
                m.d.led += _hcnt.eq(_hcnt + 1)
                with m.If(_hcnt == (HOLD - 1)):
                    m.d.led += _held.eq(0)
            m.d.comb += _led.eq(_held)

        return m


if __name__ == "__main__":
    os.environ.setdefault("LUNA_PLATFORM", "platforms:ADATFaceCycloneV")
    top_level_cli(EtherADATBaseline)
