#!/usr/bin/env python3
#
# EtherADAT bridge top-level for the ADATface board: 3 ADAT inputs and
# 2 ADAT outputs bridged to raw-Ethernet AdatNet frames (PROTOCOL.md,
# role 1 = the RP2350 part) over an RMII-attached LAN8720 PHY.
#
# Build (from gateware/, project venv activated):
#   python etheradat/etheradat_interface.py
# or with the platform selected via the environment:
#   LUNA_PLATFORM=platforms:ADATFaceCycloneIV python etheradat/etheradat_interface.py
#
# The generated LiteEth MAC core (liteeth/liteeth_rmii_mac.v) must be
# present; it is checked in and added to the build plan automatically.
#
# Wiring (J7 -> LAN8720 module), see etheradat/README.md for the full
# table and the PHY strap requirements:
#   J2 pin  8 -> TXD0        J2 pin 18 -> CRS_DV
#   J2 pin 10 -> TXD1        J2 pin 20 -> REF_CLK (direction depends
#   J2 pin 12 -> TX_EN                     on ref_clk_mode, see below)
#   J2 pin 14 -> RXD0        J2 pin 24 -> MDC
#   J2 pin 16 -> RXD1        J2 pin 26 -> MDIO
#   J2 pin  4 -> +3.3V       (J2 pin 22 unassigned: the module's nRST
#   J2 pin  6 -> GND          is NC, the PHY self-resets)

import os
import sys

from amaranth            import *
from amaranth.build      import Attrs, Pins, Resource, Subsignal
from amaranth.lib.cdc    import FFSynchronizer

from luna                import top_level_cli

from adat import ADATTransmitter, ADATReceiver

from etheradat.bridge import EtherADATBridge, RMIIClockDomain


class EtherADATInterface(Elaboratable):
    """EtherADAT bridge (role 1) on the ADATface baseboard.

    Parameters
    ----------
    ref_clk_mode : "output" — the FPGA generates the 50 MHz RMII
        reference from the board's 50 MHz oscillator and drives it into
        the LAN8720 (module must be strapped REF_CLK-in, nINTSEL);
        "input" — the module drives REF_CLK into the FPGA (default
        strap of most LAN8720 modules).
    own_mac, device_serial, session_nonce : AdatNet identity.
    """

    def __init__(self, ref_clk_mode="output", own_mac=0x02AD6E657401,
                 device_serial=0x41444146, session_nonce=0):
        assert ref_clk_mode in ("input", "output")
        self.ref_clk_mode = ref_clk_mode
        self.own_mac = own_mac
        self.device_serial = device_serial
        self.session_nonce = session_nonce

    def elaborate(self, platform):
        m = Module()

        # the 50 MHz oscillator is requested here so the RMII reference
        # (in ref_clk_mode="output") and the PLLs share the exact same net
        clk_pin = platform.request(platform.default_clk)
        m.submodules.car = car = platform.clock_domain_generator(
            clk_input=clk_pin.i)

        #
        # RMII pads and the 50 MHz reference clock
        #
        if self.ref_clk_mode == "output":
            rmii = platform.request("rmii", 0, dir={"ref_clk": "o"})
            clk50 = clk_pin.i
            m.d.comb += rmii.ref_clk.o.eq(clk50)
            rmii_clk = clk50
        else:
            rmii = platform.request("rmii", 0, dir={"ref_clk": "i"})
            rmii_clk = rmii.ref_clk.i

        m.submodules.rmii_domain = rmii_domain = RMIIClockDomain()
        m.d.comb += [
            rmii_domain.clk_in.eq(rmii_clk),
            rmii_domain.rst_in.eq(0),
        ]
        # no PHY reset line: the LAN8720 module's nRST is NC, the PHY
        # releases its own power-on reset and samples its straps then

        #
        # the bridge
        #
        m.submodules.bridge = bridge = EtherADATBridge(
            own_mac=self.own_mac,
            device_serial=self.device_serial,
            session_nonce=self.session_nonce,
            fast_clk_freq=platform.fast_domain_clock_freq,
        )

        #
        # the generated LiteEth MAC core (Verilog blackbox; all six
        # clock/reset inputs share the RMII reference clock). Port names
        # = the generated module's (liteeth/liteeth_rmii_mac.v).
        #
        import os as _os
        _mac_v = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                               "liteeth", "liteeth_rmii_mac.v")
        platform.add_file("liteeth_rmii_mac.v", open(_mac_v, "rb"))

        m.submodules.liteeth_mac = Instance("liteeth_rmii_mac",
            i_sys_clk       = ClockSignal("rmii"),
            i_sys_rst       = ResetSignal("rmii"),
            i_eth_tx_clk    = ClockSignal("rmii"),
            i_eth_tx_rst    = ResetSignal("rmii"),
            i_eth_rx_clk    = ClockSignal("rmii"),
            i_eth_rx_rst    = ResetSignal("rmii"),
            o_rmii_tx_en    = rmii.tx_en.o,
            o_rmii_txd      = Cat(rmii.txd0.o, rmii.txd1.o),
            i_rmii_crs_dv   = rmii.crs_dv.i,
            i_rmii_rxd      = Cat(rmii.rxd0.i, rmii.rxd1.i),
            i_user_sink_payload_data = bridge.mac_tx_data,
            i_user_sink_valid        = bridge.mac_tx_valid,
            i_user_sink_last         = bridge.mac_tx_last,
            o_user_sink_ready        = bridge.mac_tx_ready,
            o_user_source_payload_data = bridge.mac_rx_data,
            o_user_source_valid        = bridge.mac_rx_valid,
            o_user_source_last         = bridge.mac_rx_last,
            o_user_source_payload_error = bridge.mac_rx_error,
            i_user_source_ready        = bridge.mac_rx_ready,
        )

        #
        # MDIO (link status only; the PHY runs on strap defaults)
        #
        m.d.comb += [
            rmii.mdc.o.eq(bridge.mdio.mdc_out),
            rmii.mdio.o.eq(bridge.mdio.mdio_o),
            rmii.mdio.oe.eq(bridge.mdio.mdio_oe),
            bridge.mdio.mdio_i.eq(rmii.mdio.i),
        ]

        #
        # ADAT: 3 receivers (inputs), 2 transmitters (outputs).
        # TOSLINK connectors 1..3 carry the inputs; the outputs use the
        # transmitters of connectors 1 and 2 (each connector has an
        # independent RX and TX optic); connector 4 is unused.
        #
        toslink = [platform.request("toslink", i + 1) for i in range(3)]

        for i in range(3):
            receiver = DomainRenamer("fast")(
                ADATReceiver(platform.fast_domain_clock_freq))
            setattr(m.submodules, f"adat{i}_receiver", receiver)

            m.d.comb += [
                receiver.adat_in.eq(toslink[i].rx.i),
                bridge.builder.sample_in[i].eq(receiver.sample_out),
                bridge.builder.addr_in[i].eq(receiver.addr_out),
                bridge.builder.enable_in[i].eq(receiver.output_enable),
                bridge.builder.user_in[i].eq(receiver.user_data_out),
                bridge.builder.synced_in[i].eq(receiver.synced_out),
            ]

        for o in range(2):
            transmitter = ADATTransmitter(fifo_depth=9 * 4)
            setattr(m.submodules, f"adat{o}_transmitter", transmitter)

            m.d.comb += [
                toslink[o].tx.o.eq(transmitter.adat_out),
                transmitter.sample_in.eq(bridge.feeder.sample_out[o]),
                transmitter.addr_in.eq(bridge.feeder.addr_out[o]),
                transmitter.valid_in.eq(bridge.feeder.valid_out[o]),
                transmitter.last_in.eq(bridge.feeder.last_out[o]),
                transmitter.user_data_in.eq(bridge.feeder.user_out[o]),
                bridge.feeder.ready_in[o].eq(transmitter.ready_out),
            ]

        #
        # status LEDs (bring-up aid)
        #
        leds = platform.request("leds")
        m.d.comb += [
            leds.host.eq(bridge.link_up),          # PHY link (BMSR)
            leds.usb1.eq(bridge.pc_linked),        # PC MAC learned
            leds.usb2.eq(bridge.free_run),         # no ADAT input synced
            leds.sync1.eq(bridge.builder.synced_in[0]),
            leds.sync2.eq(bridge.builder.synced_in[1]),
            leds.sync3.eq(bridge.builder.synced_in[2]),
            # activity: MSB of the free-running frame counters
            leds.active1.eq(bridge.stats_tx_frames[22]),
            leds.active2.eq(bridge.stats_rx_frames[22]),
        ]

        return m


if __name__ == "__main__":
    os.environ.setdefault("LUNA_PLATFORM", "platforms:ADATFaceCycloneV")
    top_level_cli(EtherADATInterface)
