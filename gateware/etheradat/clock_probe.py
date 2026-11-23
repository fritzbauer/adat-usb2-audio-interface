#!/usr/bin/env python3
"""Minimal board/PHY clock probe (bring-up diagnostic).

The baseline core is clocked entirely from the LAN8720's RMII REF_CLK.
Silence on both the UART and the Ethernet link therefore means "no
50 MHz clock", not necessarily a broken design. This probe is clocked
from the board's own 50 MHz oscillator (clk50) and:

- counts rising edges of rmii.ref_clk over ~0.5 s windows,
- emits a UART dump `[0xAA][count LE32][0x55]` per window,
- blinks LEDs from the local counter and the sampled REF_CLK level.

If the UART shows windows with count ~25e6, the PHY clock is alive and
the baseline's problem is elsewhere. count == 0 means the module is not
driving REF_CLK (unpowered / REF_CLK-IN strap / unplugged).

Build: python etheradat/scripts/generate_quartus_files_clockprobe.py <dir>
"""

from amaranth            import *
from amaranth.build      import Attrs, Pins, Resource, Subsignal

from luna import top_level_cli

from etheradat.mdio import MDIOMaster

_uart_dbg = Resource("uart_dbg", 0,
    Subsignal("tx", Pins("J_2:34", dir="o"),
              Attrs(io_standard="3.3-V LVCMOS")))

# The module's reset pin is documented NC; driving J_2:22 high is a
# harmless experiment in case this module variant connects RST# there.
_phy_nrst = Resource("phy_nrst", 0,
    Subsignal("o", Pins("J_2:22", dir="o"),
              Attrs(io_standard="3.3-V LVCMOS")))


class ClockProbe(Elaboratable):
    def elaborate(self, platform):
        m = Module()

        clk = platform.request(platform.default_clk)
        m.domains.sys = ClockDomain()
        m.d.comb += ClockSignal("sys").eq(clk.i)

        # BISECT: add the same PLL clock generator the full design uses.
        WITH_PLLS = True
        if WITH_PLLS:
            m.submodules.car = car = platform.clock_domain_generator(
                clk_input=clk.i)
            sync_cnt = Signal(16)
            m.d.sync += sync_cnt.eq(sync_cnt + 1)
            from amaranth.lib.cdc import FFSynchronizer
            sync_seen = Signal(16)
            m.submodules += FFSynchronizer(sync_cnt, sync_seen,
                                           o_domain="sys")
            pll_ok = Signal()
        else:
            sync_seen = Signal(16)
            pll_ok = Signal()

        platform.add_resources([_uart_dbg, _phy_nrst])
        uart = platform.request("uart_dbg", 0)
        nrst = platform.request("phy_nrst", 0)
        m.d.comb += nrst.o.eq(1)
        leds = platform.request("leds")
        rmii = platform.request("rmii", 0, dir={"ref_clk": "i"})

        # MDIO probe, clocked from the board oscillator (independent of
        # the PHY clock): proves the module is powered and responsive.
        m.submodules.mdio = mdio = DomainRenamer("sys")(MDIOMaster(
            mdc_divider=10, poll_cycles=1 << 17,
            probe_addrs=tuple(range(32))))
        m.d.comb += [
            rmii.mdc.o.eq(mdio.mdc_out),
            rmii.mdio.o.eq(mdio.mdio_o),
            rmii.mdio.oe.eq(mdio.mdio_oe),
            mdio.mdio_i.eq(rmii.mdio.i),
        ]

        # REF_CLK edge counter (2-FF synchronizer + rising-edge detect)
        rc0 = Signal()
        rc1 = Signal()
        m.d.sys += [rc0.eq(rmii.ref_clk.i), rc1.eq(rc0)]
        rising = rc0 & ~rc1

        period = 25_000_000  # ~0.5 s at 50 MHz
        timer = Signal(25)
        edges = Signal(32)
        latched = Signal(32)
        fire = Signal()
        m.d.sys += [
            edges.eq(edges + rising),
            fire.eq(timer == period - 1),
        ]
        with m.If(fire):
            m.d.sys += [
                timer.eq(0),
                latched.eq(edges),
                edges.eq(rising),
            ]
        with m.Else():
            m.d.sys += timer.eq(timer + 1)

        # UART TX, 115200 8N1 from the 50 MHz local clock
        DIV = 434
        data = Signal(8)
        start = Signal()
        busy = Signal()
        phase = Signal(range(DIV))
        bit_i = Signal(4)
        shift = Signal(8)
        tx = Signal(reset=1)
        m.d.comb += uart.tx.eq(tx)
        with m.If(busy):
            with m.If(phase == DIV - 1):
                m.d.sys += phase.eq(0)
                with m.If(bit_i == 9):
                    m.d.sys += [busy.eq(0), tx.eq(1)]
                with m.Elif(bit_i == 8):
                    m.d.sys += [bit_i.eq(9), tx.eq(1)]      # stop bit
                with m.Else():
                    m.d.sys += [bit_i.eq(bit_i + 1), tx.eq(shift[0]),
                                shift.eq(shift >> 1)]
            with m.Else():
                m.d.sys += phase.eq(phase + 1)
        with m.Else():
            with m.If(start):
                m.d.sys += [busy.eq(1), shift.eq(data), bit_i.eq(0),
                            phase.eq(0), tx.eq(0)]

        # dump sequencer: 22 bytes per fire
        idx = Signal(5)
        dumping = Signal()
        with m.If(fire):
            m.d.sys += [dumping.eq(1), idx.eq(0)]
        with m.Elif(dumping & ~busy):
            with m.If(idx == 21):
                m.d.sys += dumping.eq(0)
            with m.Else():
                m.d.sys += idx.eq(idx + 1)
        m.d.comb += [
            start.eq(fire | (dumping & ~busy)),
            data.eq(Array([0xAA,
                           latched[0:8], latched[8:16],
                           latched[16:24], latched[24:32],
                           0x55,
                           0xBB,
                           mdio.phy_found_out,
                           Cat(mdio.phy_addr_out, 0, 0, 0)[0:8],
                           mdio.link_up_out,
                           0xCC,
                           0xDD,
                           rmii.mdio.i,
                           rmii.crs_dv.i,
                           rmii.rxd0.i,
                           rmii.rxd1.i,
                           rmii.ref_clk.i,
                           0xEE,
                           pll_ok,
                           sync_seen[0:8], sync_seen[8:16],
                           0xEF])[idx]),
        ]

        # LED sanity
        blink = Signal(26)
        m.d.sys += blink.eq(blink + 1)
        m.d.comb += [
            leds.host.eq(blink[24]),
            leds.usb1.eq(mdio.phy_found_out),
            leds.usb2.eq(rc0),
            leds.sync1.eq(blink[25]),
            leds.sync2.eq(mdio.link_up_out),
            leds.sync3.eq(0),
            leds.sync4.eq(0),
            leds.active1.eq(0),
            leds.active2.eq(0),
        ]
        return m


if __name__ == "__main__":
    import os
    os.environ.setdefault("LUNA_PLATFORM",
                          "platforms:ADATFaceCycloneIV_EP4CGX150")
    top_level_cli(ClockProbe)
