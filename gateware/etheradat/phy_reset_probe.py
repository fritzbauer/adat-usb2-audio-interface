#!/usr/bin/env python3
"""Last-resort LAN8720 bring-up: drive REF_CLK + MDIO BMCR writes.

Hypothesis: after a power event the module latched REF_CLK-IN (nINTSEL
high) while the FPGA was unconfigured, so it neither drives REF_CLK nor
releases its management interface. This design drives the 50 MHz
reference into the module from the board oscillator, pulses nRST (J2:22,
documented NC) and issues MDIO writes:

  phase A: BMCR=0x8000 (soft reset) to PHY addresses 0..31
  phase B: BMCR=0x3000 (autoneg, 100M advertised) to 0..31, repeated

MDIO writes need no response, so they work even if reads are dead.
Observable: the PC NIC carrier (ASIX) / the driver's discovery.

UART heartbeat: AA <phase> <addr> <frames lo> <frames hi> 55 @115200.
"""

import os

from amaranth            import *
from amaranth.build      import Attrs, Pins, Resource, Subsignal
from amaranth.lib.cdc    import FFSynchronizer

from luna import top_level_cli

_uart_dbg = Resource("uart_dbg", 0,
    Subsignal("tx", Pins("J_2:34", dir="o"),
              Attrs(io_standard="3.3-V LVCMOS")))
_phy_nrst = Resource("phy_nrst", 0,
    Subsignal("o", Pins("J_2:22", dir="o"),
              Attrs(io_standard="3.3-V LVCMOS")))


class PhyResetProbe(Elaboratable):
    def elaborate(self, platform):
        m = Module()
        clk = platform.request(platform.default_clk)
        m.domains.sys = ClockDomain()
        m.d.comb += ClockSignal("sys").eq(clk.i)

        platform.add_resources([_uart_dbg, _phy_nrst])
        uart = platform.request("uart_dbg", 0)
        nrst = platform.request("phy_nrst", 0)
        rmii = platform.request("rmii", 0,
                                dir={"ref_clk": "o", "mdio": "io"})

        m.d.comb += [
            nrst.o.eq(1),
            rmii.ref_clk.o.eq(clk.i),
        ]

        # ---------------- MDIO write master ----------------
        DIV = 10
        div_cnt = Signal(range(DIV))
        mdc = Signal(reset=1)
        mdc_q = Signal()
        falling = Signal()
        rising = Signal()
        m.d.comb += [
            mdc.eq(div_cnt < DIV // 2),
            falling.eq(mdc_q & ~mdc),
            rising.eq(~mdc_q & mdc),
            rmii.mdc.o.eq(mdc),
        ]
        with m.If(div_cnt == DIV - 1):
            m.d.sys += div_cnt.eq(0)
        with m.Else():
            m.d.sys += div_cnt.eq(div_cnt + 1)
        m.d.sys += mdc_q.eq(mdc)

        bits = Signal(64)
        bit_cnt = Signal(6)
        sending = Signal()
        gap = Signal(20)
        frames = Signal(16)

        # scheduler: phase 0 = reset 0x8000 to 0..31,
        #            phase 1 = autoneg 0x3000 to 0..31, repeat
        phase = Signal(1)
        addr = Signal(5)
        data = Signal(16)
        gap_full = Signal()

        m.d.comb += [
            rmii.mdio.o.eq(0),
            rmii.mdio.oe.eq(sending & ~bits[63]),
            data.eq(Mux(phase == 0, 0x8000, 0x3000)),
            gap_full.eq(~sending & (gap == (1 << 18) - 1)),
        ]

        with m.If(sending):
            with m.If(falling):
                with m.If(bit_cnt == 63):
                    m.d.sys += [
                        sending.eq(0),
                        frames.eq(frames + 1),
                        gap.eq(0),
                    ]
                with m.Else():
                    m.d.sys += [
                        bits.eq(bits << 1),
                        bit_cnt.eq(bit_cnt + 1),
                    ]
        with m.Else():
            with m.If(gap_full):
                m.d.sys += gap.eq(0)
                with m.If(addr == 31):
                    m.d.sys += [addr.eq(0), phase.eq(1)]
                with m.Else():
                    m.d.sys += addr.eq(addr + 1)
            with m.Else():
                m.d.sys += gap.eq(gap + 1)

        # load the frame for the current addr (same edge as gap_full; the
        # addr update above only affects the NEXT frame)
        with m.If(gap_full):
            m.d.sys += [
                bits.eq((Const(0xFFFFFFFF, 32) << 32) | (0b01 << 30)
                        | (0b01 << 28) | (addr << 23) | (0b10 << 16)
                        | data),
                bit_cnt.eq(0),
                sending.eq(1),
            ]

        # ---------------- UART heartbeat ----------------
        DIVU = 434
        u_data = Signal(8)
        u_start = Signal()
        u_busy = Signal()
        u_phase = Signal(range(DIVU + 1))
        u_bit = Signal(4)
        u_shift = Signal(8)
        txd = Signal(reset=1)
        m.d.comb += uart.tx.eq(txd)
        with m.If(u_busy):
            with m.If(u_phase == DIVU - 1):
                m.d.sys += u_phase.eq(0)
                with m.If(u_bit == 9):
                    m.d.sys += [u_busy.eq(0), txd.eq(1)]
                with m.Elif(u_bit == 8):
                    m.d.sys += [u_bit.eq(9), txd.eq(1)]
                with m.Else():
                    m.d.sys += [u_bit.eq(u_bit + 1), txd.eq(u_shift[0]),
                                u_shift.eq(u_shift >> 1)]
            with m.Else():
                m.d.sys += u_phase.eq(u_phase + 1)
        with m.Else():
            with m.If(u_start):
                m.d.sys += [u_busy.eq(1), u_shift.eq(u_data),
                            u_bit.eq(0), u_phase.eq(0), txd.eq(0)]

        period = 5_000_000
        timer = Signal(range(period))
        fire = Signal()
        idx = Signal(4)
        dumping = Signal()
        m.d.sys += fire.eq(0)
        with m.If(timer == period - 1):
            m.d.sys += [timer.eq(0), fire.eq(1)]
        with m.Else():
            m.d.sys += timer.eq(timer + 1)
        with m.If(fire):
            m.d.sys += [dumping.eq(1), idx.eq(0)]
        with m.Elif(dumping & ~u_busy):
            with m.If(idx == 5):
                m.d.sys += dumping.eq(0)
            with m.Else():
                m.d.sys += idx.eq(idx + 1)
        m.d.comb += [
            u_start.eq(fire | (dumping & ~u_busy)),
            u_data.eq(Array([0xAA,
                             Cat(phase, 0, 0, 0, 0, 0, 0, 0)[0:8],
                             Cat(addr, 0, 0, 0)[0:8],
                             frames[0:8], frames[8:16],
                             0x55])[idx]),
        ]
        return m


if __name__ == "__main__":
    os.environ.setdefault("LUNA_PLATFORM",
                          "platforms:ADATFaceCycloneIV_EP4CGX150")
    top_level_cli(PhyResetProbe)
