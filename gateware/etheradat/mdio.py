"""Minimal MDIO master (clause 22 read-only) for LAN8720 link status.

The bridge does not configure the PHY over MDIO: the LAN8720 runs on
its strap defaults (auto-negotiation, all speeds). MDIO is used only to
probe the PHY address (read PHY Identifier 1 = 0x0007 for the LAN8720
at the strap-selected address) and poll the BMSR link bit afterwards.

Runs in the RMII reference clock domain. MDC is a divided output;
MDIO is open-drain (drive low or release; the module's pull-up
provides the high level).

Read frame: 32x1 preamble, 01 start, 10 read-op, PHYAD[4:0],
REGAD[4:0], then 2 turnaround cycles (MDIO released) and 16 sampled
data bits (MSB first, sampled on rising MDC edges).
"""

from amaranth import Elaboratable, Module, Signal, Array, Mux

LAN8720_PHY_ID1 = 0x0007

REG_BMCR = 0
REG_BMSR = 1
REG_PHYID1 = 2

S_LOAD = 0
S_DRIVE = 1
S_TA = 2
S_DATA = 3
S_INTERP = 4
S_PAUSE = 5


def _width_for(max_value):
    return max(1, max_value.bit_length())


class MDIOMaster(Elaboratable):
    """Read-only MDIO master: probes the PHY address, polls BMSR.

    Attributes
    ----------
    mdc_out       : MDC clock output
    mdio_o        : MDIO data out (constant 0; oe gates the drive)
    mdio_oe       : MDIO output enable (active high)
    mdio_i        : MDIO data in
    link_up_out   : BMSR bit 2 of the responding PHY
    phy_addr_out  : probed PHY address, valid when phy_found_out
    phy_found_out : level: a PHY answered the identifier probe
    """

    def __init__(self, *, mdc_divider=10, poll_cycles=1 << 22,
                 probe_addrs=(0, 1)):
        self.mdc_divider = mdc_divider
        self.poll_cycles = poll_cycles
        self.probe_addrs = tuple(probe_addrs)

        self.mdc_out = Signal(reset=1)
        self.mdio_o = Signal(reset=0)
        self.mdio_oe = Signal(reset=0)
        self.mdio_i = Signal()
        self.link_up_out = Signal()
        self.phy_addr_out = Signal(5)
        self.phy_found_out = Signal()

    def elaborate(self, platform):
        m = Module()
        sync = m.d.sync
        comb = m.d.comb

        half = self.mdc_divider // 2

        # MDC generator: high during the first half of each divider cycle
        div_cnt = Signal(_width_for(self.mdc_divider - 1))
        mdc = Signal()
        mdc_q = Signal()
        falling = Signal()
        rising = Signal()
        comb += [
            mdc.eq(div_cnt < half),
            self.mdc_out.eq(mdc),
            falling.eq(mdc_q & ~mdc),
            rising.eq(~mdc_q & mdc),
        ]
        sync += mdc_q.eq(mdc)
        with m.If(div_cnt == self.mdc_divider - 1):
            sync += div_cnt.eq(0)
        with m.Else():
            sync += div_cnt.eq(div_cnt + 1)

        # sequencer
        state = Signal(3, reset=S_LOAD)
        op_bits = Signal(44)               # driven bits, MSB first
        op_cnt = Signal(_width_for(43))
        ta_cnt = Signal(2)
        data_shift = Signal(16)
        data_cnt = Signal(_width_for(15))

        probe_idx = Signal(_width_for(len(self.probe_addrs) - 1))
        found = Signal()
        phy_addr = Signal(5)
        reg_addr = Signal(5, reset=REG_PHYID1)
        poll_timer = Signal(_width_for(self.poll_cycles - 1))

        # probe addresses as constants, selected by probe_idx
        probe_addr_value = Array(
            [self.probe_addrs[min(k, len(self.probe_addrs) - 1)]
             for k in range(len(self.probe_addrs))])[probe_idx]

        comb += [
            self.phy_found_out.eq(found),
            self.phy_addr_out.eq(phy_addr),
        ]

        # MDIO open-drain drive: 1 = release (pull-up provides the level)
        comb += [
            self.mdio_o.eq(0),
            self.mdio_oe.eq((state == S_DRIVE) & ~op_bits[43]),
        ]

        # frame shift register, MSB first
        frame_value = (0xFFFFFFFF << 12) | (0b01 << 10) | (0b10 << 8)
        with m.If(falling):
            with m.If(state == S_DRIVE):
                sync += [
                    op_bits.eq(op_bits << 1),
                    op_cnt.eq(op_cnt + 1),
                ]
                with m.If(op_cnt == 43):
                    sync += [
                        state.eq(S_TA),
                        ta_cnt.eq(0),
                    ]
            with m.Elif(state == S_TA):
                sync += ta_cnt.eq(ta_cnt + 1)
                with m.If(ta_cnt == 1):
                    sync += [
                        state.eq(S_DATA),
                        data_cnt.eq(0),
                        data_shift.eq(0),
                    ]

        with m.If(rising & (state == S_DATA)):
            sync += [
                data_shift.eq((data_shift << 1) | self.mdio_i),
                data_cnt.eq(data_cnt + 1),
            ]
            with m.If(data_cnt == 15):
                sync += state.eq(S_INTERP)

        # load a new read frame
        addr_for_frame = Mux(found, phy_addr, probe_addr_value)
        sync += state.eq(state)
        with m.If(state == S_LOAD):
            sync += [
                op_bits.eq((frame_value << 0) | (addr_for_frame << 5)
                           | reg_addr),
                op_cnt.eq(0),
                state.eq(S_DRIVE),
            ]

        # interpret one register read
        with m.If(state == S_INTERP):
            with m.If(~found):
                with m.If(data_shift == LAN8720_PHY_ID1):
                    sync += [
                        found.eq(1),
                        phy_addr.eq(probe_addr_value),
                        reg_addr.eq(REG_BMSR),
                        state.eq(S_PAUSE),
                    ]
                with m.Else():
                    # next probe address, wrapping
                    sync += state.eq(S_LOAD)
                    with m.If(probe_idx == len(self.probe_addrs) - 1):
                        sync += probe_idx.eq(0)
                    with m.Else():
                        sync += probe_idx.eq(probe_idx + 1)
            with m.Else():
                sync += [
                    self.link_up_out.eq(data_shift[2]),
                    state.eq(S_PAUSE),
                ]

        # pause between register reads
        with m.If(state == S_PAUSE):
            with m.If(poll_timer == self.poll_cycles - 1):
                sync += [
                    poll_timer.eq(0),
                    state.eq(S_LOAD),
                ]
            with m.Else():
                sync += poll_timer.eq(poll_timer + 1)

        return m
