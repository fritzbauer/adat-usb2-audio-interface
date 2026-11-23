"""Tiny UART transmitter + periodic status-dump sequencer.

NOT used by the EtherADAT top (23.11.2026): although this block
simulates correctly, it produced no activity on the FPGA in several
builds (both in a raw clk50 domain and in the PLL sync domain), while
the identical inline FSM/dump used by clock_probe.py and the top works
on the same hardware. The top therefore carries an inline copy of the
probe-style dump. Kept here as reference until the difference is
understood.

The debug UART runs from the board's 50 MHz oscillator, not from the
PHY's REF_CLK, so it stays readable when the PHY/RMII side is dead
(that is exactly the failure mode this dump must diagnose).

Dump format (115200 8N1), one dump every ``period`` seconds:

    AA 01 <v0 lo> <v0 hi> <v1 lo> <v1 hi> ... 5A

All values are 16-bit little-endian. The caller supplies the value
signals, already registered in the same clock domain as this module;
the top-level synchronizes/counts events into that domain.
"""

from amaranth import Elaboratable, Module, Signal, Array, Mux


class UartTX(Elaboratable):
    """8N1 UART transmitter, one byte per start strobe."""

    def __init__(self, clk_freq=50_000_000, baud=115_200):
        self.clk_freq = clk_freq
        self.baud = baud
        self.data = Signal(8)
        self.start = Signal()
        self.busy = Signal()
        self.tx = Signal(reset=1)

    def elaborate(self, platform):
        m = Module()
        sync = m.d.sync

        div = max(1, round(self.clk_freq / self.baud))
        phase = Signal(range(div + 1))
        bit_i = Signal(4)
        shift = Signal(8)

        sync += self.tx.eq(1)
        with m.If(self.busy):
            with m.If(phase == div - 1):
                sync += phase.eq(0)
                with m.If(bit_i == 9):
                    sync += [self.busy.eq(0), self.tx.eq(1)]
                with m.Elif(bit_i == 8):
                    sync += [bit_i.eq(9), self.tx.eq(1)]      # stop bit
                with m.Else():
                    sync += [bit_i.eq(bit_i + 1), self.tx.eq(shift[0]),
                             shift.eq(shift >> 1)]
            with m.Else():
                sync += phase.eq(phase + 1)
        with m.Else():
            with m.If(self.start):
                sync += [self.busy.eq(1), shift.eq(self.data),
                         bit_i.eq(0), phase.eq(0), self.tx.eq(0)]
        return m


class PeriodicDump(Elaboratable):
    """Sends the magic/values/terminator dump every ``period`` seconds."""

    def __init__(self, values, clk_freq=50_000_000, baud=115_200,
                 period=0.5):
        self.values = list(values)
        self.clk_freq = clk_freq
        self.baud = baud
        self.period = period
        self.uart = UartTX(clk_freq, baud)
        self.tx = self.uart.tx
        self.dumps = Signal(16)

    def elaborate(self, platform):
        m = Module()
        sync = m.d.sync
        m.submodules.uart = uart = self.uart

        n = len(self.values)
        total = 2 + 2 * n + 1
        period_cycles = max(2, int(self.clk_freq * self.period))

        timer = Signal(range(period_cycles))
        fire = Signal()
        idx = Signal(range(max(total, 2)))

        sync += fire.eq(0)
        with m.If(timer == period_cycles - 1):
            sync += [timer.eq(0), fire.eq(1)]
        with m.Else():
            sync += timer.eq(timer + 1)

        val_bytes = []
        for v in self.values:
            val_bytes += [v[0:8], v[8:16]]
        byte_table = [0xAA, 0x01] + val_bytes + [0x5A]

        m.d.comb += [
            uart.data.eq(Array(byte_table)[Mux(idx < total, idx, 0)]),
        ]

        # sequencer: IDLE --fire--> LAUNCH --accepted--> WAIT --done--> ...
        S_IDLE, S_LAUNCH, S_WAIT = range(3)
        state = Signal(2, reset=S_IDLE)
        m.d.comb += uart.start.eq(state == S_LAUNCH)
        with m.Switch(state):
            with m.Case(S_IDLE):
                with m.If(fire):
                    sync += [idx.eq(0), state.eq(S_LAUNCH),
                             self.dumps.eq(self.dumps + 1)]
            with m.Case(S_LAUNCH):
                with m.If(~uart.busy):
                    sync += state.eq(S_WAIT)
            with m.Case(S_WAIT):
                with m.If(~uart.busy):
                    with m.If(idx == total - 1):
                        sync += state.eq(S_IDLE)
                    with m.Else():
                        sync += [idx.eq(idx + 1), state.eq(S_LAUNCH)]
        return m
