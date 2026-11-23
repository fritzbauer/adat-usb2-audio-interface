#!/usr/bin/env python3
"""Mixed-length MAC test: long audio frame + short hello, back-to-back.

Uses the new MAC-only blackbox (liteeth_etheradat_mac) with a minimal,
bridge-free frame generator that sends one static AdatNet hello frame
(EtherType 0x88B6, 34 bytes, broadcast) every 250 ms. The PC kernel
driver (mode=1, discovery) must lock onto it and count hello_rx.

This isolates the MAC-only wrapper + hello framing from the bridge.
UART (115200, board 50 MHz): AA 01 <sent u16> <state> <idx> <errors> 5A.

Build: python etheradat/scripts/generate_quartus_files_hellotest.py <dir>
"""

import os

from amaranth            import *
from amaranth.build      import Attrs, Pins, Resource, Subsignal

from luna import top_level_cli

_uart_dbg = Resource("uart_dbg", 0,
    Subsignal("tx", Pins("J_2:34", dir="o"),
              Attrs(io_standard="3.3-V LVCMOS")))

OWN_MAC = 0x02AD6E657401
HELLO = list(bytes.fromhex(
    "ffffffffffff" "02ad6e657401" "88b6" "41444154" "01" "00"
    "0100" "46414441" "00000000" "0f00" "0000"))
LONG = list(bytes.fromhex(
    "ffffffffffff" "02ad6e657401" "88b5" "a500" "00000000" "a805")) \
    + [(i * 37) & 0xFF for i in range(1462 - 22)]
assert len(HELLO) == 34 and len(LONG) == 1462


class MacMixedTest(Elaboratable):
    def elaborate(self, platform):
        m = Module()

        clk = platform.request(platform.default_clk)
        m.domains.sys = ClockDomain()
        m.d.comb += ClockSignal("sys").eq(clk.i)

        platform.add_resources([_uart_dbg])
        uart = platform.request("uart_dbg", 0)
        leds = platform.request("leds")

        rmii = platform.request("rmii", 0, dir={"ref_clk": "i"})
        rmii_clk = rmii.ref_clk.i
        m.domains.rmii = ClockDomain()
        m.d.comb += ClockSignal("rmii").eq(rmii_clk)
        tx_clk = Signal()
        m.d.comb += tx_clk.eq(~rmii_clk)

        mac_tx_data = Signal(8)
        mac_tx_valid = Signal()
        mac_tx_ready = Signal()
        mac_tx_last = Signal()
        mac_rx_data = Signal(8)
        mac_rx_valid = Signal()
        mac_rx_last = Signal()
        mac_rx_error = Signal()

        _mac_v = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "liteeth", "liteeth_etheradat_mac.v")
        platform.add_file("liteeth_etheradat_mac.v", open(_mac_v, "rb"))
        m.submodules.liteeth_mac = Instance("liteeth_etheradat_mac",
            i_sys_clk        = rmii_clk,
            i_sys_rst        = 0,
            i_eth_tx_clk     = rmii_clk,
            i_eth_tx_rst     = 0,
            i_eth_rx_clk     = rmii_clk,
            i_eth_rx_rst     = 0,
            o_rmii_tx_en     = rmii.tx_en.o,
            o_rmii_txd       = Cat(rmii.txd0.o, rmii.txd1.o),
            i_rmii_crs_dv    = rmii.crs_dv.i,
            i_rmii_rxd       = Cat(rmii.rxd0.i, rmii.rxd1.i),
            i_user_sink_payload_data = mac_tx_data,
            i_user_sink_valid        = mac_tx_valid,
            i_user_sink_last         = mac_tx_last,
            o_user_sink_ready        = mac_tx_ready,
            o_user_source_payload_data = mac_rx_data,
            o_user_source_valid        = mac_rx_valid,
            o_user_source_last         = mac_rx_last,
            o_user_source_payload_error = mac_rx_error,
            i_user_source_ready        = 1,
        )

        # ---- mixed frame generator (rmii domain) ----
        # state 0: idle, 1: long frame, 2: hello, 3: user-side IFG
        period = 25_000
        timer = Signal(range(period))
        due = Signal()
        state = Signal(2)
        idx = Signal(11)
        sent = Signal(16)
        ifg = Signal(5)
        m.d.rmii += due.eq(0)
        with m.If(timer == period - 1):
            m.d.rmii += [timer.eq(0), due.eq(1)]
        with m.Else():
            m.d.rmii += timer.eq(timer + 1)

        frame_len = Mux(state == 1, len(LONG), len(HELLO))
        frame_data = Signal(8)
        with m.Switch(state):
            with m.Case(1):
                m.d.comb += frame_data.eq(Array(LONG)[idx])
            with m.Case(2):
                m.d.comb += frame_data.eq(Array(HELLO)[idx])
            with m.Default():
                m.d.comb += frame_data.eq(0)

        with m.If(state == 0):
            with m.If(due):
                m.d.rmii += [state.eq(1), idx.eq(0)]
        with m.Elif(state == 3):
            with m.If(ifg == 0):
                m.d.rmii += [state.eq(2), idx.eq(0)]
            with m.Else():
                m.d.rmii += ifg.eq(ifg - 1)
        with m.Else():
            with m.If(mac_tx_ready):
                with m.If(idx == frame_len - 1):
                    with m.If(state == 1):
                        m.d.rmii += [state.eq(3), ifg.eq(15)]
                    with m.Else():
                        m.d.rmii += [state.eq(0), sent.eq(sent + 1)]
                with m.Else():
                    m.d.rmii += idx.eq(idx + 1)

        m.d.comb += [
            mac_tx_valid.eq((state == 1) | (state == 2)),
            mac_tx_data.eq(frame_data),
            mac_tx_last.eq(((state == 1) | (state == 2))
                           & (idx == frame_len - 1)),
        ]

        # ---- UART heartbeat (sys/clk50) ----
        sent_s = Signal(16)
        state_s = Signal()
        idx_s = Signal(6)
        from amaranth.lib.cdc import FFSynchronizer
        m.submodules += [
            FFSynchronizer(sent, sent_s, o_domain="sys"),
            FFSynchronizer(state, state_s, o_domain="sys"),
            FFSynchronizer(idx, idx_s, o_domain="sys"),
        ]

        DIV = 434
        data = Signal(8)
        start = Signal()
        busy = Signal()
        phase = Signal(range(DIV + 1))
        bit_i = Signal(4)
        shift = Signal(8)
        txd = Signal(reset=1)
        m.d.comb += uart.tx.eq(txd)
        with m.If(busy):
            with m.If(phase == DIV - 1):
                m.d.sys += phase.eq(0)
                with m.If(bit_i == 9):
                    m.d.sys += [busy.eq(0), txd.eq(1)]
                with m.Elif(bit_i == 8):
                    m.d.sys += [bit_i.eq(9), txd.eq(1)]
                with m.Else():
                    m.d.sys += [bit_i.eq(bit_i + 1), txd.eq(shift[0]),
                                shift.eq(shift >> 1)]
            with m.Else():
                m.d.sys += phase.eq(phase + 1)
        with m.Else():
            with m.If(start):
                m.d.sys += [busy.eq(1), shift.eq(data), bit_i.eq(0),
                            phase.eq(0), txd.eq(0)]

        ptimer = Signal(23)
        pfire = Signal()
        pidx = Signal(3)
        pdump = Signal()
        m.d.sys += pfire.eq(0)
        with m.If(ptimer == (1 << 22) - 1):
            m.d.sys += [ptimer.eq(0), pfire.eq(1)]
        with m.Else():
            m.d.sys += ptimer.eq(ptimer + 1)
        with m.If(pfire):
            m.d.sys += [pdump.eq(1), pidx.eq(0)]
        with m.Elif(pdump & ~busy):
            with m.If(pidx == 6):
                m.d.sys += pdump.eq(0)
            with m.Else():
                m.d.sys += pidx.eq(pidx + 1)
        m.d.comb += [
            start.eq(pfire | (pdump & ~busy)),
            data.eq(Array([0xAA, 0x01,
                           sent_s[0:8], sent_s[8:16],
                           Cat(state_s, 0, 0, 0, 0, 0, 0, 0)[0:8],
                           Cat(idx_s, 0, 0)[0:8],
                           0x5A])[pidx]),
        ]

        m.d.comb += [
            leds.host.eq(sent_s[3]),
            leds.usb1.eq(state_s),
            leds.usb2.eq(rmii.crs_dv.i),
        ]
        return m


if __name__ == "__main__":
    os.environ.setdefault("LUNA_PLATFORM",
                          "platforms:ADATFaceCycloneIV_EP4CGX150")
    top_level_cli(MacMixedTest)
