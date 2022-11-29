from amaranth import Elaboratable, Module, Signal, Array, Memory, signed, Cat, Mux, Instance, ClockSignal, ResetSignal
from amaranth.compat.fhdl.specials import Tristate
import os
import subprocess
import shutil
from amaranth_boards.resources import SDRAMResource, UARTResource
from utils import request_bare
class LiteXSoC(Elaboratable):
    def __init__(self, sample_width):
        self._led = Signal()
        self._i2s_rx = Signal(sample_width)
        self._i2s_tx = Signal(sample_width)
        self._lrclk = Signal()


    def elaborate(self, platform) -> Module:
        m = Module()

        platform.add_file("VexRiscv.v", open("VexRiscv.v"))
        #platform.add_file("VexRiscvLitexSmpCluster_Cc2_Iw64Is8192Iy2_Dw64Ds8192Dy2_ITs4DTs4_Ldw16_Cdma_Ood.v", open("VexRiscvLitexSmpCluster_Cc2_Iw64Is8192Iy2_Dw64Ds8192Dy2_ITs4DTs4_Ldw16_Cdma_Ood.v"))
        platform.add_file("qmtech_5cefa2.v", open("qmtech_5cefa2.v"))
        platform.add_file("qmtech_5cefa2_mem.init", open("qmtech_5cefa2_mem.init"))
        platform.add_file("qmtech_5cefa2_rom.init", open("qmtech_5cefa2_rom.init"))
        platform.add_file("qmtech_5cefa2_sram.init", open("qmtech_5cefa2_sram.init"))

        #/ home / rouven / Computer / Coden / Audio / fpga / pythondata - cpu - vexriscv / pythondata_cpu_vexriscv / verilog / VexRiscv.v
        #sdram_resource = (SDRAMResource)(platform.request("sdram"))
        #uart_resource = (UARTResource())(platform.request("uart"))
        sdram_resource = request_bare(platform, "sdram", 0)
        uart_resource = platform.request("uart", 0)
        uartbone_resource = platform.request("uart", 1)
        
        #module qmtech_5cefa2 (
        #    input  wire clk50,
        #    output wire sdram_clock,
        #    output reg  serial_tx,
        #    input  wire serial_rx,
        #    output wire [12:0] sdram_a,
        #    output wire [1:0] sdram_ba,
        #    output wire sdram_cs_n,
        #    output wire sdram_cke,
        #    output wire sdram_ras_n,
        #    output wire sdram_cas_n,
        #    output wire sdram_we_n,
        #    inout  wire [15:0] sdram_dq,
        #    output wire [1:0] sdram_dm,
        #    output reg  user_led0,
        #    input  wire i2s_rx_rx,
        #    input  wire i2s_rx_clk,
        #    input  wire i2s_rx_sync,
        #    output wire i2s_tx_tx,
        #    input  wire i2s_tx_clk,
        #    input  wire i2s_tx_sync,
        #    output reg  serial_debug_tx,
        #    input  wire serial_debug_rx
        #);

        #dq = Signal(16)
        #m.d.comb += dq.eq(sdram_resource.dq.i)
        #m.d.comb += sdram_resource.dq.o.eq(dq.o)

        #dq = Tristate(sdram_resource.dq.o, sdram_resource.dq.oe, sdram_resource.dq.i)
        m.submodules.litex_soc = Instance("qmtech_5cefa2",
            i_clk50 = ClockSignal("clk50"),
            o_sdram_clock = sdram_resource.clk,
            o_serial_tx=uart_resource.tx,
            i_serial_rx=uart_resource.rx,
            o_sdram_a=sdram_resource.a,
            o_sdram_ba=sdram_resource.ba,
            o_sdram_cs_n=sdram_resource.cs,
            o_sdram_cke=sdram_resource.clk_en,
            o_sdram_ras_n=sdram_resource.ras,
            o_sdram_cas_n=sdram_resource.cas,
            o_sdram_we_n=sdram_resource.we,
            io_sdram_dq=sdram_resource.dq,
            o_sdram_dm=sdram_resource.dqm,
            o_user_led0=self._led,
            i_i2s_rx_rx=self._i2s_rx,
            i_i2s_rx_clk=~ClockSignal("dac"),
            i_i2s_rx_sync=self._lrclk,
            o_i2s_tx_tx=self._i2s_tx,
            i_i2s_tx_clk=~ClockSignal("dac"),
            i_i2s_tx_sync=self._lrclk,
            o_serial_debug_tx=uartbone_resource.tx,
            i_serial_debug_rx=uartbone_resource.rx
        )


        return m
