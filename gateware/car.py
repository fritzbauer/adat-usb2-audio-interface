from amaranth         import *
from amaranth.build   import *
from amaranth.lib.cdc import ResetSynchronizer


class ClockDomainGeneratorBase():
    NO_PHASE_SHIFT  = 0

    def wire_up_reset(self, m, reset):
        m.submodules.reset_sync_fast = ResetSynchronizer(reset, domain="fast")
        m.submodules.reset_sync_usb  = ResetSynchronizer(reset, domain="usb")
        m.submodules.reset_sync_sync = ResetSynchronizer(reset, domain="sync")
        m.submodules.reset_sync_dac  = ResetSynchronizer(reset, domain="dac")
        m.submodules.reset_sync_adat = ResetSynchronizer(reset, domain="adat")

class IntelFPGAClockDomainGenerator(Elaboratable, ClockDomainGeneratorBase):
    ADAT_DIV_48k    = 83
    ADAT_MULT_48k   = 17

    ADAT_DIV_44_1k  = 62
    ADAT_MULT_44_1k = 14

    DUTY_CYCLE      = 50

    def __init__(self, *, clock_frequencies=None, clock_signal_name=None):
        pass

    def elaborate(self, platform):
        m = Module()

        # Create our domains
        m.domains.usb  = ClockDomain("usb")
        m.domains.sync = ClockDomain("sync")
        m.domains.fast = ClockDomain("fast")
        m.domains.adat = ClockDomain("adat")
        m.domains.dac  = ClockDomain("dac")

        clk = platform.request(platform.default_clk)

        main_clocks    = Signal(5)
        audio_clocks   = Signal(4)
        fast_clock_48k = Signal()

        sys_locked   = Signal()
        audio_locked = Signal()
        fast_locked  = Signal()
        reset        = Signal()

        m.submodules.mainpll = Instance("ALTPLL",
            p_BANDWIDTH_TYPE         = "AUTO",

            # USB clock: 60MHz
            p_CLK0_DIVIDE_BY         = 5,
            p_CLK0_DUTY_CYCLE        = self.DUTY_CYCLE,
            p_CLK0_MULTIPLY_BY       = 6,
            p_CLK0_PHASE_SHIFT       = self.NO_PHASE_SHIFT,

            # 44.1k ADAT Clock 11.2896 MHz = 44.1kHz * 256
            p_CLK1_DIVIDE_BY         = self.ADAT_DIV_44_1k,
            p_CLK1_DUTY_CYCLE        = self.DUTY_CYCLE,
            p_CLK1_MULTIPLY_BY       = self.ADAT_MULT_44_1k,
            p_CLK1_PHASE_SHIFT       = self.NO_PHASE_SHIFT,

            # I2S DAC clock 44.1k = 3.072 MHz = 44.1kHz * 32 bit * 2 channels
            p_CLK2_DIVIDE_BY         = self.ADAT_DIV_44_1k * 4,
            p_CLK2_DUTY_CYCLE        = self.DUTY_CYCLE,
            p_CLK2_MULTIPLY_BY       = self.ADAT_MULT_44_1k,
            p_CLK2_PHASE_SHIFT       = self.NO_PHASE_SHIFT,

            # ADAT sampling clock = 44.1kHz * 256 * 8 times oversampling
            p_CLK3_DIVIDE_BY         = self.ADAT_DIV_44_1k,
            p_CLK3_DUTY_CYCLE        = self.DUTY_CYCLE,
            p_CLK3_MULTIPLY_BY       = self.ADAT_MULT_44_1k * 8,
            p_CLK3_PHASE_SHIFT       = self.NO_PHASE_SHIFT,

            # ADAT transmit domain clock = 44.1kHz * 5 output terminals
            p_CLK4_DIVIDE_BY         = self.ADAT_DIV_44_1k,
            p_CLK4_DUTY_CYCLE        = self.DUTY_CYCLE,
            p_CLK4_MULTIPLY_BY       = self.ADAT_MULT_44_1k * 5,
            p_CLK4_PHASE_SHIFT       = self.NO_PHASE_SHIFT,

            # 50MHz = 20000 picoseconds
            p_INCLK0_INPUT_FREQUENCY = 20000,
            p_OPERATION_MODE         = "NORMAL",

            # Drive our clock from the USB clock
            # coming from the USB clock pin of the USB3300
            i_inclk  = clk.i,
            o_clk    = main_clocks,
            o_locked = sys_locked,
        )

        m.submodules.audiopll = Instance("ALTPLL",
            p_BANDWIDTH_TYPE         = "AUTO",

            # ADAT clock = 12.288 MHz = 48 kHz * 256
            p_CLK0_DIVIDE_BY         = self.ADAT_DIV_48k,
            p_CLK0_DUTY_CYCLE        = self.DUTY_CYCLE,
            p_CLK0_MULTIPLY_BY       = self.ADAT_MULT_48k,
            p_CLK0_PHASE_SHIFT       = self.NO_PHASE_SHIFT,

            # I2S DAC clock 48k = 3.072 MHz = 48 kHz * 32 bit * 2 channels
            p_CLK1_DIVIDE_BY         = self.ADAT_DIV_48k * 4,
            p_CLK1_DUTY_CYCLE        = self.DUTY_CYCLE,
            p_CLK1_MULTIPLY_BY       = self.ADAT_MULT_48k,
            p_CLK1_PHASE_SHIFT       = self.NO_PHASE_SHIFT,

            # ADAT sampling clock = 48 kHz * 256 * 8 times oversampling
            p_CLK2_DIVIDE_BY         = self.ADAT_DIV_48k,
            p_CLK2_DUTY_CYCLE        = self.DUTY_CYCLE,
            p_CLK2_MULTIPLY_BY       = self.ADAT_MULT_48k * 8,
            p_CLK2_PHASE_SHIFT       = self.NO_PHASE_SHIFT,

            # ADAT transmit domain clock = 48 kHz * 256 * 5 output terminals
            p_CLK3_DIVIDE_BY         = self.ADAT_DIV_48k,
            p_CLK3_DUTY_CYCLE        = self.DUTY_CYCLE,
            p_CLK3_MULTIPLY_BY       = self.ADAT_MULT_48k * 5,
            p_CLK3_PHASE_SHIFT       = self.NO_PHASE_SHIFT,

            p_INCLK0_INPUT_FREQUENCY = 16667,
            p_OPERATION_MODE         = "NORMAL",

            i_inclk  = main_clocks[0],
            o_clk    = audio_clocks,
            o_locked = audio_locked,
        )

        m.submodules.fastpll = Instance("ALTPLL",
            p_BANDWIDTH_TYPE         = "AUTO",

            # ADAT sampling clock = 48 kHz * 256 * 8 times oversampling
            p_CLK0_DIVIDE_BY         = 1,
            p_CLK0_DUTY_CYCLE        = self.DUTY_CYCLE,
            p_CLK0_MULTIPLY_BY       = platform.fast_multiplier,
            p_CLK0_PHASE_SHIFT       = self.NO_PHASE_SHIFT,

            p_INCLK0_INPUT_FREQUENCY = 81395,
            p_OPERATION_MODE         = "NORMAL",

            i_inclk  = audio_clocks[0],
            o_clk    = fast_clock_48k,
            o_locked = fast_locked,
        )


        m.d.comb += [
            reset.eq(~(sys_locked & audio_locked & fast_locked)),
            ClockSignal("fast").eq(fast_clock_48k),
            ClockSignal("usb") .eq(main_clocks[0]),
            ClockSignal("adat").eq(audio_clocks[0]),
            ClockSignal("dac").eq(audio_clocks[1]),
            ClockSignal("sync").eq(audio_clocks[3]),
        ]

        self.wire_up_reset(m, reset)

        return m


class IntelCycloneVFPGAClockDomainGenerator(Elaboratable, ClockDomainGeneratorBase):

    def __init__(self, *, clock_frequencies=None, clock_signal_name=None):
        pass

    def elaborate(self, platform):
        m = Module()

        # Create our domains
        # usb: USB clock: 60MHz
        # adat: ADAT clock = 12.288 MHz = 48 kHz * 256
        # dac: I2S DAC clock 48k = 3.072 MHz = 48 kHz * 32 bit * 2 channels
        # sync: ADAT transmit domain clock = 61.44 MHz = 48 kHz * 256 * 5 output terminals
        # fast: ADAT sampling clock = 98.304 MHz = 48 kHz * 256 * 8 times oversampling
        m.domains.usb  = ClockDomain("usb")
        m.domains.sync = ClockDomain("sync")
        m.domains.fast = ClockDomain("fast")
        m.domains.adat = ClockDomain("adat")
        m.domains.dac  = ClockDomain("dac")

        clk = platform.request(platform.default_clk)

        main_clock    = Signal()
        audio_clocks  = Signal(4)

        sys_locked    = Signal()
        audio_locked  = Signal()

        reset         = Signal()

        m.submodules.mainpll = Instance("altera_pll",
            p_pll_type="General",
            p_pll_subtype="General",
            p_fractional_vco_multiplier="false",
            p_operation_mode="normal",
            p_reference_clock_frequency="50.0 MHz",
            p_number_of_clocks="1",

            p_output_clock_frequency0="60.000000 MHz",


            # Drive our clock from the internal 50MHz clock
            i_refclk = clk,
            o_outclk = main_clock,
            o_locked = sys_locked
        )

        m.submodules.audiopll = Instance("altera_pll",
            p_pll_type="General",
            p_pll_subtype="General",
            p_fractional_vco_multiplier="true",
            p_operation_mode="normal",
            p_reference_clock_frequency="60.0 MHz",
            p_number_of_clocks="4",

            p_output_clock_frequency0="12.288000 MHz",
            p_output_clock_frequency1="3.072000 MHz",
            p_output_clock_frequency2="61.440000 MHz",
            p_output_clock_frequency3="98.304000 MHz",

            # Drive our clock from the mainpll
            i_refclk=main_clock,
            o_outclk=audio_clocks,
            o_locked=audio_locked
        )

        m.d.comb += [
            reset.eq(~(sys_locked & audio_locked)),
            ClockSignal("usb") .eq(main_clock),
            ClockSignal("adat").eq(audio_clocks[0]),
            ClockSignal("dac").eq(audio_clocks[1]),
            ClockSignal("sync").eq(audio_clocks[2]),
            ClockSignal("fast").eq(audio_clocks[3])
        ]

        self.wire_up_reset(m, reset)

        return m

class Xilinx7SeriesClockDomainGenerator(Elaboratable, ClockDomainGeneratorBase):
    ADAT_DIV_48k    = 83
    ADAT_MULT_48k   = 17
    DUTY_CYCLE      = 0.5

    def __init__(self, *, clock_frequencies=None, clock_signal_name=None):
        pass

    def elaborate(self, platform):
        m = Module()

        # Create our domains
        m.domains.usb  = ClockDomain("usb")
        m.domains.sync = ClockDomain("sync")
        m.domains.fast = ClockDomain("fast")
        m.domains.adat = ClockDomain("adat")
        m.domains.dac  = ClockDomain("dac")

        clk = platform.request(platform.default_clk)

        main_clocks    = Signal()
        audio_clocks   = Signal(4)
        fast_clock_48k = Signal()

        sys_locked   = Signal()
        audio_locked = Signal()
        fast_locked  = Signal()
        reset        = Signal()

        mainpll_feedback  = Signal()
        audiopll_feedback = Signal()
        fastpll_feedback = Signal()

        m.submodules.mainpll = Instance("PLLE2_ADV",
            p_CLKIN1_PERIOD        = 20,
            p_BANDWIDTH            = "OPTIMIZED",
            p_COMPENSATION         = "ZHOLD",
            p_STARTUP_WAIT         = "FALSE",

            p_DIVCLK_DIVIDE        = 1,
            p_CLKFBOUT_MULT        = 30,
            p_CLKFBOUT_PHASE       = self.NO_PHASE_SHIFT,

            # 60MHz
            p_CLKOUT0_DIVIDE       = 25,
            p_CLKOUT0_PHASE        = self.NO_PHASE_SHIFT,
            p_CLKOUT0_DUTY_CYCLE   = self.DUTY_CYCLE,

            i_CLKFBIN              = mainpll_feedback,
            o_CLKFBOUT             = mainpll_feedback,
            i_CLKIN1               = clk,
            o_CLKOUT0              = main_clocks,
            o_LOCKED               = sys_locked,
        )

        m.submodules.audiopll = Instance("PLLE2_ADV",
            p_CLKIN1_PERIOD        = 16.666,
            p_BANDWIDTH            = "OPTIMIZED",
            p_COMPENSATION         = "ZHOLD",
            p_STARTUP_WAIT         = "FALSE",
            p_DIVCLK_DIVIDE        = 1,
            p_CLKFBOUT_MULT        = self.ADAT_MULT_48k,
            p_CLKFBOUT_PHASE       = self.NO_PHASE_SHIFT,

            # ADAT clock = 12.288 MHz = 48 kHz * 256
            p_CLKOUT2_DIVIDE       = self.ADAT_DIV_48k,
            p_CLKOUT2_PHASE        = self.NO_PHASE_SHIFT,
            p_CLKOUT2_DUTY_CYCLE   = self.DUTY_CYCLE,


            # ADAT sampling clock = 48 kHz * 256 * 8 times oversampling
            p_CLKOUT0_DIVIDE       = self.ADAT_DIV_48k / 8,
            p_CLKOUT0_PHASE        = self.NO_PHASE_SHIFT,
            p_CLKOUT0_DUTY_CYCLE   = self.DUTY_CYCLE,

            # ADAT transmit domain clock = 48 kHz * 256 * 5 output terminals
            p_CLKOUT3_DIVIDE       = self.ADAT_DIV_48k / 5,
            p_CLKOUT3_PHASE        = self.NO_PHASE_SHIFT,
            p_CLKOUT3_DUTY_CYCLE   = self.DUTY_CYCLE,

            i_CLKFBIN              = audiopll_feedback,
            o_CLKFBOUT             = audiopll_feedback,
            i_CLKIN1               = main_clocks[0],
            o_CLKOUT0              = audio_clocks[2],
            o_CLKOUT2              = audio_clocks[0],
            o_CLKOUT3              = audio_clocks[3],
            o_LOCKED               = audio_locked,
        )

        VCO_SCALER_FAST = 1

        m.submodules.fastpll = Instance("PLLE2_ADV",
            p_CLKIN1_PERIOD        = 10.172,
            p_BANDWIDTH            = "OPTIMIZED",
            p_COMPENSATION         = "ZHOLD",
            p_STARTUP_WAIT         = "FALSE",
            p_DIVCLK_DIVIDE        = 1,
            p_CLKFBOUT_MULT        = VCO_SCALER_FAST * platform.fast_multiplier,
            p_CLKFBOUT_PHASE       = self.NO_PHASE_SHIFT,

            # Fast clock = 48 kHz * 256 * 9
            p_CLKOUT0_DIVIDE       = VCO_SCALER_FAST,
            p_CLKOUT0_PHASE        = self.NO_PHASE_SHIFT,
            p_CLKOUT0_DUTY_CYCLE   = self.DUTY_CYCLE,

            # I2S DAC clock 48k = 3.072 MHz = 48 kHz * 32 bit * 2 channels
            p_CLKOUT1_DIVIDE       = VCO_SCALER_FAST * platform.fast_multiplier * 4,
            p_CLKOUT1_PHASE        = self.NO_PHASE_SHIFT,
            p_CLKOUT1_DUTY_CYCLE   = self.DUTY_CYCLE,

            i_CLKFBIN              = fastpll_feedback,
            o_CLKFBOUT             = fastpll_feedback,
            i_CLKIN1               = audio_clocks[2],
            o_CLKOUT0              = fast_clock_48k,
            o_CLKOUT1              = audio_clocks[1],
            o_LOCKED               = fast_locked,
        )


        m.d.comb += [
            reset.eq(~(sys_locked & audio_locked & fast_locked)),
            ClockSignal("fast").eq(fast_clock_48k),
            ClockSignal("usb") .eq(main_clocks[0]),
            ClockSignal("adat").eq(audio_clocks[0]),
            ClockSignal("dac").eq(audio_clocks[1]),
            ClockSignal("sync").eq(audio_clocks[3]),
        ]

        self.wire_up_reset(m, reset)

        return m

