from amaranth import *
from amaranth.build import *

from amaranth_boards.resources import *
from amaranth_boards.qmtech_ep4ce   import QMTechEP4CEPlatform
from amaranth_boards.qmtech_5cefa2  import QMTech5CEFA2Platform
from amaranth_boards.qmtech_10cl006 import QMTech10CL006Platform
#from amaranth_boards.qmtech_xc7a35t import QMTechXC7A35TPlatform
from amaranth_boards.qmtech_ep4cgx150 import QMTechEP4CGX150Platform

from luna.gateware.platform.core import LUNAPlatform

from car                     import IntelFPGAClockDomainGenerator, IntelCycloneVFPGAClockDomainGenerator, Xilinx7SeriesClockDomainGenerator
from adatface_rev0_baseboard import ADATFaceRev0Baseboard

class IntelFPGAParameters:
    QSF_ADDITIONS = r"""
            set_global_assignment -name OPTIMIZATION_MODE "Aggressive Performance"
            set_global_assignment -name FITTER_EFFORT "Standard Fit"
            set_global_assignment -name PHYSICAL_SYNTHESIS_EFFORT "Extra"
            set_instance_assignment -name DECREASE_INPUT_DELAY_TO_INPUT_REGISTER OFF -to *ulpi*
            set_instance_assignment -name INCREASE_DELAY_TO_OUTPUT_PIN OFF -to *ulpi*
            set_global_assignment -name NUM_PARALLEL_PROCESSORS ALL
        """

    SDC_ADDITIONS = r"""
            derive_pll_clocks
            derive_clock_uncertainty
            # sync clock domain crossing to ADAT clock domain crossing
            set_max_delay -from [get_clocks {car|audiopll|auto_generated|pll1|clk[3]}]  -to  [get_clocks {car|audiopll|auto_generated|pll1|clk[0]}] 5

            # USB to fast clock domain crossing
            set_max_delay -from [get_clocks {car|mainpll|auto_generated|pll1|clk[0]}]  -to  [get_clocks {car|fastopll|auto_generated|pll1|clk[0]}] 5

            # EtherADAT: the RMII domain runs on the raw 50 MHz oscillator
            # (clk50), the audio domains on the PLL outputs. All transfers
            # between them are gray-coded counters or FFSynchronizers, i.e.
            # asynchronous by design; without this the STA times the
            # transfers at the worst phase relationship and reports large
            # (harmless) setup violations on the CDC paths.
            set_clock_groups -asynchronous \
                -group [get_clocks {clk50*}] \
                -group [get_clocks {car|audiopll|* car|mainpll|* car|fastpll|*}]
        """

class ADATFaceCycloneV(QMTech5CEFA2Platform, LUNAPlatform):
    # NOTE: fast_multiplier must be 8, not 9: car.py's
    # IntelCycloneVFPGAClockDomainGenerator generates the fast domain at
    # 98.304 MHz = 48 kHz * 256 * 8 (same PLL VCO as the 12.288 MHz ADAT
    # bit clock, integer dividers 40:5, so the actual ratio is exactly
    # 8.000). The old value 9 declared 110.592 MHz which does not exist on
    # this board; ADATReceiver derives its bit timing from this number.
    fast_multiplier        = 8
    clock_domain_generator = IntelCycloneVFPGAClockDomainGenerator
    fast_domain_clock_freq = int(48e3 * 256 * fast_multiplier)

    @property
    def file_templates(self):
        templates = super().file_templates
        templates["{{name}}.qsf"] += IntelFPGAParameters.QSF_ADDITIONS
        templates["{{name}}.sdc"] += IntelFPGAParameters.SDC_ADDITIONS
        return templates

    def __init__(self):
        self.resources += ADATFaceRev0Baseboard.resources(Attrs(io_standard="3.3-V LVCMOS"))
        # swap connector numbers, because on ADATface the connector
        # names are swapped compared to the QMTech daughterboard
        self.connectors[0].number = 3
        self.connectors[1].number = 2
        super().__init__(standalone=False)

class ADATFaceCycloneIV(QMTechEP4CEPlatform, LUNAPlatform):
    fast_multiplier        = 9
    clock_domain_generator = IntelFPGAClockDomainGenerator
    fast_domain_clock_freq = int(48e3 * 256 * fast_multiplier)

    @property
    def file_templates(self):
        templates = super().file_templates
        templates["{{name}}.qsf"] += IntelFPGAParameters.QSF_ADDITIONS
        templates["{{name}}.sdc"] += IntelFPGAParameters.SDC_ADDITIONS
        return templates

    def __init__(self):
        self.resources += ADATFaceRev0Baseboard.resources(Attrs(io_standard="3.3-V LVCMOS"))
        # swap connector numbers, because on ADATface the connector
        # names are swapped compared to the QMTech daughterboard
        self.connectors[0].number = 3
        self.connectors[1].number = 2
        super().__init__(no_kluts=55, standalone=False)


class ADATFaceCycloneIV_EP4CGX150(QMTechEP4CGX150Platform, LUNAPlatform):
    fast_multiplier        = 9
    clock_domain_generator = IntelFPGAClockDomainGenerator
    fast_domain_clock_freq = int(48e3 * 256 * fast_multiplier)

    @property
    def file_templates(self):
        templates = super().file_templates
        templates["{{name}}.qsf"] += IntelFPGAParameters.QSF_ADDITIONS
        templates["{{name}}.sdc"] += IntelFPGAParameters.SDC_ADDITIONS
        return templates

    def __init__(self):
        self.resources += ADATFaceRev0Baseboard.resources(Attrs(io_standard="3.3-V LVCMOS"))
        # swap connector numbers, because on ADATface the connector
        # names are swapped compared to the QMTech daughterboard
        self.connectors[0].number = 3
        self.connectors[1].number = 2
        super().__init__(standalone=False)

# This is here just for experimental reasons.
# right now the design probably would not fit into this device anymore
class ADATFaceCyclone10(QMTech10CL006Platform, LUNAPlatform):
    clock_domain_generator = IntelFPGAClockDomainGenerator
    fast_multiplier        = 9
    fast_domain_clock_freq = int(48e3 * 256 * fast_multiplier)

    @property
    def file_templates(self):
        templates = super().file_templates
        templates["{{name}}.qsf"] += IntelFPGAParameters.QSF_ADDITIONS
        templates["{{name}}.sdc"] += IntelFPGAParameters.SDC_ADDITIONS
        return templates

    def __init__(self):
        self.resources += ADATFaceRev0Baseboard.resources(Attrs(io_standard="3.3-V LVCMOS"))
        # swap connector numbers, because on ADATface the connector
        # names are swapped compared to the QMTech daughterboard
        self.connectors[0].number = 3
        self.connectors[1].number = 2

        super().__init__(standalone=False)

#class ADATFaceArtix7(QMTechXC7A35TPlatform, LUNAPlatform):
#    clock_domain_generator = Xilinx7SeriesClockDomainGenerator
#    fast_multiplier        = 9
#    fast_domain_clock_freq = int(48e3 * 256 * fast_multiplier)
#
#    @property
#    def file_templates(self):
#        templates = super().file_templates
#        return templates
#
#    def __init__(self):
#        self.resources += ADATFaceRev0Baseboard.resources(Attrs(IOSTANDARD="LVCMOS33"))
#        # swap connector numbers, because on ADATface the connector
#        # names are swapped compared to the QMTech daughterboard
#        self.connectors[0].number = 3
#        self.connectors[1].number = 2
#
#        super().__init__(standalone=False)