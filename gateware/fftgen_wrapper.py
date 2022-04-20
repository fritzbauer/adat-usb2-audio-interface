from amaranth import Elaboratable, Module, Signal, Array, Memory, signed, Cat, Mux, Instance, ClockSignal, ResetSignal
from  amaranth.lib.cdc import ResetSynchronizer
import os
import subprocess

class FFTGenWrapper(Elaboratable):
    def __init__(self, tapcount, bitwidth, dspcount, inverse):
        fftgen_path = '/home/rouven/Computer/Coden/Audio/dblclockfft/sw/fftgen'
        #fftgen_path = './fftgen'
        args = []
        args.append(fftgen_path)
        args.extend(["-f", str(tapcount)])
        args.extend(["-n", str(bitwidth)])
        args.extend(["-m", str(bitwidth)])
        args.extend(["-x", str(4)])
        args.extend(["-p", str(dspcount)])
        args.extend(["-d", "./fft-core/"])

        #cmd = f"{fftgen_path} -f {tapcount} -n {bitwidth} -m {bitwidth} -x 4 -p {dspcount}"
        if inverse:
            #cmd = f"{cmd} -i"
            args.append("-i")
        #-f taps
        #-n input bitwidth
        #-m output bitwidth
        #-x twiddle factor (additional bits for internal calculation)
        #-p Number of DSPs to use
        #-k only expect a sample maximum every n clocks (saves DSPs)

        print(args)
        process = subprocess.Popen(args,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(f"Stdout: {stdout}")
        print(f"Stderr: {stderr}")

        self.valid_in = Signal()
        self.sample_in = Signal(bitwidth)
        self.sample_out = Signal(bitwidth)
        self.valid_out = Signal()


    def elaborate(self, platform) -> Module:
        m = Module()

        directory = os.fsencode("./fft-core/")
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            platform.add_file(filename, open(os.path.join(directory,file)))

        m.submodules.fft = Instance("fftmain",
            i_clk = ClockSignal("sync"),
            i_reset = ResetSignal("sync"),
            i_ce = self.valid_in,
            i_sample = self.sample_in,
            o_result = self.sample_out,
            o_sync = self.valid_out
        )

        return m