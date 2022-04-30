from amaranth import Elaboratable, Module, Signal, Array, Memory, signed, Cat, Mux, Instance, ClockSignal, ResetSignal
from  amaranth.lib.cdc import ResetSynchronizer
import os
import subprocess
import shutil

class FFTGenWrapper(Elaboratable):
    wrapper_id = 0
    files = {}
    def __init__(self, tapcount, in_bitwidth, out_bitwidth, dspcount, inverse):
        self.id = FFTGenWrapper.wrapper_id
        FFTGenWrapper.wrapper_id += 1

        self.inverse = inverse
        self.folder_path = f"./fft-core_{self.id}/"


        #if not os.path.exists(self.folder_path):
        try:
            shutil.rmtree(self.folder_path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        fftgen_path = '/home/rouven/Computer/Coden/Audio/dblclockfft/sw/fftgen'
        #fftgen_path = './fftgen'
        args = []
        args.append(fftgen_path)
        args.extend(["-f", str(tapcount)])
        args.extend(["-n", str(in_bitwidth)])
        args.extend(["-m", str(out_bitwidth)])
        args.extend(["-x", str(0)])
        args.extend(["-k", str(500)])
        args.extend(["-p", str(dspcount)])
        args.extend(["-d", self.folder_path])

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

        if inverse:
            print("Patching ifft files")
            for file in ["ifftmain.v", "ifftstage.v"]:
                file_path = os.path.join(self.folder_path, file)
                with open(file_path) as f:
                    newText = f.read().replace('fftstage', 'ifftstage')

                with open(file_path, "w") as f:
                    f.write(newText)
            print("Patching ifft files completed.")

        self.valid_in = Signal()
        self.sample_in = Signal(in_bitwidth)
        self.sample_out = Signal(out_bitwidth)
        self.valid_out = Signal()


    def elaborate(self, platform) -> Module:
        m = Module()

        directory = os.fsencode(f"./fft-core_{self.id}/")

        for file in os.listdir(directory):
            #print(file)
            filename = os.fsdecode(file)
            if filename not in self.files:
                self.files[filename] = filename
                platform.add_file(filename, open(os.path.join(directory,file)))


        if self.inverse:
            instance_name = 'ifftmain'
        else:
            instance_name = 'fftmain'

        m.submodules.fft = Instance(instance_name,
            i_i_clk = ClockSignal("sync"),
            i_i_reset = ResetSignal("sync"),
            i_i_ce = self.valid_in,
            i_i_sample = self.sample_in,
            o_o_result = self.sample_out,
            o_o_sync = self.valid_out
        )

        return m

class FFTGenWrapperTest(Elaboratable):
    def __init__(self, tapcount, in_bitwidth, out_bitwidth):
        self._tapcount = tapcount
        self._out_bitwidth = out_bitwidth

        self.valid_in = Signal()
        self.sample_in = Signal(in_bitwidth*2)
        self.sample_out = Signal(out_bitwidth*2)
        self.valid_out = Signal()
        self.gotSamples = Signal(range(self._tapcount*3))
        self.fftCount = Signal(range(self._tapcount))
        self.outputSamples = Signal.like(self.gotSamples)



    def elaborate(self, platform) -> Module:
        m = Module()

        m.d.sync += [
            self.sample_out.eq(0),
            self.valid_out.eq(0),
        ]

        with m.If(self.valid_in):
            m.d.sync += self.gotSamples.eq(self.gotSamples + 1)

            #with m.If(self.gotSamples % self._tapcount == 0):
            #    m.d.sync += self.fftCount.eq(self.fftCount + 1)
            sig1 = Signal(10)
            sig2 = Signal(10)
            m.d.sync += [
                sig1.eq(self.fftCount // (self._tapcount-1))
            ]
            m.d.sync += self.fftCount.eq(self.gotSamples // (self._tapcount-2) - self.outputSamples//(self._tapcount-2))

            #with m.If(self.outputSamples % self._tapcount-1 == 0):
             #   m.d.sync += self.fftCount.eq(self.fftCount - 1)


        with m.If(self.fftCount > 0):
            m.d.sync += [
                self.outputSamples.eq(self.outputSamples + 1),
                self.sample_out.eq(((10000 + self.gotSamples*1000 + self.outputSamples) << self._out_bitwidth) + 30000 + self.gotSamples*1000 + self.outputSamples),
                self.valid_out.eq(1),
            ]



        return m
