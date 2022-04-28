import wave
#import matplotlib.pyplot as plt
import numpy as np
import math

from datetime import datetime
import binascii
from cmath import exp, pi

class FFTConvolver:

    def __init__(self):
        self.testing = False
        self.sample_rate = 48000
        self.bitwidth = 24

        if self.testing:
            self.tapcount = 4096
        else:
            self.tapcount = 4096

        self.ir_filename = '/home/rouven/tmp/IR_4800.wav'
        self.samples_filename = '/home/rouven/tmp/CS.wav'
        self.output_filename = '/home/rouven/tmp/CS_modfft.wav'

    def run(self):
        print(f"{datetime.now().strftime('%H:%M:%S')}: Loading Taps")
        taps, _ = self.load_wav_file(self.ir_filename, self.bitwidth)
        taps = taps[:self.tapcount]
        
        print(f"{datetime.now().strftime('%H:%M:%S')}: Loading samples")
        samples, _ = self.load_wav_file(self.samples_filename, self.bitwidth)

        if self.testing:
            samples = samples[30 * self.sample_rate:30 * self.sample_rate + self.tapcount]
        else:
            samples = samples[30 * self.sample_rate:40 * self.sample_rate]

        print(f"{datetime.now().strftime('%H:%M:%S')}: Calculating")
        out_data = self.calculate_channel_fft(samples, taps)

        print(f"{datetime.now().strftime('%H:%M:%S')}: Saving output")
        self.save_wav_file(self.output_filename, out_data, self.bitwidth)


    def calculate_channel_fft(self, samples, taps):
        output_samples = samples.copy()
        samplecount = len(samples[:,0])
        tapcount = len(taps[:,0])

        stepsize = 128 #fftsize-tapcount +1 #or 32
        fftsize = 2 << (stepsize - 1).bit_length()
        slices = tapcount // stepsize
        offsets = range(0,samplecount,stepsize)
        previous_samples1 = np.zeros((slices, fftsize), dtype=np.complex128)
        previous_samples2 = np.zeros((slices, fftsize), dtype=np.complex128)

        tapsfft1 = np.zeros((slices, fftsize), dtype=np.complex128)
        tapsfft2 = np.zeros((slices, fftsize), dtype=np.complex128)
        for j in range(slices):
            #tapsfft1[j] = np.fft.fft(taps[j*stepsize:(j+1)*stepsize, 0], n=fftsize)
            #tapsfft2[j] = np.fft.fft(taps[j*stepsize:(j+1)*stepsize, 1], n=fftsize)
            z1 = np.zeros((fftsize))
            z2 = np.zeros((fftsize))
            z1[:stepsize] = taps[j * stepsize:(j + 1) * stepsize, 0]
            z2[:stepsize] = taps[j * stepsize:(j + 1) * stepsize, 1]
            tapsfft1[j] = FFTConvolver.FFT(z1)
            tapsfft2[j] = FFTConvolver.FFT(z2)

        buffer1 = np.zeros(samplecount + fftsize, dtype=np.complex128)
        buffer2 = np.zeros(samplecount + fftsize, dtype=np.complex128)

        for n in offsets:
            if (n // stepsize) % 200 == 0:
                print(f"{n}/{samplecount}")

            np.roll(previous_samples1, fftsize)
            np.roll(previous_samples2, fftsize)
            z1 = np.zeros((fftsize))
            z2 = np.zeros((fftsize))
            z1[:stepsize] = samples[n:n + stepsize, 0]
            z2[:stepsize] = samples[n:n + stepsize, 0]
            previous_samples1[0] = FFTConvolver.FFT(z1)
            previous_samples2[0] = FFTConvolver.FFT(z2)

            slice1 = np.zeros((fftsize), dtype=np.complex128)
            slice2 = np.zeros((fftsize), dtype=np.complex128)

            for j in range(slices):

                convolved1 = previous_samples1[j] * tapsfft1[j] + previous_samples2[j] * tapsfft2[j]
                convolved2 = previous_samples2[j] * tapsfft1[j] + previous_samples1[j] * tapsfft2[j]
                slice1 += convolved1
                slice2 += convolved2
            #print(f"Slice1: {slice1}")

            modifiedsignal1 = FFTConvolver.IFFT(slice1)
            modifiedsignal2 = FFTConvolver.IFFT(slice2)

            #print(f"Modifiedshape: {np.shape(modifiedsignal1)}: {modifiedsignal1}")
            buffer1[n:n + fftsize] += modifiedsignal1
            buffer2[n:n + fftsize] += modifiedsignal2

        for i in range(samplecount):
            #if i < 40:
            #    print(f"Output: res: {int(res[i])}; abs: {np.abs(res[i])}; real: {np.real(res[i])}; img: {np.imag(res[i])}")
            output_samples[i, 0] = int(np.real(buffer1[i])) >> self.bitwidth#(self.bitwidth+12)
            output_samples[i, 1] = int(np.real(buffer2[i])) >> self.bitwidth#(self.bitwidth+12)

        if self.testing:
            print(f"taplen: {len(taps)}")
            print(f"Samplecount: {samplecount}; fftsize: {fftsize}; stepsize: {stepsize}")
            print(f"tapsfft len: {np.shape(tapsfft1)}")
            print(f"np.abs: {np.abs(buffer1[:samplecount])}")
            print(f"output_samples: {output_samples[3900:4200,0]}")
            print(f"{datetime.now().strftime('%H:%M:%S')}: Finished")

        return output_samples

    def load_wav_file(self, filename, bitwidth):
        with wave.open(filename, 'rb') as wav:
            raw_data = wav.readframes(wav.getnframes())
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            samplewidth = wav.getsampwidth()  # 3 bytes = 24bit
        
        # bring the signal to the desired bitwidth
        normalize_factor = bitwidth - samplewidth * 8

        bytes_per_Frame = channels * samplewidth
        data = np.zeros((len(raw_data) // bytes_per_Frame, channels), dtype='int64')

        for i in range(0, len(raw_data), bytes_per_Frame):
            for channel in range(channels):
                sample = int.from_bytes(raw_data[i + channel * samplewidth:i + (channel+1) * samplewidth], byteorder='little', signed=True)
                data[i // bytes_per_Frame, channel] = (sample * 2 ** normalize_factor) #>> (bitwidth//2)#normalize to bitwidth

        assert self.validate_pcm_data(data, bitwidth)

        return data, sample_rate

    def save_wav_file(self, filename, data, bitwidth):
        assert self.validate_pcm_data(data, bitwidth)
        out_wav = bytearray()
        for i in range(len(data)):
            out_wav.extend(int(data[i, 0]).to_bytes(math.ceil(bitwidth/8), 'little', signed=True))
            out_wav.extend(int(data[i, 1]).to_bytes(math.ceil(bitwidth/8), 'little', signed=True))

        print(data[:, 0])

        # out = signal.fftconvolve(samples[:10,0], taps[:3,0], mode="full")
        # output_samples[:,0] = out

        # samples = [5,3,2,7,4,1,3,2,1,1,0]
        # taps = [1,2,3]
        # out = signal.fftconvolve(samples, taps, mode="same")
        # print(out)
        #length = 10
        # assert np.allclose(out[:length],output_samples[:length,0]), f"out: {out[:length]}; outsam: {output_samples[:length,0]}"

        with wave.open(filename, 'wb') as wav:
            wav.setnchannels(2)
            wav.setsampwidth(3)  # 3 bytes = 24bit
            wav.setframerate(self.sample_rate)
            wav.setnframes(len(data))
            wav.writeframesraw(out_wav)

    def validate_pcm_data(self, data, bitwidth):
        valid = True
        for sample in range(len(data)):
            if not -1 * 2 ** (bitwidth - 1) <= data[sample, 0] <= 1 * 2 ** (bitwidth - 1) - 1:
                print(f"Out of range: {data[sample, 0]}")
                valid = False

            if not -1 * 2 ** (bitwidth - 1) <= data[sample, 1] <= 1 * 2 ** (bitwidth - 1) - 1:
                print(f"Out of range: {data[sample, 1]}")
                valid = False
        return valid

    #https://dsp.stackexchange.com/questions/70649/fft-for-long-waveform
    #https://gist.github.com/dannvix/12c26919b6e182afb1f724b051e1ad7a
    #https://rosettacode.org/wiki/Fast_Fourier_transform
    # >> Discrete Fourier transform for sampled signals
    # x [in]: sampled signals, a list of magnitudes (real numbers)
    # yr [out]: real parts of the sinusoids
    # yi [out]: imaginary parts of the sinusoids
    def DFT(x):
        N, out = len(x), []
        for k in range(N):
            real, imag = 0, 0
            for n in range(N):
                theta = -k * (2 * math.pi) * (float(n) / N)
                real += int(x[n]) * math.cos(theta)
                imag += int(x[n]) * math.sin(theta)
            out.append(complex(real, imag))
        return out

    def FFT(x):
        x = np.asarray(x, dtype=np.complex128)
        data_length = len(x)

        bits = round(math.log(data_length) / math.log(2))
        for j in range(1, data_length):
            swapPos = FFTConvolver.Reverse_Bits(j, bits)
            if swapPos <= j:
                continue

            temp = x[j]
            x[j] = x[swapPos]
            x[swapPos] = temp

        N = 2
        while N <= data_length:
            for i in range(0, data_length, N):
                for k in range(N//2):
                    evenIndex = i + k
                    oddIndex = i + k + (N // 2)
                    even = x[evenIndex]
                    odd = x[oddIndex]

                    term = -2 * math.pi * k / float(N)
                    exp = complex(math.cos(term), math.sin(term)) * odd

                    x[evenIndex] = even + exp
                    x[oddIndex] = even - exp

            N = N << 1
        return x

    def FFT_Recurse(x):
        N = len(x)
        if N <= 1:
            return x

        even = FFTConvolver.FFT2(x[0::2])
        odd = FFTConvolver.FFT2(x[1::2])
        T = [exp(-2j * pi * k / N) * odd[k] for k in range(N // 2)]
        return [even[k] + T[k] for k in range(N // 2)] + \
               [even[k] - T[k] for k in range(N // 2)]


    def IFFT(x):
        x = np.asarray(x, dtype=np.complex128)
        N = len(x)
        for i in range(N):
            x[i] = x[i].conjugate()
        out = FFTConvolver.FFT(x)
        for i in range(N):
            out[i] = out[i].conjugate() / N

        return out

    def Reverse_Bits(n, no_of_bits):
        result = 0
        for i in range(no_of_bits):
            result <<= 1
            result |= n & 1
            n >>= 1
        return result



if __name__ == "__main__":
    FFTConvolver().run()
    fft = FFTConvolver.FFT2([1, 1, 1, 1, 0, 0, 0, 0])
    #fft = FFTConvolver.DFT2([1, 1, 1, 1, 0, 0, 0, 0] + [1, 1, 1, 1, 0, 0, 0, 0])
    #fft = FFTConvolver.DFT2(np.arange(128))
    offt = []
    for i in fft:
        offt.append(i)
    print(offt)
    ifft = FFTConvolver.IFFT2(fft)
    out = []
    for i in ifft:
        out.append(round(abs(i)))
    print(out)
