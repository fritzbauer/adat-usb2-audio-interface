import wave
#import matplotlib.pyplot as plt
import numpy as np
import math

from datetime import datetime
import binascii

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
        #else:
        #    samples = samples[30 * self.sample_rate:40 * self.sample_rate]

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
            tapsfft1[j] = np.fft.fft(taps[j*stepsize:(j+1)*stepsize, 0], n=fftsize)
            tapsfft2[j] = np.fft.fft(taps[j*stepsize:(j+1)*stepsize, 1], n=fftsize)

        buffer1 = np.zeros(samplecount + fftsize, dtype=np.complex128)
        buffer2 = np.zeros(samplecount + fftsize, dtype=np.complex128)

        for n in offsets:
            np.roll(previous_samples1, fftsize)
            np.roll(previous_samples2, fftsize)
            previous_samples1[0] = np.fft.fft(samples[n:n + stepsize, 0], n=fftsize)
            previous_samples2[0] = np.fft.fft(samples[n:n + stepsize, 1], n=fftsize)
            # print(f"signalfft len: {np.shape(signalfft1)}")

            slice1 = np.zeros((fftsize), dtype=np.complex128)
            slice2 = np.zeros((fftsize), dtype=np.complex128)

            for j in range(slices):

                convolved1 = previous_samples1[j] * tapsfft1[j] + previous_samples2[j] * tapsfft2[j]
                convolved2 = previous_samples2[j] * tapsfft1[j] + previous_samples1[j] * tapsfft2[j]
                slice1 += convolved1
                slice2 += convolved2
            #print(f"Slice1: {slice1}")

            modifiedsignal1 = np.fft.ifft(slice1, n=fftsize)
            modifiedsignal2 = np.fft.ifft(slice2, n=fftsize)

            #print(f"Modifiedshape: {np.shape(modifiedsignal1)}: {modifiedsignal1}")
            buffer1[n:n + fftsize] += modifiedsignal1
            buffer2[n:n + fftsize] += modifiedsignal2

        for i in range(samplecount):
            #if i < 40:
            #    print(f"Output: res: {int(res[i])}; abs: {np.abs(res[i])}; real: {np.real(res[i])}; img: {np.imag(res[i])}")
            output_samples[i, 0] = int(np.real(buffer1[i])) >> self.bitwidth
            output_samples[i, 1] = int(np.real(buffer2[i])) >> self.bitwidth

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
                data[i // bytes_per_Frame, channel] = sample * 2 ** normalize_factor #normalize to bitwidth

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

if __name__ == "__main__":
    FFTConvolver().run()
