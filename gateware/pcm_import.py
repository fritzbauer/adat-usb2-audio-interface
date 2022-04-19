import wave
import numpy as np

class PCMImport:   

    def load_wav_file(filename, bitwidth):
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

        assert PCMImport.validate_pcm_data(data, bitwidth)

        return data, sample_rate


    def validate_pcm_data(data, bitwidth):
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
