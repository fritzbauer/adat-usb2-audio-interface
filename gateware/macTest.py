import wave

import numpy as np

from datetime import datetime
from threading import Thread

import binascii


testing = False
sample_rate = 48000
bitwidth = 24

if testing:
    tapcount = 16
else:
    tapcount = 4096


with wave.open('IRs/IR_4800.wav', 'rb') as wav:
    ir_data = wav.readframes(wav.getnframes())
ir_sig = np.zeros((len(ir_data)//6,2), dtype='int32')

for i in range(0,len(ir_data), 6):
    ir_sig[i//6, 0] = int.from_bytes(ir_data[i:i + 3], byteorder='little', signed=True)
    ir_sig[i//6, 1] = int.from_bytes(ir_data[i + 3:i + 6], byteorder='little', signed=True)

with wave.open('IRs/CS.wav', 'rb') as wav:
    sample_data = wav.readframes(wav.getnframes())

sig = np.zeros((len(sample_data)//6,2), dtype='int32')
for i in range(0,len(sample_data), 6):
    sig[i//6, 0] = int.from_bytes(sample_data[i:i+3], byteorder='little', signed=True)
    sig[i // 6, 1] = int.from_bytes(sample_data[i+3:i + 6], byteorder='little', signed=True)


def calculate_channel(output_samples, samples, taps, tapcount, sample_rate, channel, testing):
    print(f"{datetime.now().strftime('%H:%M:%S')}: Starting")
    for sample in range(len(samples)):
        if sample % (sample_rate//2) == 0:
            print(f"{datetime.now().strftime('%H:%M:%S')}: {sample}")
        sum = 0
        #print(f"Sample: {sample}")
        for tap in range(tapcount):

            if tap > sample:
                break
            #left  += (int(samples[sample - tap, 0] * taps[tap, 0]) >> bitwidth) + (int(samples[sample - tap, 1] * taps[tap, 1]) >> bitwidth)
            #right += (int(samples[sample - tap, 1] * taps[tap, 0]) >> bitwidth) + (int(samples[sample - tap, 0] * taps[tap, 1]) >> bitwidth)
            if channel == 0:
                bleed_channel = 1
            else:
                bleed_channel = 0

            val1 = int(samples[sample - tap, channel]) * int(taps[tap,0])
            val1s = val1 >> (bitwidth)
            val2 = int(samples[sample - tap, bleed_channel]) * int(taps[tap,1])
            val2s = val2 >> (bitwidth)

            sum1 = val1s + val2s
            sum += sum1#(sum1 >> (bitwidth-1))

            if testing:
                print(f"Sample: {sample} Tap: {tap}; Bleedchannel: {bleed_channel}")
                print(f"\t\tChannel value: {samples[sample - tap, channel]}; {type(samples[sample - tap, channel])}")
                print(f"\t\tBleedchannel value: {samples[sample - tap, bleed_channel]}; {type(samples[sample - tap, bleed_channel])}")
                print(f"\t\tTap value: {taps[tap, 0]}; {type(taps[tap,0])}")
                print(f"\t\tBleedtap value: {taps[tap, 1]}; {type(taps[tap,0])}")
                print(f"\t\tCalc1: {val1}; {type(val1)}")
                print(f"\t\tCalc1s: {val1s}; {type(val1s)}")
                print(f"\t\tCalc2: {val2}; {type(val2)}")
                print(f"\t\tCalc2s: {val2s}; {type(val2s)}")
                print(f"\t\tSum: {sum1}; {type(sum1)}")
                print(f"\t\tSum shifted : {sum1 >> (bitwidth)}")
                print(f"\t\tOverallsum: {sum} ; {type(sum)}")

        output_samples[sample, channel] = np.int32(sum)
        if testing:
            print(f"\t\t{sum}; type: {type(sum)}")
            print(f"\t\tOut: {output_samples[sample, channel]}; {type(output_samples[sample, channel])}")

    print(f"{datetime.now().strftime('%H:%M:%S')}: Finished")
    return

taps = ir_sig[:tapcount].copy()

for tap in range(len(taps)):
    if not -8388608 <= taps[tap, 0] <= 8388607:
        print(f"Out of range: {taps[tap, 0]}")

    if not -8388608 <= taps[tap, 1] <= 8388607:
        print(f"Out of range: {taps[tap, 1]}")

if testing:
    samples = sig[30*sample_rate:30*sample_rate+tapcount]
else:
    samples = sig[30*sample_rate:35*sample_rate]

for sample in range(len(samples)):
    #samples[sample, 0] = int(samples[sample, 0] * 2 ** (bitwidth-1)-1)
    #samples[sample, 1] = int(samples[sample, 1] * 2 ** (bitwidth-1)-1)

    if not -8388608 <= samples[sample, 0] <= 8388607:
        print(f"Out of range: {samples[sample, 0]}")

    if not -8388608 <= samples[sample, 1] <= 8388607:
        print(f"Out of range: {samples[sample, 1]}")


output_samples = samples.copy()

thread1 = Thread(target=calculate_channel, args=(output_samples, samples, taps, tapcount, sample_rate, 0, testing))
thread1.start()
thread2 = Thread(target=calculate_channel, args=(output_samples, samples, taps, tapcount, sample_rate, 1, testing))
thread2.start()
#
#
thread1.join()
thread2.join()

#print(output_samples)

out_wav = bytearray()
for i in range(len(output_samples)):
    out_wav.extend(int(output_samples[i,0]).to_bytes(3, 'little', signed=True))
    out_wav.extend(int(output_samples[i,1]).to_bytes(3, 'little', signed=True))

with wave.open("IRs/CS_mod2.wav", 'wb') as wav:
    wav.setnchannels(2)
    wav.setsampwidth(3) #3 bytes = 24bit
    wav.setframerate(sample_rate)
    wav.setnframes(len(samples))
    wav.writeframesraw(out_wav)
