import wave
import os
import sys
import struct
import soundfile as sf
from SingleMicNR import SingleMicNR
import matplotlib.pyplot as plt
os.chdir("D:\Study report\Single mic NR\Single_mic_noise_reduction\Single_mic_noise_reduction")
pcm16bit=16
mono=1
byte=8
bit16=2**16-1
bit16_half=bit16>>1
rec=wave.open("0.wav",mode="rb")
dataSoundfile, samplerate = sf.read('0.wav')
#wavout=wave.open("output.wav",mode="wb")
if rec.getnchannels()!=mono:
    print("wav file isn't mono")
    rec.close()
    sys.exit()
else:
    rec_sample_rate=rec.getframerate()
    rec_bit_rate=rec.getsampwidth()*byte
    total_frame_num=rec.getnframes()
    rec_data=rec.readframes(total_frame_num)

    if rec_bit_rate==pcm16bit:
        data=struct.unpack("%ih" % total_frame_num,rec_data)
        data=list(data)
        pcm_data=[]
        i=0
        while i<len(data):
            buf=data[i]/bit16_half
            pcm_data.append(buf)
            i=i+1
        rec.close()
        Mydemo = SingleMicNR(dataSoundfile,samplerate)
        MyNRdata=Mydemo.Magnitude_Subtraction()
        sf.write('output.wav', MyNRdata, samplerate)
        # j=0
        # output=[]
        #         # while j<len(MySMdata):
        #         #     buf=int(MySMdata[j]*bit16_half)
        #         #     packed_value = struct.pack('h',buf)
        #         #     output.append(packed_value)
        #         #     j=j+1

        # wavout.setparams((1, 2, 16000,0, 'NONE', 'not compressed'))
        # output_str = b''.join(output)
        # wavout.writeframes(output_str)
        # wavout.close()

