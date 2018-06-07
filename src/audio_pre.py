import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.ndimage

# from scipy.io import wavfile
# from scipy.signal import butter, lfilter
# from audio_util import *

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
import pdb
import wavio
### Parameters ###

audio_data = sorted(glob(os.path.join("./data/flickr_audio/wavs/", "*.wav")))
print(len(audio_data))
# Grab your wav and filter it
maxlen = 0
for ad in audio_data:#[4000:5000]:
    # Only use a short clip for our demo
    # pdb.set_trace()
    try:
        # pdb.set_trace()
        wave_obj = wavio.read(ad)
        rate = wave_obj.rate
        sig = np.squeeze(wave_obj.data)
        # (rate,sig) = wav.read(ad)
    except TypeError:
        # print(ad)
        (rate,sig) = wav.read(ad)
    # only short than 10 seconds
    if np.shape(sig)[0]/float(rate) > 10:
        sig = sig[0:rate*10]
    # Mel-filter bank
    sig = sig - np.mean(sig)
    fbank_feat = logfbank(sig, rate, winlen=0.025,\
        winstep=0.01,nfilt=40,nfft=512,lowfreq=0,highfreq=None,preemph=0.97)
    pdb.set_trace()
    if fbank_feat.shape[0] < 1024:
        # pdb.set_trace()
        zero_pad = np.zeros((1024-fbank_feat.shape[0], 40))
        fbank_feat = np.concatenate([fbank_feat, zero_pad], 0)

    maxlen = max(maxlen, fbank_feat.shape[0])

# ./data/flickr_audio/wavs/2865703567_52de2444f2_0.wav
print(maxlen)