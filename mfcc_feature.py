# coding=utf-8
import tensorflow as tf
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(1)
import numpy as np
import os
np.set_printoptions(suppress=True, threshold=np.NaN)

localDataPath = "/Users/bruce/Downloads/speech_commands_v0.02/eight/"
fileList = os.listdir(localDataPath)
a = list()
for wavPath in fileList:
    filePath = os.path.join(localDataPath, wavPath)
    print(filePath)
    # (rate, sig) = wav.read("/Users/bruce/Downloads/speech_commands_v0.02/house/e102119e_nohash_1.wav")
    (rate, sig) = wav.read(filePath)
    mfcc_feat = mfcc(sig, rate)
    # print(mfcc_feat.shape)
    # a.append(mfcc_feat.shape[0])
    # print(set(a.append(mfcc_feat.shape[0])))
# d_mfcc_feat = delta(mfcc_feat, 2)
#     fbank_feat = logfbank(sig, rate)
#     print(fbank_feat.shape)
# print(fbank_feat[1:3, :])
# print(set(a))