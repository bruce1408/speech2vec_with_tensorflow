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
for wavPath in fileList:
    filePath = os.path.join(localDataPath, wavPath)
    print(filePath)
    (rate, sig) = wav.read(filePath)
    mfcc_feat = mfcc(sig, rate)
    print(mfcc_feat.shape)
    print(mfcc_feat)
    # d_mfcc_feat = delta(mfcc_feat, 2)
    # print(d_mfcc_feat.shape)
    fbank_feat = logfbank(sig, rate)
    # print(fbank_feat.shape)
    # print(fbank_feat[1:3, :])


# 音频切词
# from pydub import AudioSegment
# from pydub.silence import split_on_silence
#
# # 这里silence_thresh是认定小于-70dBFS以下的为silence，发现小于-70dBFS部分超过 700毫秒，就进行拆分。这样子分割成一段一段的。
#
# from pydub.silence import split_on_silence
# sound_file = AudioSegment.from_wav("/Users/bruce/Downloads/speech/LibriSpeech"
#                                    "/train-clean-100/wavFile/103-1240-0025.wav")
# audio_chunks = split_on_silence(sound_file, min_silence_len=20, silence_thresh=-33)  # 这个参数是最优的
# for i, chunk in enumerate(audio_chunks):
#     out_file = "/Users/bruce/Downloads/speech/LibriSpeech/train-clean-100/wavFile/split_{0}.wav".format(i)
#     print("exporting", out_file)
#     chunk.export(out_file, format="wav")
# ffmpeg -i 103-1240-0057.wav -f segment -segment_time 1 -c copy out%03d.wav


# 统计一句话中单词的个数
# dictionary = dict()
# textFilePath = "/Users/bruce/Downloads/speech/LibriSpeech/train-clean-100/103/103-1240.trans.txt"
# with open(textFilePath, 'r') as f:
#     for line in f:
#         line = line.strip('\n').split(' ')
#         dictionary[line[0]] = line.__len__()-1
# print(dictionary)

