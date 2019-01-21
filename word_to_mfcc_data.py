# coding=utf-8
import h5py
import numpy as np
import warnings
import os
warnings.filterwarnings(action='ignore')
from pydub import AudioSegment
from pydub.silence import split_on_silence


# 对每一个wav文件进行切分
# 这里silence_thresh是认定小于-70dBFS以下的为silence，发现小于-70dBFS部分超过 700毫秒，就进行拆分。这样子分割成一段一段的。


# secondLevelList = list()
# txtFilepath = list()
# firstLevel = '/Users/bruce/Downloads/speech/LibriSpeech/train_waveFile_100'
# firstLevelList = os.listdir('/Users/bruce/Downloads/speech/LibriSpeech/train_waveFile_100')
# for catalog in firstLevelList:
#     secondLevelfile = os.path.join(firstLevel, catalog)
#     secondLevelList.append(secondLevelfile)
#
# for filePath in secondLevelList:
#     thirdPath = os.listdir(filePath)
#     for finalPath in thirdPath:
#         txtFilepath.append(os.path.join(filePath, finalPath))
# print(secondLevelList)

# dt = h5py.special_dtype(vlen=str)
# wavFilePath = np.array(secondLevelList)
# f = h5py.File('./train_100_wav_FilePath.hdf5', 'w')
# f.attrs['wavFilePath'] = np.string_(secondLevelList)

# waveFilePath = list()
# r = h5py.File('./train_100_wav_FilePath.hdf5', 'r')
# for i in r.attrs:
#     # print(r.attrs[i])
#     for j in r.attrs[i]:
#         for k in os.listdir(j):
#             waveFilePath.append(os.path.join(j, k))
# print(waveFilePath)
#
#
# dt = h5py.special_dtype(vlen=str)
# wavFilePath = np.array(waveFilePath)
# f = h5py.File('./train_100_wav_Path.hdf5', 'w')
# f.attrs['wavFilePath'] = np.string_(waveFilePath)

wav = h5py.File('./train_100_wav_Path.hdf5', 'r')

for file in wav.attrs['wavFilePath']:

    for wavfile in os.listdir(file):
        each_wave = os.path.join(file, wavfile).decode('utf-8')
        split_file_name = each_wave.split('.')[0]
        newSplitName = split_file_name.replace('train_waveFile_100', 'train_100_split_wav')
        os.makedirs(newSplitName)

        sound_file = AudioSegment.from_wav(each_wave)
        audio_chunks = split_on_silence(sound_file, min_silence_len=20, silence_thresh=-33)  # 这个参数是最优的
        for i, chunk in enumerate(audio_chunks):
            out_file = (newSplitName + "/split_{0}" + ".wav").format(i)
            print("exporting", out_file)
            chunk.export(out_file, format="wav")