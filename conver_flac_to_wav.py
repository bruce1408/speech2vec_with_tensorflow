# coding=utf-8
from pydub import AudioSegment
import os
import h5py
import numpy as np
from tqdm import tqdm, trange

# 生成flac目录
localPath = "/Users/bruce/Downloads/speech/LibriSpeech/train-clean-100/"
# localPath = '/Users/bruce/Downloads/train_360/train-clean-360/'
wavPath = "/Users/bruce/Downloads/speech/LibriSpeech/train_waveFile_100"
convertFile = '.wav'
firstDir = os.listdir(localPath)
thirdDir = list()
dictionary = dict()
flacFinalPath = list()
finalPath = list()
for i in firstDir:
    secondDir = os.listdir(os.path.join(localPath, i))
    dictionary[i] = secondDir

for filePath in dictionary:
    localPathFile = os.path.join(localPath, filePath)
    for j in dictionary[filePath]:
        wavPathFile = os.path.join(localPathFile, j)
        flacFinalPath.append(wavPathFile)

print("="*20)
for filePath in dictionary:
    localPathFile = os.path.join(wavPath, filePath)
    for j in dictionary[filePath]:
        wavPathFile = os.path.join(localPathFile, j)
        finalPath.append(wavPathFile)


print(flacFinalPath)
print(finalPath)
# 保存 flac 原始音频文件的目录
# dt = h5py.special_dtype(vlen=str)
# flacFinal = np.array(flacFinalPath)
#
# f = h5py.File('./totalFlacPath.hdf5', 'w')
# f.attrs['flacPath'] = np.string_(flacFinalPath)
# f.attrs['wavPath'] = np.string_(finalPath)

# r = h5py.File('./totalFlacPath.hdf5', 'r')
# for i in r.attrs:
#     print(r.attrs[i])
#     for j in (r.attrs[i]):
#         print(os.listdir(j))

# f.create_dataset('flacPath', data=flacFinalPath, dtype=dt)
# f['flacPath'] = flacFinal
# print(flacFinal)

# 递归创建目录
# for i in finalPath:
#     os.makedirs(i)

# 完成重flac文件到wav文件的转换。
# for flacDir, wavDir in zip(flacFinalPath, finalPath):
#     localPath = flacDir
#     wavPath = wavDir
#     convertFile = '.wav'
#     fileNameList = os.listdir(localPath)
#     for file in tqdm(fileNameList, desc='copying...'):
#         flacPath = os.path.join(localPath, file)
#         # print(flacPath)
#         if flacPath.endswith('.txt'):
#             pass
#         else:
#             preFile = os.path.splitext(file)[0]
#             wavPathSave = os.path.join(wavPath, preFile)+convertFile
#             # print(wavPathSave)
#             AudioSegment.from_file(flacPath).export(wavPathSave, format='wav')





