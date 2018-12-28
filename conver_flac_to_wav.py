# coding=utf-8
from pydub import AudioSegment
import os
from time import sleep
from tqdm import tqdm, trange
localPath = "/Users/bruce/Downloads/speech/LibriSpeech/train-clean-100/"
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
# print(flacFinalPath)
print("==============================================================")
for filePath in dictionary:
    localPathFile = os.path.join(wavPath, filePath)
    for j in dictionary[filePath]:
        wavPathFile = os.path.join(localPathFile, j)
        finalPath.append(wavPathFile)
# print(finalPath)
# 递归创建目录
# for i in finalPath:
#     os.makedirs(i)

for flacDir, wavDir in zip(flacFinalPath, finalPath):
    localPath = flacDir
    wavPath = wavDir
    convertFile = '.wav'
    fileNameList = os.listdir(localPath)
    for file in tqdm(fileNameList, desc='copying...'):
        sleep(0.01)
        flacPath = os.path.join(localPath, file)
        # print(flacPath)
        if flacPath.endswith('.txt'):
            pass
        else:
            preFile = os.path.splitext(file)[0]
            wavPathSave = os.path.join(wavPath, preFile)+convertFile
            # print(wavPathSave)
            AudioSegment.from_file(flacPath).export(wavPathSave, format='wav')






