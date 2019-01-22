# coding=utf-8
import h5py
import numpy as np
import warnings
import os
import pickle
from time import sleep
from tqdm import tqdm, trange
from math import sqrt
from python_speech_features import mfcc
import scipy.io.wavfile as wav
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
#
# print(secondLevelList)
# for filePath in secondLevelList:
#     thirdPath = os.listdir(filePath)
#     for finalPath in thirdPath:
#         txtFilepath.append(os.path.join(filePath, finalPath))
# print(secondLevelList)

# dt = h5py.special_dtype(vlen=str)
# f = h5py.File('./train_100_wav_FilePath.hdf5', 'w')
# f.attrs['wavFilePath'] = np.string_(secondLevelList)
#
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
# f = h5py.File('./train_100_wav_Path.hdf5', 'w')
# f.attrs['wavFilePath'] = np.string_(waveFilePath)

# split a sentence wav file into words
# wav = h5py.File('./train_100_wav_Path.hdf5', 'r')
# for file in tqdm(wav.attrs['wavFilePath'], desc="======== fileNum ======="):
#     for wavfile in os.listdir(file):
#         each_wave = os.path.join(file, wavfile).decode('utf-8')
#         split_file_name = each_wave.split('.')[0]
#         newSplitName = split_file_name.replace('train_waveFile_100', 'train_100_split_wav')
#         os.makedirs(newSplitName)
#         sound_file = AudioSegment.from_wav(each_wave)
#         audio_chunks = split_on_silence(sound_file, min_silence_len=20, silence_thresh=-33)  # 这个参数是最优的
#         for i, chunk in enumerate(audio_chunks):
#             out_file = (newSplitName + "/split_{0}" + ".wav").format(i)
#             chunk.export(out_file, format="wav")
# print('wav split done!')


# wav = h5py.File('./train_100_wav_Path.hdf5', 'r')
# for file in tqdm(wav.attrs['wavFilePath'], desc="======== fileNum ======="):
#     for wavfile in os.listdir(file):
#         each_wave = os.path.join(file, wavfile).decode('utf-8')
#         split_file_name = each_wave.split('.')[0]
#         newSplitName = split_file_name.replace('train_waveFile_100', 'train_100_split_wav')
#         print(newSplitName)


# pathlist = ["/Users/bruce/Downloads/speech/LibriSpeech/train_100_split_wav/103/1240/103-1240-0000/split_0.wav",
#             "/Users/bruce/Downloads/speech/LibriSpeech/train_100_split_wav/103/1240/103-1240-0000/split_1.wav",
#             "/Users/bruce/Downloads/speech/LibriSpeech/train_100_split_wav/103/1240/103-1240-0000/split_2.wav"]
#
#
# a = list()
# for i in pathlist:
#     (rate, sig) = wav.read(i)
#     a.append(np.array(mfcc(sig, rate)))
#     # print(mfcc_feat.shape)
#     # print(mfcc_feat)
# # print(a[0].shape)
#
#
# def same_dim(v1, v2):
#     dim1 = v1.shape[0]
#     dim2 = v2.shape[0]
#     if dim1 < dim2:
#         v2 = v2[0:dim1, :]
#     elif dim2 < dim1:
#         v1 = v1[0:dim2, :]
#     return cos_sim(v1, v2)
#
#
# def cos_sim(vector_a, vector_b):
#     """
#     计算两个向量之间的余弦相似度
#     :param vector_a: 向量 a
#     :param vector_b: 向量 b
#     :return: sim
#     """
#     vector_a = np.mat(vector_a)
#     vector_b = np.mat(vector_b)
#     num = vector_a * vector_b.T
#     denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
#     cos = num / denom
#     return np.mean(cos)
#

#
#
# print(cos_sim(a[1], a[2]))
# print(same_dim(a[0], a[1]))


# 统计每句话单词的次数
# textlist = list()
# wav = h5py.File('./train_100_wav_FilePath.hdf5', 'r')
# for file in wav.attrs['wavFilePath']:
#     for wavfile in os.listdir(file):
#         each_wave = os.path.join(file, wavfile).decode('utf-8')
#         txtFile = each_wave.replace('train_waveFile_100', 'Librispeechtext')
#         textlist.append(txtFile)
# print(textlist)
# dt = h5py.special_dtype(vlen=str)
# f = h5py.File('./train_100_txt_path.hdf5', 'w')
# f.attrs['wavFilePath'] = np.string_(textlist)

# txtList = list()
# wav = h5py.File('./train_100_txt_path.hdf5', 'r')
# for file in wav.attrs['wavFilePath']:
#     for each_txt in os.listdir(file):
#         txtList.append(os.path.join(file, each_txt))
# print(txtList)


# 统计文本单词数目
# wordCount = dict()
# for eachTxtPath in txtList:
#     f = open(eachTxtPath)
#     for line in f:
#         txtName = line.strip('\n').split(' ')[0]
#         line = line.strip('\n').split(' ')[1:]
#         wordCount[txtName] = len(line)
# print(wordCount)


# wordCountFile = open('wordCount.pickle', 'wb')
# pickle.dump(wordCount, wordCountFile)
# wordCountFile.close()

splitDict = dict()
wav = h5py.File('./train_100_wav_FilePath.hdf5', 'r')
for file in wav.attrs['wavFilePath']:
    for wavfile in os.listdir(file):
        each_wave = os.path.join(file, wavfile).decode('utf-8')
        split_file_name = each_wave.split('.')[0]
        newSplitName = split_file_name.replace('train_waveFile_100', 'train_100_split_wav')
        eachFile = os.listdir(newSplitName)
        for splitFile in eachFile:
            fileCount = len(os.listdir(os.path.join(newSplitName, splitFile)))
            splitDict[splitFile] = fileCount

# print(splitDict)

wordCountRead = open('./wordCount.pickle', 'rb')
countDict = pickle.load(wordCountRead)

indexNum = 0
for i in countDict:
    if(countDict[i]>splitDict[i]):
        indexNum+=1
        print('分割的数目小于原来的数目的标签是：', i)
print(indexNum)

