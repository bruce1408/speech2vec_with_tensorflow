# coding=utf-8
import numpy as np
import os
import h5py
import warnings
warnings.filterwarnings(action='ignore')
secondLevelList = list()
txtFilepath = list()
firstLevel = '/Users/bruce/Downloads/speech/LibriSpeech/train-clean-100'
firstLevelList = os.listdir('/Users/bruce/Downloads/speech/LibriSpeech/train-clean-100')
for catalog in firstLevelList:
    secondLevelfile = os.path.join(firstLevel, catalog)
    secondLevelList.append(secondLevelfile)

for filePath in secondLevelList:
    thirdPath = os.listdir(filePath)
    for finalPath in thirdPath:
        txtFilepath.append(os.path.join(filePath, finalPath))


# print(txtFilepath)
# save file to h5py
# f = h5py.File('./totalFlacPath.hdf5', 'w')
# f.attrs['flacPath'] = np.string_(txtFilepath)  # 保存键值
# f.attrs['wavPath'] = np.string_(finalPath)


# load the data from h5py
textFileList = list()
r = h5py.File('./totalFlacPath.hdf5', 'r')
finalList = r.attrs['flacPath']
for i in finalList:
    for filename in os.listdir(i):
        textName = os.path.join(i, filename)
        textName = str(textName, encoding='utf-8')
        if textName.endswith('.txt'):
            textFileList.append(textName)


# # copy file
import shutil
modifyTextfile = list()
for textPath in textFileList:
    modifyTextfile.append(textPath.replace('train-clean-100', 'Librispeechtext'))

# # 创建目录 & 拷贝文件
# # for i in modifyTextfile:
# #     os.makedirs(i)
# for sourceText, targetText in zip(textFileList, modifyTextfile):
#     shutil.copyfile(sourceText, targetText)

# 将每个txt文件合并成一个
# f = open('/Users/bruce/Downloads/speech/totalText2vec.txt', 'a')
# for textPath in modifyTextfile:
#     u = open(textPath, 'r')
#     for line in u:
#         f.write(line)
#
# f.close()

# 每个单词变小写，改成 gensim 需要输入的形式。
totalText = list()
with open('/Users/bruce/Downloads/speech/totalText2vec.txt', 'r') as f:
    for line in f:
        str=' '
        line = line.strip('\n').split(' ')[1:]
        line = [x.lower() for x in line]
        # line = str.join(line)
        # print(line)
        totalText.append(line)


# 训练词向量
from gensim.models import word2vec
model = word2vec.Word2Vec(totalText, sg=1, size=50, window=3, min_count=1)  # sg=1 表示skip-gram模型
# print(model['keep'])
print(model.most_similar(['love']))
model.wv.save_word2vec_format('model.txt', binary=False)

# 保存不重复的单词到wordtext
# totalText = set(totalText)
# print(totalText)
# for word in totalText:
#     word = word.lower()
#     # print(word)
#     if word.endswith('\'s'):
#         print(word)
# with open('/Users/bruce/Downloads/speech/wordtext.txt', 'w') as l:
#     for word in totalText:
#         word = word.lower()
#         l.write(word+'\n')

