# coding=utf-8
import numpy as np
import pickle
import h5py


def save_pickle(filePath, data):
    wordCountFile = open(filePath, 'wb')
    pickle.dump(data, wordCountFile)
    wordCountFile.close()


def load_pickle(pickleFilePath):
    file = open(pickleFilePath, 'rb')
    countDict = pickle.load(file)


def save_hdf5(filePath, data):
    f = h5py.File(filePath, 'w')
    f.attrs['wavFilePath'] = np.string_(data)


def load_hdf5(hdf5Path):
    f = h5py.File(hdf5Path, 'r')
    for keys in f.attrs:
        print(f.attrs[keys])
