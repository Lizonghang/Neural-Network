# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def replace_nan_with_mean(D):
    m, n = np.shape(D)
    for i in range(n):
        meanVal = np.mean(D[np.nonzero(~np.isnan(D[:, i].A))[0], i])
        D[np.nonzero(np.isnan(D[:, i].A))[0], i] = meanVal
    return D


def load_data(filename, start=0, end=None):
    data = []
    with open(filename, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            digits = line.strip().split()
            data.append(map(float, digits[start: end]))
    return np.matrix(data)


def plot_rate(eig_val, n_components=20):
    plt.figure()
    eig_sum = eig_val.sum()
    plt.plot(range(n_components), [eig_val[n] / eig_sum for n in range(n_components)])
    plt.xlabel(u'主成分数目')
    plt.ylabel(u'方差百分比')
    plt.show()


D = load_data('secom.data')
D = replace_nan_with_mean(D)
pca = PCA(n_components=20)
pca.fit(D)
plot_rate(pca.explained_variance_, n_components=20)
