# coding=utf-8
import random
import numpy as np
import matplotlib.pyplot as plt
from compiler.ast import flatten
from types import NoneType


def load_data(filename, start=0, end=None, label=None):
    data = []
    labels = []
    with open(filename, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            digits = line.strip().split()
            data.append(map(float, digits[start: end]))
            if label: labels.append(map(float, digits[label]))
    return np.matrix(data), np.matrix(labels)


def plot1D(D, title='', show=False, color=None, labels=None):
    m, n = np.shape(D)
    if n != 1: return
    plt.figure(0)
    if type(labels) != NoneType:
        label_set = list(set(flatten(labels.tolist())))
        for i in range(len(color)):
            index = np.nonzero(labels[:, 0] == label_set[i])[0]
            samples = D[index, :]
            xs = samples[:, 0]
            ys = np.zeros(np.shape(xs))
            c = color[i]
            plt.scatter(flatten(xs.tolist()), ys, c=c, edgecolor='')
    else:
        xs = D[:, 0]
        ys = np.zeros(np.shape(xs))
        plt.scatter(flatten(xs.tolist()), ys, c=color, edgecolor='')
    plt.title(title)
    if show: plt.show()


def plot2D(D, title='', show=False, color=None, labels=None):
    m, n = np.shape(D)
    if n != 2: return
    plt.figure(0)
    if type(labels) != NoneType:
        label_set = list(set(flatten(labels.tolist())))
        for i in range(len(color)):
            index = np.nonzero(labels[:, 0] == label_set[i])[0]
            samples = D[index, :]
            xs = samples[:, 0]
            ys = samples[:, 1]
            c = color[i]
            plt.scatter(flatten(xs.tolist()), flatten(ys.tolist()), c=c, edgecolor='')
    else:
        xs = D[:, 0]
        ys = D[:, 1]
        plt.scatter(flatten(xs.tolist()), flatten(ys.tolist()), c=color, edgecolor='')
    plt.title(title)
    if show: plt.show()


def PCA(D, n_components, calcu_eig_val_only=False):
    # 样本中心化
    Dmean = np.mean(D, axis=0)
    Dc = D - Dmean
    # 计算协方差矩阵
    m, n = np.shape(D)
    C = Dc.T * Dc / float(m)
    # 对协方差矩阵做特征值分解
    eig_val, eig_vec = np.linalg.eig(C)
    if calcu_eig_val_only: return eig_val
    # 取最大的d_个特征值所对应的特征向量
    eig_val_index = np.argsort(eig_val)
    eig_val_index = eig_val_index[:-(n_components + 1):-1]
    eig_vec_ = eig_vec[:, eig_val_index]
    # 将初始数据转换到低维空间
    D_ = Dc * eig_vec_
    reconD_ = D_ * eig_vec_.T + Dmean
    return D_, reconD_


# 在数据集1中的示例
D, _ = load_data('data1.txt')
D_, reconD_ = PCA(D, n_components=1)
plot2D(D, show=False, color='b', title='Origin 2D samples')
plot2D(reconD_, show=True, color='r', title='Origin 2D samples(blue) & Reconstruct 2D samples(red)')
plot1D(D_, show=True, color='r', title='1D samples transport from origin 2D samples')

# 在数据集2中的示例
D, labels = load_data('data2.txt', end=-1, label=-1)
D_, reconD_ = PCA(D, n_components=1)
label_set = list(set(flatten(labels.tolist())))
r = lambda: random.randint(0, 255)
colormap = ['#%02X%02X%02X' % (r(), r(), r()) for i in range(len(label_set))]
plot2D(D, show=False, title='Origin 2D samples', labels=labels, color=colormap)
plot2D(reconD_, show=True, title='Origin 2D samples & Reconstruct 2D samples', labels=labels, color=colormap)
plot1D(D_, show=True, title='1D samples transport from origin 2D samples', labels=labels, color=colormap)


# 在半导体数据集中的示例
def replace_nan_with_mean(D):
    m, n = np.shape(D)
    for i in range(n):
        meanVal = np.mean(D[np.nonzero(~np.isnan(D[:, i].A))[0], i])
        D[np.nonzero(np.isnan(D[:, i].A))[0], i] = meanVal
    return D


def plot_rate(eig_val, n_components=20):
    plt.figure()
    eig_sum = eig_val.sum()
    plt.plot(range(n_components), [eig_val[n] / eig_sum for n in range(n_components)])
    plt.xlabel(u'主成分数目')
    plt.ylabel(u'方差百分比')
    plt.show()


D, labels = load_data('secom.data')
D = replace_nan_with_mean(D)
eig_val = PCA(D, n_components=np.inf, calcu_eig_val_only=True)
print 'Origin D has %d dimensions.' % len(eig_val)
print
print 'eig_val = '
print eig_val
plot_rate(eig_val, n_components=20)
