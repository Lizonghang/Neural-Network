# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    data = []
    with open(filename) as fp:
        for line in fp.readlines():
            digits = [char for char in line.strip()]
            data.append(map(float, digits))
    return np.matrix(data)


def show_digits(D, thresh_val=0.5, show=True):
    n_samples, n_features = np.shape(D)
    for i in range(n_samples):
        for j in range(n_features):
            if D[i, j] > thresh_val: D[i, j] = 1.0
            else: D[i, j] = 0.0
    plt.figure()
    plt.imshow(D, cmap='gray')
    if show: plt.show()


def calc_n_components(S):
    S = S / np.sum(S)
    sum = 0.0
    for i in range(len(S)):
        sum += S[i]
        if sum >= 0.9: return i


def compress(D):
    U, S, VT = np.linalg.svd(D)
    # n_components = calc_n_components(S.copy())
    n_components = 3
    S_ = np.diag(S[:n_components])
    return U[:, :n_components], S_, VT[:n_components, :]


def construct(U, S, VT):
    return U * S * VT


D = load_data('digit.txt')
U_, S_, VT_ = compress(D)
D_ = construct(U_, S_, VT_)
show_digits(D, show=False)
show_digits(D_, show=False)
plt.show()
print 'Origin size: ', D.size
print 'Size after compress: ', U_.size + S_.size + VT_.size
