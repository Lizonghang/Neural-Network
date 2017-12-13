import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from types import NoneType


def load_train_data():
    print 'Loading train data ...'
    df = pd.read_csv('dataset/training.csv').dropna()
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' ') / 255.0)
    X = np.vstack(df['Image'])
    X = X.reshape((-1, 96, 96, 1))
    y = df[df.columns[:-1]].values / 96.0
    X, y = shuffle(X, y)
    return X, y


def load_test_data():
    print 'Loading test data ...'
    df = pd.read_csv('dataset/test.csv')
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' ') / 255.0)
    X = np.vstack(df['Image'])
    X = X.reshape((-1, 96, 96, 1))
    return X


def display(X, y_true=None, y_pred=None, savefig=False):
    import time
    plt.figure()
    plt.imshow(X.reshape((96, 96)), cmap='gray')
    plt.axis('off')
    if type(y_true) is not NoneType:
        plt.scatter(y_true[0::2] * 96.0, y_true[1::2] * 96.0, c='y', marker='o')
    if type(y_pred) is not NoneType:
        y_pred = y_pred.clip(0, 1)
        plt.scatter(y_pred[0::2] * 96.0, y_pred[1::2] * 96.0, c='r', marker='x')
    if savefig:
        if not os.path.exists('img'):
            os.mkdir('img')
        plt.savefig('img/{}.png'.format(int(time.time()*100)))
    else:
        plt.show()
    plt.close()


def batch_display(X, y_true=None, y_pred=None, savefig=False):
    import time
    X = X.reshape((4, 96, 96))
    im = np.zeros((96*2, 96*2))
    for i in range(2):
        for j in range(2):
            im[i*96:(i+1)*96, j*96:(j+1)*96] = X[i*2+j]
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    if type(y_true) is not NoneType:
        for i in range(2):
            for j in range(2):
                plt.scatter(y_true[i*2+j, 0::2]*96.0 + j*96.0, y_true[i*2+j, 1::2]*96.0 + i*96.0, c='y', marker='o', s=10)
    if type(y_pred) is not NoneType:
        y_pred = y_pred.clip(0, 1)
        for i in range(2):
            for j in range(2):
                plt.scatter(y_pred[i*2+j, 0::2]*96.0 + j*96.0, y_pred[i*2+j, 1::2]*96.0 + i*96.0, c='r', marker='x', s=10)
    if savefig:
        plt.savefig('runlog/{}.png'.format(int(time.time()*100)))
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    train_X, train_y = load_train_data()
    validation_X = train_X[:100, ...]
    validation_y = train_y[:100, ...]
    train_X = train_X[100:, ...]
    train_y = train_y[100:, ...]
    test_X = load_test_data()
    for i in range(validation_X.shape[0]):
        display(validation_X[i], y_true=validation_y[i], savefig=True)
