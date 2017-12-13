import pandas as pd
import numpy as np
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from types import NoneType

np.random.seed(1)


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


def make_submission(test_labels):
    test_labels *= 96.0
    test_labels = test_labels.clip(0, 96)

    lookup_table = pd.read_csv('dataset/IdLookupTable.csv')
    values = []

    cols = ["left_eye_center_x", "left_eye_center_y", "right_eye_center_x", "right_eye_center_y",
            "left_eye_inner_corner_x", "left_eye_inner_corner_y", "left_eye_outer_corner_x",
            "left_eye_outer_corner_y", "right_eye_inner_corner_x", "right_eye_inner_corner_y",
            "right_eye_outer_corner_x", "right_eye_outer_corner_y", "left_eyebrow_inner_end_x",
            "left_eyebrow_inner_end_y", "left_eyebrow_outer_end_x", "left_eyebrow_outer_end_y",
            "right_eyebrow_inner_end_x", "right_eyebrow_inner_end_y", "right_eyebrow_outer_end_x",
            "right_eyebrow_outer_end_y", "nose_tip_x", "nose_tip_y", "mouth_left_corner_x",
            "mouth_left_corner_y", "mouth_right_corner_x", "mouth_right_corner_y", "mouth_center_top_lip_x",
            "mouth_center_top_lip_y", "mouth_center_bottom_lip_x", "mouth_center_bottom_lip_y"]

    for index, row in lookup_table.iterrows():
        values.append((
            row['RowId'],
            test_labels[row.ImageId - 1][cols.index(row.FeatureName)],
        ))
    submission = pd.DataFrame(values, columns=('RowId', 'Location'))
    submission.to_csv('dataset/submission.csv', index=False)


if __name__ == '__main__':
    train_X, train_y = load_train_data()
    validation_X = train_X[:100, ...]
    validation_y = train_y[:100, ...]
    train_X = train_X[100:, ...]
    train_y = train_y[100:, ...]
    test_X = load_test_data()
    for i in range(validation_X.shape[0]):
        display(validation_X[i], y_true=validation_y[i], savefig=True)
