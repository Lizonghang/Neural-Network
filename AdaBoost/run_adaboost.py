import numpy as np
import matplotlib.pyplot as plt
from compiler.ast import flatten


class Stump(object):
    def __init__(self, samples, labels, D):
        self.samples = samples
        self.labels = labels
        self.D = D
        self._build()

    def _build(self):
        m, n = np.shape(self.samples)
        step_num = 10.0
        best_stump = {}
        best_classify_result = None
        min_error = np.inf
        for i in range(n):
            range_min = self.samples[:, i].min()
            range_max = self.samples[:, i].max()
            step_size = (range_max - range_min)/step_num
            for j in range(-1, int(step_num) + 1):
                for inequal in ['lt', 'gt']:
                    thresh_val = (range_min + j*step_size)
                    classify_result = self.classify(self.samples, i, thresh_val, inequal)
                    error_array = np.matrix(np.ones((m, 1)))
                    error_array[classify_result == self.labels.T] = 0
                    e = float(self.D.T * error_array)
                    if e < min_error:
                        min_error = e
                        best_classify_result = classify_result.copy()
                        best_stump['attr'] = i
                        best_stump['thresh_val'] = thresh_val
                        best_stump['inequal'] = inequal
        self.best_stump = best_stump
        self.min_error = min_error
        self.best_classify_result = best_classify_result

    def classify(self, samples, attr, thresh_val, inequal):
        m, n = np.shape(samples)
        classify_result = np.ones((m, 1))
        if inequal == 'lt':
            classify_result[samples[:, attr] < thresh_val] = -1.0
        else:
            classify_result[samples[:, attr] > thresh_val] = -1.0
        return classify_result


class AdaBoost(object):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
        self.primary_classifiers = []
        m, n = np.shape(self.samples)
        self.D = np.matrix(np.ones((m, 1)) / m)

    def train(self, T=95):
        m, n = np.shape(self.samples)
        H = np.matrix(np.zeros((m, 1)))
        error_rate = 1.0
        for i in range(1, T + 1):
            stump = Stump(self.samples, self.labels, self.D)
            print 'Stump {0} error: {1}'.format(i, stump.min_error)
            stump.alpha = float(0.5 * np.log((1 - stump.min_error) / max(stump.min_error, 1e-16)))
            self.primary_classifiers.append(stump)
            D = np.multiply(self.D, np.exp(-1 * stump.alpha * np.multiply(self.labels.T, stump.best_classify_result)))
            self.D = D / D.sum()
            H += stump.alpha * stump.best_classify_result
            error_array = np.sign(H) != self.labels.T
            error_rate = float(error_array.sum()) / m
            print 'H(x) error: {0}'.format(error_rate)
            print
            if error_rate == 0.0:  break
        self.H = H

    def classify(self, samples):
        m, n = np.shape(samples)
        H = np.matrix(np.zeros((m, 1)))
        for i in range(len(self.primary_classifiers)):
            stump = self.primary_classifiers[i]
            classify_result = stump.classify(
                samples,
                stump.best_stump['attr'],
                stump.best_stump['thresh_val'],
                stump.best_stump['inequal']
            )
            H += stump.alpha * classify_result
        return np.sign(H)

    def plot_ROC(self):
        cursor = (1.0, 1.0)
        m, n = np.shape(self.samples)
        y_sum = 0.0
        positive_num = (self.labels == 1.0).sum()
        y_step = 1 / float(positive_num)
        x_step = 1 / float(m - positive_num)
        sorted_index = flatten(self.H.reshape((1, m)).argsort().tolist())
        figure = plt.figure()
        figure.clf()
        ax = plt.subplot(111)
        labels = flatten(self.labels.tolist())
        for index in sorted_index:
            if labels[index] == 1.0:
                delX = 0
                delY = y_step
            else:
                delX = x_step
                delY = 0
                y_sum += cursor[1]
            ax.plot([cursor[0], cursor[0] - delX], [cursor[1], cursor[1] - delY], c='b')
            cursor = (cursor[0] - delX, cursor[1] - delY)
        ax.plot([0, 1], [0, 1], 'b--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for AdaBoost')
        ax.axis([0, 1, 0, 1])
        print 'AUC = {0}'.format(y_sum * x_step)
        print
        plt.show()


def load_data(filename):
    samples = []
    labels = []
    with open(filename) as fp:
        for item in fp.readlines():
            _ = item.strip().split()
            samples.append([int(float(attr)) for attr in _[0: -1]])
            labels.append(int(float(_[-1])))
    return np.matrix(samples), np.matrix(labels)


if __name__ == '__main__':
    # train
    train_samples, train_labels = load_data('train.txt')
    boost = AdaBoost(train_samples, train_labels)
    boost.train()
    boost.plot_ROC()
    # test
    test_samples, test_labels = load_data('test.txt')
    m, n = np.shape(test_samples)
    error_array = boost.classify(test_samples) != test_labels.T
    print 'Test error: ', float(error_array.sum()) / m
