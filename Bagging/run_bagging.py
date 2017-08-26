import random
import numpy as np
from compiler.ast import flatten


class Stump(object):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
        m, n = np.shape(self.samples)
        self.D = np.matrix(np.ones((m, 1)) / m)
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


class Bagging(object):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
        self.primary_classifiers = []

    def _sampling(self, M):
        m, n = np.shape(self.samples)
        sample_index = []
        for i in range(M):
            sample_index.append(random.choice(range(m)))
        samples_M = [self.samples.tolist()[index] for index in sample_index]
        labels_M = [flatten(self.labels.tolist())[index] for index in sample_index]
        return np.matrix(samples_M), np.matrix(labels_M)

    def train(self, T=50):
        m, n = np.shape(self.samples)
        for i in range(1, T + 1):
            samples_M, labels_M = self._sampling(m/2)
            stump = Stump(samples_M, labels_M)
            print 'Stump {0} error: {1}'.format(i, stump.min_error)
            self.primary_classifiers.append(stump)

    def classify(self, samples=None, use_test_set=False):
        classify_result = []
        vote = []
        for stump in self.primary_classifiers:
            if not use_test_set:
                samples = stump.samples
            vote.append(flatten(stump.classify(
                samples,
                stump.best_stump['attr'],
                stump.best_stump['thresh_val'],
                stump.best_stump['inequal']
            ).tolist()))
        vote = np.matrix(vote)
        m, n = np.shape(vote)
        for i in range(n):
            attrs = flatten(vote[:, i].tolist())
            positive_counter = 0
            for attr in attrs:
                if attr == 1.0:
                    positive_counter += 1
            if positive_counter > m / 2.0:
                classify_result.append(1.0)
            else:
                classify_result.append(-1.0)
        return classify_result


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
    bagging = Bagging(train_samples, train_labels)
    bagging.train()
    # eval
    test_samples, test_labels = load_data('test.txt')
    classify_result = bagging.classify(test_samples, use_test_set=True)
    test_labels = flatten(test_labels.tolist())
    error_counter = 0
    for i in range(len(classify_result)):
        if test_labels[i] != classify_result[i]:
            error_counter += 1
    print 'Test error: {0}'.format(float(error_counter) / len(classify_result))
