import numpy as np
import random
import re


class NaiveBayes(object):

    def __init__(self, train_set, train_class, test_set, test_class):
        self.train_set = train_set
        self.train_class = train_class
        self.test_set = test_set
        self.test_class = test_class

        self.word_set = None
        self.p0_vector = None
        self.p1_vector = None
        self.p_class1 = None

    def _create_word_set(self, doc_set):
        word_set = set([])
        for doc in doc_set:
            word_set = word_set | set(doc)
        self.word_set = list(word_set)

    def _convert_words_to_vector_bag(self, input_words):
        vector = [0] * len(self.word_set)
        for word in input_words:
            if word in self.word_set:
                vector[self.word_set.index(word)] += 1
        return vector

    def _convert_list_to_matrix(self, list_data, list_class):
        matrix_data = []
        matrix_class = np.array(list_class)
        for i in range(len(list_data)):
            matrix_data.append(self._convert_words_to_vector_bag(list_data[i]))
        return matrix_data, matrix_class

    def update_data(self, train_set, train_class, test_set, test_class):
        self.train_set = train_set
        self.train_class = train_class
        self.test_set = test_set
        self.test_class = test_class

    def train(self):
        self._create_word_set(self.train_set)

        matrix_data, matrix_class = self._convert_list_to_matrix(self.train_set, self.train_class)

        p0_molecular = np.ones(len(self.word_set))
        p1_molecular = np.ones(len(self.word_set))
        p0_denominator = 2.0
        p1_denominator = 2.0

        for i in range(len(matrix_data)):
            if matrix_class[i] == 1:
                p1_molecular += matrix_data[i]
                p1_denominator += sum(matrix_data[i])
            else:
                p0_molecular += matrix_data[i]
                p0_denominator += sum(matrix_data[i])
        self.p0_vector = np.log(p0_molecular / p0_denominator)
        self.p1_vector = np.log(p1_molecular / p1_denominator)
        self.p_class1 = (np.sum(matrix_class) + 1) / float(len(matrix_class) + 2.0)

    def classify(self, doc):
        doc = self._convert_words_to_vector_bag(doc)
        p1 = np.sum(doc * self.p1_vector) + np.log(self.p_class1)
        p0 = np.sum(doc * self.p0_vector) + np.log(1.0 - self.p_class1)
        result = 1 if p1 > p0 else 0
        return result

    def eval(self, log=False):
        err_num = 0
        for i in range(len(self.test_set)):
            eval_result = self.classify(self.test_set[i])
            if log:
                print 'iter {0}: {1}'.format(i+1, self.test_set[i])
                print 'eval class: {0}, real class: {1}'.format(eval_result, self.test_class[i])
            if eval_result != self.test_class[i]:
                err_num += 1
                if log:
                    print 'error!'
            if log:
                print
        error_rate = float(err_num) / len(self.test_set)
        if log:
            print 'error rate: ', error_rate
            print
        return error_rate


if __name__ == '__main__':
    email_list = []
    class_list = []
    for i in range(1, 26):
        email_list.append([word.lower() for word in re.split(r'\W*', open('email/spam/%d.txt' % i).read()) if len(word) > 2])
        class_list.append(1)
        email_list.append([word.lower() for word in re.split(r'\W*', open('email/ham/%d.txt' % i).read()) if len(word) > 2])
        class_list.append(0)

    eval_results = []
    for i in range(100):
        test_email_index = random.sample(range(50), 10)
        train_email_index = [index for index in range(50) if index not in test_email_index]

        test_email_set = [email_list[i] for i in test_email_index]
        test_class_set = [class_list[i] for i in test_email_index]
        train_email_set = [email_list[i] for i in train_email_index]
        train_class_set = [class_list[i] for i in train_email_index]

        if 'bayes' in globals():
            bayes.update_data(train_email_set, train_class_set, test_email_set, test_class_set)
        else:
            bayes = NaiveBayes(train_email_set, train_class_set, test_email_set, test_class_set)

        bayes.train()

        result = bayes.eval()

        eval_results.append(result)

    print 'average error rate: {0}%'.format(sum(eval_results) / float(len(eval_results)) * 100)
