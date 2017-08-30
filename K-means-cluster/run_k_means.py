# coding=utf-8
import numpy as np
import random
import matplotlib.pyplot as plt


def plot(samples, u, cluster_assment=None, noncolor=True, loop_counter=0):
    if np.shape(samples)[1] != 2:  raise ValueError('Only 2D array supported.')

    plt.close()
    plt.figure(0)

    if noncolor:
        xs_center = u[:, 0]
        ys_center = u[:, 1]
        xs = samples[:, 0]
        ys = samples[:, 1]
        plt.scatter(xs.tolist(), ys.tolist(), c='r', edgecolors='')
        plt.scatter(xs_center.tolist(), ys_center.tolist(), s=40, marker='s', c='b', edgecolors='')
    else:
        K = np.shape(u)[0]
        color_map = ["r", "g", "c", "m", "y", "k"]
        if K > len(color_map):
            raise ValueError("Only K<={0} supported.".format(len(color_map)))

        for i in range(K):
            color = color_map[i]
            cluster_i_index = np.nonzero(cluster_assment[:, 0] == i)[0]
            xs = samples[cluster_i_index, 0]
            ys = samples[cluster_i_index, 1]
            plt.scatter(xs.tolist(), ys.tolist(), c=color, edgecolors='')

            x_center = u[i, 0]
            y_center = u[i, 1]
            plt.scatter(x_center.tolist(), y_center.tolist(), s=40, marker='s', c=color, edgecolors='')

    plt.title(u"Loop {0} times".format(loop_counter))
    plt.show()


def load_data(filename):
    samples = []
    with open(filename) as fp:
        for item in fp.readlines():
            _ = item.strip().split()
            samples.append(map(float, _))
    return np.matrix(samples)


class Kmean(object):
    def __init__(self, train_samples):
        self.train_samples = train_samples

    def _calc_minkowski_dist(self, xi, xj, p):
        return np.power(np.sum(np.power(np.abs(xi - xj), p)), 1.0 / p)

    def _gen_random_center(self, K):
        m, n = np.shape(self.train_samples)
        u = np.matrix(np.zeros((K, n)))
        u_index = 0
        for sample_index in random.sample(range(m), K):
            u[u_index, :] = self.train_samples[sample_index, :]
            u_index += 1
        return u

    def train(self, K):
        m, n = np.shape(self.train_samples)
        # 记录簇索引, 误差
        cluster_assment = np.matrix(np.zeros((m, 2)))
        # 随机选择K个样本作为初始均值向量
        u = self._gen_random_center(K=K)
        # 绘制初始样本集
        plot(self.train_samples, u, noncolor=True)
        # 记录簇更新状态
        cluster_changed = True
        # 均值向量或簇标记未更新时停止训练
        train_loop_counter = 0
        while cluster_changed:
            cluster_changed = False
            train_loop_counter += 1
            print 'Training Loop {0}'.format(train_loop_counter)

            for j in range(m):
                """
                1.计算样本xj与各均值向量ui的距离
                2.根据距离最近的均值向量确定xj的簇标记(记录于cluster_assment中)
                3.将样本xj划入相应的簇
                """
                min_dist = np.inf
                best_cluster_index = -1
                for i in range(K):
                    dist = self._calc_minkowski_dist(self.train_samples[j, :], u[i, :], 2)
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster_index = i
                # 检查均值向量或簇标记是否更新
                if cluster_assment[j, 0] != best_cluster_index:  cluster_changed = True
                # 更新样本所属簇标记与误差值
                cluster_assment[j, :] = int(best_cluster_index), min_dist ** 2

            for i in range(K):
                """
                计算并更新均值向量u
                """
                cluster_i_samples = self.train_samples[np.nonzero(cluster_assment[:, 0] == i)[0]]
                u[i, :] = np.mean(cluster_i_samples, axis=0)
                print 'Cluster {0} has {1} samples'.format(i, np.shape(cluster_i_samples)[0])

            print 'Total error: {0}'.format(cluster_assment[:, 1].sum())
            print

            # 绘制第train_loop_counter次训练得到的样本聚类
            # plot(self.train_samples, u, cluster_assment, noncolor=True, loop_counter=train_loop_counter)
            plot(self.train_samples, u, cluster_assment, noncolor=False, loop_counter=train_loop_counter)

        return u, cluster_assment


if __name__ == '__main__':
    train_samples = load_data('data.txt')
    kmean = Kmean(train_samples)
    u, cluster_assment = kmean.train(K=4)
