# coding=utf-8
import numpy as np
import random
import matplotlib.pyplot as plt
from compiler.ast import flatten


def plot(samples, u, cluster_assment=None, noncolor=True, loop_counter=0):
    if np.shape(samples)[1] != 2: raise ValueError('Only 2D array supported.')

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
    """
    K均值聚类

    缺点:可能收敛到局部极小值,在大规模数据集上收敛较慢.

    算法:
    1. 创建K个点作为起始质心(常是从样本中随机选择)
    2. 当任意一个点的簇分配结果发生改变时
    3.    对数据集中的每个数据点
    4.        对每个质心
    5.            计算质心与数据点之间的距离
    6.        将数据点分配到距其最近的簇
    7.    对每一个簇,计算簇中所有点的均值并将均值作为质心
    """
    def __init__(self, train_samples, log_and_plot=True):
        self.train_samples = train_samples
        self.log_and_plot = log_and_plot

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
        if self.log_and_plot: plot(self.train_samples, u, noncolor=True)
        # 记录簇更新状态
        cluster_changed = True
        # 均值向量或簇标记未更新时停止训练
        train_loop_counter = 0
        while cluster_changed:
            cluster_changed = False
            train_loop_counter += 1
            if self.log_and_plot: print 'Training Loop {0}'.format(train_loop_counter)

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
                cluster_i_index = np.nonzero(cluster_assment[:, 0] == i)[0]
                cluster_i_samples = self.train_samples[cluster_i_index]
                u[i, :] = np.mean(cluster_i_samples, axis=0)
                if self.log_and_plot: print 'Cluster {0} has {1} samples, error={2}'.format(i, np.shape(cluster_i_samples)[0], cluster_assment[cluster_i_index, 1].sum())

            if self.log_and_plot: print 'Total error: {0}'.format(cluster_assment[:, 1].sum())
            if self.log_and_plot: print
            # 绘制第train_loop_counter次训练得到的样本聚类
            if self.log_and_plot: plot(self.train_samples, u, cluster_assment, noncolor=False, loop_counter=train_loop_counter)

        return u, cluster_assment


class BisectingKmean(object):
    """
    二分K均值聚类

    优点:克服K均值聚类收敛于局部极小值的问题.

    算法:
    1.将所有点看作一个簇
    2.当簇数目小于K时
    3.对每一个簇
    4.    计算总误差
    5.    在给定的簇上进行K均值聚类(K=2)
    6.    计算将该簇一分为二后的总误差
    7.选择使得总误差最小的簇进行划分
    """
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
        # 计算初始质心位置
        init_center = np.mean(self.train_samples, axis=0).tolist()[0]
        center_list = [init_center]
        # 绘制初始样本集
        plot(self.train_samples, np.matrix(center_list[0]), noncolor=True)
        # 初始化簇索引,误差
        for j in range(m):
            cluster_assment[j, :] = 0, self._calc_minkowski_dist(self.train_samples[j, :], init_center, 2) ** 2

        while len(center_list) < K:
            lowest_SSE = np.inf

            # 得到最佳划分类别i,划分后两个质心best_split_u,划分后误差best_split_cluster_assment
            for i in range(len(center_list)):
                cluster_samples = self.train_samples[np.nonzero(cluster_assment[:, 0].A == i)[0], :]
                kmean = Kmean(cluster_samples, log_and_plot=False)
                split_u, split_cluster_assment = kmean.train(K=2)
                del kmean
                SSE_split = split_cluster_assment[:, 1].sum()
                SSE_not_split = cluster_assment[np.nonzero(cluster_assment[:, 0].A != i)[0], 1].sum()
                if SSE_not_split + SSE_split < lowest_SSE:
                    best_split_cluster = i
                    best_split_u = split_u
                    best_split_cluster_assment = split_cluster_assment.copy()
                    lowest_SSE = SSE_not_split + SSE_split

            # 根据最佳划分更新簇索引与误差
            best_split_cluster_assment[np.nonzero(best_split_cluster_assment[:, 0].A == 1)[0], 0] = len(center_list)
            best_split_cluster_assment[np.nonzero(best_split_cluster_assment[:, 0].A == 0)[0], 0] = best_split_cluster
            cluster_assment[np.nonzero(cluster_assment[:, 0].A == best_split_cluster)[0], :] = best_split_cluster_assment

            # 更新质心数组
            center_list[best_split_cluster] = best_split_u[0, :]
            center_list.append(best_split_u[1, :])

            # 绘制currK时的样本聚类
            plot(self.train_samples, np.matrix([flatten(center.tolist()) for center in center_list]), cluster_assment, noncolor=False, loop_counter=len(center_list) - 1)

        return np.matrix([flatten(center.tolist()) for center in center_list]), cluster_assment


if __name__ == '__main__':
    train_samples = load_data('data1.txt')
    kmean = Kmean(train_samples)
    u, cluster_assment = kmean.train(K=4)
    bikmean = BisectingKmean(train_samples)
    u, cluster_assment = bikmean.train(K=4)

    train_samples = load_data('data2.txt')
    kmean = Kmean(train_samples)
    u, cluster_assment = kmean.train(K=3)
    bikmean = BisectingKmean(train_samples)
    u, cluster_assment = bikmean.train(K=3)
