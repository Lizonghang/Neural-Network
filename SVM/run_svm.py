# coding=utf-8
from numpy import *
from matplotlib import pyplot as plt
from compiler.ast import flatten


def kernel_transform(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':    # 线性核函数
        K = X * A.T
    elif kTup[0] == 'rbf':  # 径向基核函数 Radial Basis Function
        """
                              -||x-y||^2
        高斯核函数:  k(x,y)=exp(——————————)
                                 2σ^2
        """
        for j in range(m):
            delta_row = X[j, :] - A
            K[j] = delta_row * delta_row.T
        K = exp(-1*K/(kTup[1]**2))  # 此处kTup[1]为高斯核中的σ参数
    else:
        raise NameError('unrecognized kernel')
    return K


class SVMDB(object):
    def __init__(self, samples, labels, C, tol, kTup):
        self.X = samples
        self.labels = labels
        self.C = C
        self.tol = tol  # 可容忍度,决定可容忍间隔带范围
        self.m = shape(samples)[0]  # 样本总数
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))   # 误差缓存,第一列为有效标志位
        self.K = mat(zeros((self.m, self.m)))   # 核矩阵,为半正定矩阵
        for i in range(self.m):
            self.K[:, i] = kernel_transform(self.X, self.X[i, :], kTup)  # kTup存储核函数的信息,第一个参数为核函数类型,其他参数为可选


def load_samples_and_labels(filename):
    samples = []
    labels = []
    with open(filename) as fp:
        for line in fp.readlines():
            item = line.strip().split('\t')
            samples.append([float(item[0]), float(item[1])])
            labels.append(float(item[2]))
    return samples, labels


def clip_alpha(aj, H, L):
    return max(min(aj, H), L)


def calculate_Ei(db, i):
    # f_Xi = float(multiply(db.alphas, db.labels).T * (db.X * db.X[i, :].T) + db.b)
    # 使用核函数的模型运算
    f_Xi = float(multiply(db.alphas, db.labels).T * db.K[:, i] + db.b)
    Ei = f_Xi - float(db.labels[i])
    return Ei


def random_select_j(i, m):
    j_list = range(0, m)
    j_list.remove(i)
    j = random.choice(j_list)
    return j


def select_j(i, db, Ei):
    max_k = -1
    max_deltaE = 0
    Ej = 0
    db.eCache[i] = [1, Ei]
    valid_eCache_list = nonzero(db.eCache[:, 0].A)[0]
    if len(valid_eCache_list) > 1:
        for k in valid_eCache_list:
            if k == i:
                continue
            Ek = calculate_Ei(db, k)
            deltaE = abs(Ei - Ek)
            if deltaE > max_deltaE:
                max_k = k
                max_deltaE = deltaE
                Ej = Ek
        return max_k, Ej
    else:
        j = random_select_j(i, db.m)
        Ej = calculate_Ei(db, j)
    return j, Ej


def update_Ek(db, k):
    Ek = calculate_Ei(db, k)
    db.eCache[k] = [1, Ek]


def innerL(i, db):
    Ei = calculate_Ei(db, i)
    # 判断样本是否在可容忍的间隔带内
    if ((db.labels[i] * Ei < -db.tol) and (db.alphas[i] < db.C)) or ((db.labels[i] * Ei > db.tol) and (db.alphas[i] > 0)):
        # 选择与样本i间隔最大的样本j
        j, Ej = select_j(i, db, Ei)

        ai_old = db.alphas[i].copy()
        aj_old = db.alphas[j].copy()

        """
        如果aj_new > H, 则aj_new = H;如果aj_new < L,则aj_new = L.其中H和L分别为aj_new的上下界,有:

        H = min(C, C + aj_old - ai_old)   s.t. yi * yj = -1
        H = min(C, aj_old + ai_old)       s.t. yi * yj = 1

        L = max(0, aj_old - ai_old)       s.t. yi * yj = -1
        L = max(0, ai_old + aj_old - C)   s.t. yi * yj = 1

        这一约束的意义在于使得ai_new和aj_new均位于矩形域[0,C]x[0,C]中.
        """
        if db.labels[i] != db.labels[j]:
            L = max(0, db.alphas[j] - db.alphas[i])
            H = min(db.C, db.C + db.alphas[j] - db.alphas[i])
        else:
            L = max(0, db.alphas[j] + db.alphas[i] - db.C)
            H = min(db.C, db.alphas[j] + db.alphas[i])
        if L == H:
            print "L == H"
            return 0

        # eta = 2.0 * db.X[i, :] * db.X[j, :].T - db.X[i, :] * db.X[i, :].T - db.X[j, :] * db.X[j, :].T
        # 使用核函数计算aj最优修改量:
        eta = 2.0 * db.K[i, j] - db.K[i, i] - db.K[j, j]
        if eta >= 0:
            print "eta >= 0"
            return 0

        # aj最优更新量 = yj * (Ei - Ej) / (Kii + Kjj - 2 * Kij)
        db.alphas[j] -= db.labels[j] * (Ei - Ej) / eta
        db.alphas[j] = clip_alpha(db.alphas[j], H, L)
        update_Ek(db, j)
        if abs(db.alphas[j] - aj_old) < 1e-5:
            print "aj almost not changed"
            return 0

        # ai更新量 = yi * yj * (aj_old - aj_new)
        db.alphas[i] += db.labels[j] * db.labels[i] * (aj_old - db.alphas[j])
        update_Ek(db, i)

        # 更新b
        # b1 = db.b - Ei - db.labels[i] * (db.alphas[i] - ai_old) * db.X[i, :] * db.X[i, :].T - db.labels[j] * (db.alphas[j] - aj_old) * db.X[i, :] * db.X[j, :].T
        # b2 = db.b - Ej - db.labels[i] * (db.alphas[i] - ai_old) * db.X[i, :] * db.X[j, :].T - db.labels[j] * (db.alphas[j] - aj_old) * db.X[j, :] * db.X[j, :].T
        # 使用核函数更新b
        b1 = db.b - Ei - db.labels[i] * (db.alphas[i] - ai_old) * db.K[i, i] - db.labels[j] * (db.alphas[j] - aj_old) * db.K[i, j]
        b2 = db.b - Ej - db.labels[i] * (db.alphas[i] - ai_old) * db.K[i, j] - db.labels[j] * (db.alphas[j] - aj_old) * db.K[j, j]
        if 0 < db.alphas[i] < db.C:
            db.b = b1
        elif 0 < db.alphas[j] < db.C:
            db.b = b2
        else:
            db.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smo(samples, labels, C, tol, max_iter, kTup=('lin', 0)):
    samples = mat(samples)
    labels = mat(labels).transpose()
    db = SVMDB(samples, labels, C, tol, kTup)

    sigma = 1.3
    err = 0.001
    m, n = shape(samples)
    E = {}
    for iter in range(max_iter):
        print "iteration number: %d" % iter

        violate_most_value = 0
        violate_most_i = -1

        # 选择违反KKT条件最严重的样本
        for i in range(m):
            f_xi = float(multiply(db.alphas, db.labels).T * kernel_transform(samples, samples[i, :], ('rbf', sigma))) + db.b
            E[i] = float(f_xi - labels[i])

            # 违反KKT条件:
            # ai = 0        >> yi * f_xi >= 1-e             >>  yi * f_xi < 1-e
            # 0 < ai < C    >> 1-e <= yi * f_xi <= 1+e      >>  yi * f_xi < 1-e or yi * f_xi > 1+e
            # ai = C        >> yi * f_xi <= 1+e             >>  yi * f_xi > 1+e
            if db.alphas[i] == 0 and db.labels[i] * f_xi < 1-err:
                err_abs = abs(1 - err - db.labels[i] * f_xi)
                if abs(1 - err - db.labels[i] * f_xi) > violate_most_value:
                    violate_most_value = err_abs
                    violate_most_i = i
            if 0 < db.alphas[i] < C and (db.labels[i] * f_xi < 1-err or db.labels[i] * f_xi > 1 + err):
                err_abs = max(abs(1 - err - db.labels[i] * f_xi), abs(1 + err - db.labels[i] * f_xi))
                if err_abs > violate_most_value:
                    violate_most_value = err_abs
                    violate_most_i = i
            if db.alphas[i] == C and db.labels[i] * f_xi > 1 + err:
                err_abs = abs(1 + err - db.labels[i] * f_xi)
                if err_abs > violate_most_value:
                    violate_most_value = err_abs
                    violate_most_i = i

        innerL(violate_most_i, db)
    return db.b, db.alphas


"""
def smo(samples, labels, C, tol, max_iter, kTup=('lin', 0)):
    db = SVMDB(mat(samples), mat(labels).transpose(), C, tol, kTup)

    iter = 0
    entire_set = True
    alpha_pairs_changed = 0

    # 当迭代次数超过指定最大迭代值,或遍历整个集合都未对任意alpha pair进行修改,就退出循环
    while iter < max_iter and (alpha_pairs_changed > 0 or entire_set):
        alpha_pairs_changed = 0
        if entire_set:
            for i in range(db.m):
                alpha_pairs_changed += innerL(i, db)
                print "fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alpha_pairs_changed)
            iter += 1
        else:  # 注意:此处没有选择违反KKT最严重的样本点,而是对满足0<ai<C时yi*f(xi)-1≠0的所有样本点逐一选择更新
            non_bound_i_list = nonzero((db.alphas.A > 0) * (db.alphas.A < C))[0]
            for i in non_bound_i_list:
                alpha_pairs_changed += innerL(i, db)
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alpha_pairs_changed)
            iter += 1

        if entire_set:
            entire_set = False
        elif alpha_pairs_changed == 0:
            entire_set = True

        print "iteration number: %d" % iter

    return db.b, db.alphas
"""


def calculate_W(alphas, samples, labels):
    X = mat(samples)
    labels = mat(labels).transpose()
    m, n = shape(X)
    W = zeros((n, 1))
    for i in range(m):
        W += multiply(alphas[i] * labels[i], X[i, :].T)  # 实际仅支持向量参与计算
    return W


def plot_samples(samples, labels, support_vector=False):
    samples_classA = []
    samples_classB = []
    for i in range(len(labels)):
        if labels[i] == 1:
            samples_classA.append(samples[i])
        else:
            samples_classB.append(samples[i])

    x1_classA = [sample[0] for sample in samples_classA]
    x2_classA = [sample[1] for sample in samples_classA]
    if support_vector:
        plt.plot(x1_classA, x2_classA, 'rs')
    else:
        plt.plot(x1_classA, x2_classA, 'ro')

    x1_classB = [sample[0] for sample in samples_classB]
    x2_classB = [sample[1] for sample in samples_classB]
    if support_vector:
        plt.plot(x1_classB, x2_classB, 'bs')
    else:
        plt.plot(x1_classB, x2_classB, 'bo')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axis([-1.5, 1.5, -1.5, 1.5])
    plt.draw()


def eval_rbf(C=200, tol=0.0001, max_iter=10000, sigma=1.3):
    # 对训练集评估
    samples, labels = load_samples_and_labels('rbf_train.txt')
    plot_samples(samples, labels)
    b, alphas = smo(samples, labels, C, tol, max_iter, ('rbf', sigma))
    W = calculate_W(alphas, samples, labels)

    samples = mat(samples)
    labels = mat(labels).transpose()

    # 仅使用支持向量进行评估
    support_vector_index = nonzero(alphas.A > 0)[0]
    support_vector_samples = samples[support_vector_index]
    support_vector_labels = labels[support_vector_index]
    print "there are %d Support Vectors" % shape(support_vector_samples)[0]
    plot_samples(support_vector_samples.tolist(), flatten(support_vector_labels.tolist()), support_vector=True)

    # 计算训练集错误率
    m, n = shape(samples)
    error_count = 0
    for i in range(m):
        kernel_result = kernel_transform(support_vector_samples, samples[i, :], ('rbf', sigma))
        predict = kernel_result.T * multiply(support_vector_labels, alphas[support_vector_index]) + b
        if sign(predict) != sign(labels[i]):
            error_count += 1
    print "the training error rate is: %f%%" % (float(error_count) / m * 100)

    # 对测试集评估
    samples, labels = load_samples_and_labels('rbf_test.txt')
    samples = mat(samples)
    labels = mat(labels).transpose()

    # 计算测试集错误率
    m, n = shape(samples)
    error_count = 0
    for i in range(m):
        kernel_result = kernel_transform(support_vector_samples, samples[i, :], ('rbf', sigma))
        predict = kernel_result.T * multiply(support_vector_labels, alphas[support_vector_index]) + b
        if sign(predict) != sign(labels[i]):
            error_count += 1
    print "the test error rate is: %f%%" % (float(error_count) / m * 100)

    plt.show()


if __name__ == '__main__':
    eval_rbf(max_iter=100)
