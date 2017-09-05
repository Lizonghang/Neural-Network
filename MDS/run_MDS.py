# coding=utf-8
import random
import numpy as np
import matplotlib.pyplot as plt

# D为原始样本距离矩阵
D = np.array([[0, 411, 213, 219, 296, 397],
              [411, 0, 204, 203, 120, 152],
              [213, 204, 0, 73, 136, 245],
              [219, 203, 73, 0, 90, 191],
              [296, 120, 136, 90, 0, 109],
              [397, 152, 245, 191, 109, 0]])

m = np.shape(D)[1]

# 根据D计算矩阵B
B = np.array(np.zeros((m, m)))
for i in range(m):
    for j in range(m):
        B[i, j] = B[j, i] = -0.5 * (
            D[i, j] ** 2
            - 1.0 / m * np.dot(D[i, :], D[i, :])
            - 1.0 / m * np.dot(D[:, j], D[:, j])
            + 1.0 / (m ** 2) * (D ** 2).sum()
        )

# 对矩阵B做特征值分解
eig_val, eig_vec = np.linalg.eig(B)

# 取eigVal_为d_个最大特征值所构成的对角矩阵,eigVec_为相应的特征向量矩阵
# 注: 取出的维度对应的特征值不能为负
d_ = 2

eig_map = {}
for i in range(m):
    eig_map[eig_val[i]] = eig_vec[:, i]

eig_val_ = np.zeros((d_,))
eig_vec_ = np.zeros((m, d_))

for d in range(d_):
    val = sorted(eig_map.keys(), reverse=True)[d]
    if val < 0: raise ValueError("Dimension %d eig_val %s less than zero" % (d + 1, val))
    vec = eig_map[val]
    eig_val_[d] = val
    eig_vec_[:, d] = vec

# 计算样本的低维坐标
X = np.dot(eig_vec_, np.diag(np.sqrt(eig_val_)))

print '降维到%s维后的样本坐标为:' % np.shape(X)[1]
print X


# 绘制2D降维样本点
def plot2D(X):
    m, n = np.shape(X)

    plt.figure()

    # 绘制样本点
    xs = X[:, 0]
    ys = X[:, 1]
    plt.scatter(xs, ys, c='r', edgecolor='', s=40)

    # 绘制连线与距离
    for i in range(m):
        for j in range(i + 1, m):
            # 生成线条与文本颜色
            r = lambda: random.randint(0, 255)
            color = '#%02X%02X%02X' % (r(), r(), r())
            # 绘制连线
            xs = [X[i][0], X[j][0]]
            ys = [X[i][1], X[j][1]]
            plt.plot(xs, ys, c=color)
            # 绘制距离
            center = [(X[i][0] + X[j][0]) / 2.0, (X[i][1] + X[j][1]) / 2.0]
            dist_ = np.linalg.norm(X[i] - X[j])
            plt.text(center[0], center[1], '%.1f' % dist_, color=color)

    plt.title('Samples after MDS')
    plt.show()

if np.shape(X)[1] == 2: plot2D(X)
