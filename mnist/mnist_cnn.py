# coding=utf-8
import tensorflow as tf
import input_data
import time

mnist = input_data.read_data_sets('handwrite_datasets', one_hot=True)


# 权重初始化.权重矩阵在初始化时应加入少量噪声来打破对称性以及避免0梯度.
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


# 由于使用的是ReLU神经元,使用一个较小的正数初始化偏置项,以避免神经元节点输出恒为0.
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


# 使用vinilla版本,卷积使用1步长,扩展边距保持输入与输出同样大小
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化使用2x2的最大池化
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# x 存储了N行784(28x28)列图像数据,并将其转换为28x28x1(长宽深度)的4维数据集,以做二维卷积运算
x = tf.placeholder('float', [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积层+池化层.由28x28x1卷积后为28x28x32,池化后为14x14x32
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积层+池化层.由14x14x32卷积后为14x14x64,池化后为7x7x64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层.假设该层有1024个神经元,共计有7x7x64个权值变量+1024个偏置变量.需要将第二层池化层输出reshape为向量参与矩阵乘法
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout层.为了减少过拟合.在输出层之前加入.keep_prob表示保留率.训练过程中启动dropout,测试过程中关闭dropout.
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层.该层与Dropout层全连接,使用softmax模型进行分类概率估计.
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# y_ 记录了期望了输出分类,以one-hot形式存储
y_ = tf.placeholder('float', [None, 10])

# 使用交叉熵代价函数进行梯度下降与BP
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 使用测试数据估计分类精度
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# 图的生成与初始化
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 开始训练
with tf.device("/gpu:1"):
    for i in range(1001):
        batch = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print 'step {num}, training accuracy {accuracy}'.format(num=i, accuracy=train_accuracy)

# 模型评估
print 'test accuracy {0}'.format(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
sess.close()
