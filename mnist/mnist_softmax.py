# coding=utf-8
import input_data
import tensorflow as tf


class MnistNet(object):
    def __init__(self, lr=0.1):
        self.lr = lr
        self.mnist = input_data.read_data_sets('handwrite_datasets', one_hot=True)
        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter("logs_new/", self.sess.graph)

    def _build_net(self):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        # with tf.name_scope('layer_1'):
        #     self.W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.3, name='W1'))
        #     self.b1 = tf.Variable(tf.constant(0.1, shape=[1, 256]), name='b1')
        #     self.l1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)
        # with tf.name_scope('layer_2'):
        #     self.W2 = tf.Variable(tf.random_normal([256, 64], stddev=0.3, name='W2'))
        #     self.b2 = tf.Variable(tf.constant(0.1, shape=[1, 64]), name='b2')
        #     self.l2 = tf.nn.relu(tf.matmul(self.l1, self.W2) + self.b2)
        # with tf.name_scope('layer_3'):
        #     self.W3 = tf.Variable(tf.random_normal([64, 32], stddev=0.3, name='W3'))
        #     self.b3 = tf.Variable(tf.constant(0.1, shape=[1, 32]), name='b3')
        #     self.l3 = tf.nn.relu(tf.matmul(self.l2, self.W3) + self.b3)
        # with tf.name_scope('layer_4'):
        #     self.W4 = tf.Variable(tf.random_normal([32, 10], stddev=0.3, name='W4'))
        #     self.b4 = tf.Variable(tf.constant(0.1, shape=[1, 10]), name='b4')
        #     self.l4 = tf.matmul(self.l3, self.W4) + self.b4
        with tf.name_scope('hidden'):
            self.W1 = tf.Variable(tf.random_normal([784, 10], stddev=0.3, name='W1'))
            self.b1 = tf.Variable(tf.constant(0.1, shape=[1, 10]), name='b1')
            self.l1 = tf.matmul(self.x, self.W1) + self.b1
        with tf.name_scope('softmax'):
            # self.y = tf.nn.softmax(self.l4, name='output')
            self.y = tf.nn.softmax(self.l1, name='output')
        with tf.name_scope('train'):
            self.y_ = tf.placeholder(tf.float32, [None, 10])
            self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y + 0.01))
            self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cross_entropy)
        with tf.name_scope('eval'):
            self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


if __name__ == '__main__':
    net = MnistNet()
    for i in range(10000):
        batch_xs, batch_ys = net.mnist.train.next_batch(100)
        # print net.sess.run(tf.log(net.y), {net.x: batch_xs, net.y_: batch_ys})
        # print net.sess.run(net.cross_entropy, {net.x: batch_xs, net.y_: batch_ys})
        net.sess.run(net.train_op, {net.x: batch_xs, net.y_: batch_ys})
        if i % 1000 == 0:
            print 'step ', i
            print net.sess.run(net.accuracy, feed_dict={net.x: net.mnist.test.images, net.y_: net.mnist.test.labels})
