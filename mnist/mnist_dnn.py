# coding=utf-8
import input_data
import tensorflow as tf


class MnistNet(object):
    def __init__(self, lr=0.05):
        self.lr = lr
        self.mnist = input_data.read_data_sets('handwrite_datasets', one_hot=True)
        self._build_net()
        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter("logs", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        with tf.name_scope('layer_1'):
            self.W1 = tf.Variable(tf.random_uniform([784, 10], -0.05, 0.05, name='W1'))
            self.b1 = tf.Variable(tf.constant(0.1, shape=[1, 10]), name='b1')
            self.l1 = tf.matmul(self.x, self.W1) + self.b1
            tf.summary.histogram('layer_1/W1', self.W1)
            tf.summary.histogram('layer_1/b1', self.b1)
        with tf.name_scope('softmax'):
            self.y = tf.nn.softmax(self.l1, name='output')
        with tf.name_scope('train'):
            self.y_ = tf.placeholder(tf.float32, [None, 10])
            self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y + 0.01))
            self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cross_entropy)
            tf.summary.scalar('loss', self.cross_entropy)
        with tf.name_scope('eval'):
            self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)


if __name__ == '__main__':
    net = MnistNet()
    for i in range(30000):
        batch_xs, batch_ys = net.mnist.train.next_batch(100)
        net.sess.run(net.train_op, {net.x: batch_xs, net.y_: batch_ys})
        if i % 1000 == 0:
            summary = net.sess.run(tf.summary.merge_all())
            net.writer.add_summary(summary, i)
            print 'step {0}, accuracy = {1}'.format(
                i,
                net.sess.run(net.accuracy, feed_dict={net.x: net.mnist.test.images, net.y_: net.mnist.test.labels})
            )
