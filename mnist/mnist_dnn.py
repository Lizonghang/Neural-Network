# coding=utf-8
import input_data
import tensorflow as tf


class MnistNet(object):
    def __init__(self, lr=0.05):
        self.lr = lr
        self.mnist = input_data.read_data_sets('handwrite_datasets', one_hot=True)

        # Step 1: Define Your Network Model
        # ---------------------------------
        self._build_net()
        # ---------------------------------

        # Step 4: Create a Session and Initialize all the network variables.
        # ---------------------------------
        self.sess = tf.Session()
        # Optional: To Output summary log file
        self.writer = tf.summary.FileWriter("logs", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        # ---------------------------------

    def _build_net(self):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Step 1: Define Your Network Model
        # ---------------------------------
        with tf.name_scope('layer_1'):
            W1 = tf.Variable(tf.truncated_normal([784, 1024], mean=0.0, stddev=0.01, name='W1'))
            b1 = tf.Variable(tf.constant(0.0, shape=[1, 1024]), name='b1')
            l1 = tf.nn.relu(tf.matmul(self.x, W1) + b1, name='relu')
            # Optional: Output summary log file
            # tf.summary.histogram('layer_1/W1', W1)
            # tf.summary.histogram('layer_1/b1', b1)

        with tf.name_scope('dropout'):
            dropout_ = tf.nn.dropout(l1, keep_prob=self.keep_prob)

        with tf.name_scope('layer_2'):
            W2 = tf.Variable(tf.truncated_normal([1024, 10], mean=0.0, stddev=0.01, name='W2'))
            b2 = tf.Variable(tf.constant(0.0, shape=[1, 10]), name='b2')
            l2 = tf.matmul(dropout_, W2) + b2

        with tf.name_scope('softmax'):
            self.y = tf.nn.softmax(l2, name='output')
        # ---------------------------------

        # Step 2: Define Your Loss Function
        # ---------------------------------
        with tf.name_scope('loss'):
            self.y_ = tf.placeholder(tf.float32, [None, 10], name='y_')
            self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y + 0.01), name='loss')
        # ---------------------------------

        # Step 3: Define Your Optimizer
        # ---------------------------------
        with tf.name_scope('train'):
            self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cross_entropy)
        # ---------------------------------

        # Optional: Eval
        with tf.name_scope('eval'):
            self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name='accuracy')
            tf.summary.scalar('accuracy', self.accuracy)


if __name__ == '__main__':
    net = MnistNet()

    # Step 5: Load your dataset and start training
    for i in range(30000):
        batch_xs, batch_ys = net.mnist.train.next_batch(64)
        net.sess.run(net.train_op, feed_dict={
            net.x: batch_xs,
            net.y_: batch_ys,
            net.keep_prob: 0.5
        })

        # Optional: Output summary log file
        if i % 100 == 0:
            # summary = net.sess.run(tf.summary.merge_all(), feed_dict={net.x: batch_xs, net.y_: batch_ys})
            # net.writer.add_summary(summary, i)
            print 'step {0}, accuracy = {1}'.format(i, net.sess.run(net.accuracy, feed_dict={
                net.x: net.mnist.test.images,
                net.y_: net.mnist.test.labels,
                net.keep_prob: 1.0}))
