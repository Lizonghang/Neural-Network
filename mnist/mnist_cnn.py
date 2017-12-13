# coding=utf-8
import tensorflow as tf
import input_data

mnist = input_data.read_data_sets('handwrite_datasets', one_hot=True)


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder('float', [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope('layer_1'):
    with tf.name_scope('conv_1'):
        W1 = weight_variable([5, 5, 1, 32])
        b1 = bias_variable([32])
        h1 = tf.nn.relu(conv2d(x_image, W1) + b1)
    with tf.name_scope('pool_1'):
        p1 = max_pool_2x2(h1)
    with tf.name_scope('summary_layer1'):
        tf.summary.histogram('W1', W1)
        tf.summary.histogram('b1', b1)
        tf.summary.histogram('h1', h1)
        tf.summary.histogram('p1', p1)

with tf.name_scope('layer_2'):
    with tf.name_scope('conv_2'):
        W2 = weight_variable([5, 5, 32, 64])
        b2 = bias_variable([64])
        h2 = tf.nn.relu(conv2d(p1, W2) + b2)
    with tf.name_scope('pool_2'):
        p2 = max_pool_2x2(h2)
    with tf.name_scope('summary_layer2'):
        tf.summary.histogram('W1', W2)
        tf.summary.histogram('b1', b2)
        tf.summary.histogram('h1', h2)
        tf.summary.histogram('p1', p2)

with tf.name_scope('full_connect'):
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    p2_flat = tf.reshape(p2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(p2_flat, W_fc1) + b_fc1)
    with tf.name_scope('summary_full_connect'):
        tf.summary.histogram('W_fc1', W_fc1)
        tf.summary.histogram('b_fc1', b_fc1)
        tf.summary.histogram('h_fc1', h_fc1)

with tf.name_scope('drop_out'):
    keep_prob = tf.placeholder('float')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    with tf.name_scope('summary_drop_out'):
        tf.summary.histogram('drop', h_fc1_drop)

with tf.name_scope('output_layer'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    with tf.name_scope('summary_output_layer'):
        tf.summary.histogram('W_fc2', W_fc2)
        tf.summary.histogram('b_fc2', b_fc2)
        tf.summary.histogram('y', y)

with tf.name_scope('loss'):
    y_ = tf.placeholder('float', [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    with tf.name_scope('summary_loss'):
        tf.summary.scalar('loss', cross_entropy)

with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

with tf.name_scope('eval'):
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    with tf.name_scope('summary_eval'):
        tf.summary.scalar('accuracy', accuracy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs', graph=sess.graph)

    for i in range(10000):
        batch = mnist.train.next_batch(64)
        sess.run(train_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        if i % 100 == 0:
            print 'step {0}, testing accuracy {1}'.format(i, sess.run(accuracy, feed_dict={
                x: mnist.test.images,
                y_: mnist.test.labels,
                keep_prob: 1.0
            }))

            summary = sess.run(tf.summary.merge_all(), feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            writer.add_summary(summary, i)
