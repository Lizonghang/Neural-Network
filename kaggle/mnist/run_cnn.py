import tensorflow as tf
import numpy as np
import pandas as pd


IMAGE_SIZE = 28
NUM_CLASSES = 10
INITIAL_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_FACTOR = 0.1
BATCH_SIZE = 100
MOVING_AVERAGE_DECAY = 0.9999
MAX_TRAINING_STEP = 10000


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inference(images):
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 1, 32], stddev=0.01, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)

    # norm1
    norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # pool1
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 32, 64], stddev=0.01, wd=0.0)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        dim = 1
        for d in pool2.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool2, [BATCH_SIZE, dim])

        weights = _variable_with_weight_decay('weights', shape=[dim, 256], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[256, 64], stddev=0.04, wd=0.04)
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    # softmax output
    with tf.variable_scope('softmax_output') as scope:
        weights = _variable_with_weight_decay('weights', [64, NUM_CLASSES], stddev=1.0 / 64, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_output = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

    return softmax_output


def calculate_loss(logits, labels):
    # Reshape the labels into a dense Tensor of shape [BATCH_SIZE, NUM_CLASSES].
    sparse_labels = tf.reshape(labels, [BATCH_SIZE, 1])
    indices = tf.reshape(tf.range(BATCH_SIZE), [BATCH_SIZE, 1])
    concated = tf.concat([indices, sparse_labels], 1)
    dense_labels = tf.sparse_to_dense(concated, [BATCH_SIZE, NUM_CLASSES], 1.0, 0.0)
    # Calculate the average cross entropy loss across the barch.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=dense_labels, name='cross_entropy_per_sample')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='batch_mean_cross_entropy')
    # The total loss is defined as the cross entropy loss plus all of the weight decay terms(L2 loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(total_loss, global_step):
    # Consider decay the learning rate exponentially on the number of steps
    # use tf.train.exponential_decay()
    lr = INITIAL_LEARNING_RATE
    # Compute gradients.
    # optimizer = tf.train.GradientDescentOptimizer(lr)
    # optimizer = tf.train.RMSPropOptimizer(lr)
    optimizer = tf.train.AdamOptimizer(lr)
    grads = optimizer.compute_gradients(total_loss)
    # Apply gradients
    apply_gradients_op = optimizer.apply_gradients(grads, global_step=global_step)
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradients_op, variable_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op


if __name__ == '__main__':
    train_set = pd.read_csv('train.csv')
    train_labels = train_set['label']
    train_set.drop("label", axis=1, inplace=True)
    valid_set = train_set.iloc[40000:]
    valid_labels = train_labels.iloc[40000:]
    train_set = train_set.head(40000).copy()
    train_labels = train_labels.head(40000).copy()

    with tf.Graph().as_default():

        with tf.Session() as sess:

            global_step = tf.Variable(0, trainable=False)

            images = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1], name='batch_images')
            labels = tf.placeholder(tf.int32, [None, ], name='batch_labels')

            # Build a Graph that computes the logits predictions from the inference model.
            logits = inference(images)

            # Calculate loss.
            loss = calculate_loss(logits, labels)

            # Trains the model with one batch of samples and updates the model parameters.
            train_op = train(loss, global_step)

            sess.run(tf.global_variables_initializer())

            for step in xrange(MAX_TRAINING_STEP):
                # Get batch images.
                for batch_index in range(train_set.shape[0] / BATCH_SIZE):
                    batch_images = train_set.iloc[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE].astype(np.float32).values.reshape((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
                    batch_labels = train_labels.iloc[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE].astype(np.int32).tolist()
                    sess.run(train_op, feed_dict={images: batch_images, labels: batch_labels})

                if step % 100 == 0:
                    correct_counter = 0
                    for valid_batch_num in xrange(2000 / BATCH_SIZE):
                        batch_images_valid = valid_set.iloc[valid_batch_num * BATCH_SIZE: (valid_batch_num + 1) * BATCH_SIZE].astype(np.float32).values.reshape((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
                        batch_labels_valid = valid_labels.iloc[valid_batch_num * BATCH_SIZE: (valid_batch_num + 1) * BATCH_SIZE].astype(np.int32).tolist()
                        logits_valid_set = sess.run(logits, feed_dict={images: batch_images_valid})
                        predict = logits_valid_set.argmax(axis=1)
                        correct_counter += (np.array(batch_labels_valid) == np.array(predict)).sum()
                    accuracy = correct_counter / float(2000 / BATCH_SIZE * BATCH_SIZE)
                    print 'Accuracy = {0} at step {1}'.format(accuracy, step)
