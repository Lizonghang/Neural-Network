import tensorflow as tf
import numpy as np
import pandas as pd


IMAGE_SIZE = 28
NUM_CLASSES = 10
INITIAL_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_FACTOR = 0.1
BATCH_SIZE = 1
MOVING_AVERAGE_DECAY = 0.9999
MAX_TRAINING_STEP = 10000
TRAIN_NUM = 80
VALID_NUM = 20


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def inference(images):
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_on_cpu('weights', shape=[5, 5, 1, 32], initializer=tf.truncated_normal_initializer(1e-4))
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.tanh(bias, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_on_cpu('weights', shape=[5, 5, 32, 64], initializer=tf.truncated_normal_initializer(1e-4))
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.tanh(bias, name=scope.name)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        dim = 1
        for d in pool2.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool2, [-1, dim])

        weights = _variable_on_cpu('weights', shape=[dim, 256], initializer=tf.truncated_normal_initializer(1e-4))
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        local3 = tf.nn.tanh(tf.matmul(reshape, weights) + biases, name=scope.name)

    # dropout
    with tf.variable_scope('dropout') as scope:
        dropout = tf.nn.dropout(local3, keep_prob=0.8, name=scope.name)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_on_cpu('weights', shape=[256, NUM_CLASSES], initializer=tf.truncated_normal_initializer(1e-4))
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        local4 = tf.nn.tanh(tf.matmul(dropout, weights) + biases, name=scope.name)

    # softmax output
    with tf.variable_scope('softmax_output') as scope:
        softmax_output = tf.nn.softmax(local4, name=scope.name)

    return softmax_output


def calculate_loss(logits, labels):
    # Reshape the labels into a dense Tensor of shape [BATCH_SIZE, NUM_CLASSES].
    sparse_labels = tf.reshape(labels, [BATCH_SIZE, 1])
    indices = tf.reshape(tf.range(BATCH_SIZE), [BATCH_SIZE, 1])
    concated = tf.concat([indices, sparse_labels], 1)
    dense_labels = tf.sparse_to_dense(concated, [BATCH_SIZE, NUM_CLASSES], 1.0, 0.0)
    # Calculate the average cross entropy loss across the barch.
    cross_entropy_mean = tf.reduce_mean(dense_labels * tf.log(logits + 0.01), name='batch_mean_cross_entropy')
    # The total loss is defined as the cross entropy loss plus all of the weight decay terms(L2 loss)
    return cross_entropy_mean


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
    print 'Loading train dataset ...'
    train_set = pd.read_csv('mini_train.csv')
    train_labels = train_set['label']
    train_set.drop("label", axis=1, inplace=True)

    print 'Standardize each image ...'
    samples_num = train_set.shape[0]
    for i in xrange(samples_num):
        image = train_set.iloc[i].values
        float_image = np.divide(image, 255.0)
        train_set.iloc[i] = float_image - float_image.mean()

    print 'Split train dataset and valid dataset'
    valid_set = train_set.iloc[TRAIN_NUM:]
    valid_labels = train_labels.iloc[TRAIN_NUM:]
    train_set = train_set.head(TRAIN_NUM).copy()
    train_labels = train_labels.head(TRAIN_NUM).copy()

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
            # train_op = train(loss, global_step)
            train_op = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(loss)

            sess.run(tf.global_variables_initializer())

            # print sess.run(logits, feed_dict={images: train_set.iloc[0].astype(np.float32).values.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 1))})

            for step in xrange(MAX_TRAINING_STEP):
                Get batch images.
                for batch_index in range(train_set.shape[0] / BATCH_SIZE):
                    batch_images = train_set.iloc[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE].astype(np.float32).values.reshape((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
                    batch_labels = train_labels.iloc[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE].astype(np.int32).tolist()
                    sess.run([train_op], feed_dict={images: batch_images, labels: batch_labels})

                if step % 10 == 0:
                    correct_counter = 0
                    for valid_batch_num in xrange(VALID_NUM / BATCH_SIZE):
                        batch_images_valid = valid_set.iloc[valid_batch_num * BATCH_SIZE: (valid_batch_num + 1) * BATCH_SIZE].astype(np.float32).values.reshape((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
                        batch_labels_valid = valid_labels.iloc[valid_batch_num * BATCH_SIZE: (valid_batch_num + 1) * BATCH_SIZE].astype(np.int32).tolist()
                        logits_valid_set = sess.run(logits, feed_dict={images: batch_images_valid})
                        predict = logits_valid_set.argmax(axis=1)
                        correct_counter += (np.array(batch_labels_valid) == np.array(predict)).sum()
                    accuracy = correct_counter / float(VALID_NUM / BATCH_SIZE * BATCH_SIZE)
                    print 'Accuracy = {0}% at step {1}'.format(accuracy * 100, step)


