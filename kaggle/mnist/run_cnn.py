import tensorflow as tf
import numpy as np
import pandas as pd


IMAGE_SIZE = 28
NUM_CLASSES = 10
INITIAL_LEARNING_RATE = 0.01
LEARNING_RATE_DECAY_FACTOR = 0.5
BATCH_SIZE = 50
# BATCH_SIZE = 5
MOVING_AVERAGE_DECAY = 0.9999
MAX_TRAINING_STEP = 100
TRAIN_NUM = 42000
# TRAIN_NUM = 40000
# VALID_NUM = 2000
# TRAIN_NUM = 80
# VALID_NUM = 20
EVAL_STEP = 10


def inference(images):
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.01))
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[32]))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.01))
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[64]))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        dim = 1
        for d in pool2.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool2, [-1, dim])

        weights = tf.Variable(tf.truncated_normal([dim, 256], stddev=0.01))
        biases = tf.Variable(tf.constant(0.1, shape=[256]))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # batch normalization
    norm3 = tf.layers.batch_normalization(local3)

    # dropout
    with tf.variable_scope('dropout') as scope:
        dropout = tf.nn.dropout(norm3, keep_prob=0.8, name=scope.name)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = tf.Variable(tf.truncated_normal([256, NUM_CLASSES], stddev=0.01))
        biases = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))
        local4 = tf.add(tf.matmul(dropout, weights), biases, name=scope.name)

    return local4


def calculate_loss(logits, labels):
    # Reshape the labels into a dense Tensor of shape [BATCH_SIZE, NUM_CLASSES].
    sparse_labels = tf.reshape(labels, [BATCH_SIZE, 1])
    indices = tf.reshape(tf.range(BATCH_SIZE), [BATCH_SIZE, 1])
    concated = tf.concat([indices, sparse_labels], 1)
    dense_labels = tf.sparse_to_dense(concated, [BATCH_SIZE, NUM_CLASSES], 1.0, 0.0)
    # Calculate the average cross entropy loss across the barch.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=dense_labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # The total loss is defined as the cross entropy loss plus all of the weight decay terms(L2 loss)
    return cross_entropy_mean


def train(total_loss, global_step):
    # Consider decay the learning rate exponentially on the number of steps
    # use tf.train.exponential_decay()
    # global lr
    decay_step = EVAL_STEP * TRAIN_NUM / BATCH_SIZE
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_step,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
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
    return train_op, lr


if __name__ == '__main__':
    print 'Loading train dataset ...'
    # train_set = pd.read_csv('mini_train.csv')
    train_set = pd.read_csv('train.csv')
    train_labels = train_set['label']
    train_set.drop("label", axis=1, inplace=True)

    print 'Standardize train images ...'
    samples_num = train_set.shape[0]
    for i in xrange(samples_num):
        image = train_set.iloc[i].values
        float_image = np.divide(image, 255.0)
        train_set.iloc[i] = float_image

    # print 'Split train dataset and valid dataset'
    # valid_set = train_set.iloc[TRAIN_NUM:]
    # valid_labels = train_labels.iloc[TRAIN_NUM:]
    # train_set = train_set.head(TRAIN_NUM).copy()
    # train_labels = train_labels.head(TRAIN_NUM).copy()

    with tf.Session() as sess:

        global_step = tf.Variable(0, trainable=False)

        images = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1], name='batch_images')
        labels = tf.placeholder(tf.int32, [None, ], name='batch_labels')

        # Build a Graph that computes the logits predictions from the inference model.
        logits = inference(images)

        # Calculate loss.
        loss = calculate_loss(logits, labels)

        # Trains the model with one batch of samples and updates the model parameters.
        train_op, lr = train(loss, global_step)
        # train_op = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(loss)

        sess.run(tf.global_variables_initializer())

        for step in xrange(MAX_TRAINING_STEP):
            # Batch training
            print 'Processing step {}, waiting to train {} batches ...'.format(step, train_set.shape[0] / BATCH_SIZE)

            for batch_index in range(train_set.shape[0] / BATCH_SIZE):
                batch_images = train_set.iloc[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE].astype(np.float32).values.reshape((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
                batch_labels = train_labels.iloc[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE].astype(np.int32).tolist()
                sess.run([train_op], feed_dict={images: batch_images, labels: batch_labels})

            # if step % EVAL_STEP == 0:
            #     images_valid = valid_set.astype(np.float32).values.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))
            #     labels_valid = valid_labels.astype(np.int32).tolist()
            #     logits_valid_set = sess.run(logits, feed_dict={images: images_valid})
            #     predict = logits_valid_set.argmax(axis=1)
            #     correct_counter = (np.array(labels_valid) == np.array(predict)).sum()
            #     accuracy = correct_counter / float(len(predict))
            #     print 'Accuracy = {0}% at step {1}, current learning rate is {2}'.format(accuracy * 100, step, sess.run(lr))

        print 'Loading train dataset ...'
        test_set = pd.read_csv('test.csv')

        print 'Standardize test images ...'
        samples_num = test_set.shape[0]
        for i in xrange(samples_num):
            image = test_set.iloc[i].values
            float_image = np.divide(image, 255.0)
            test_set.iloc[i] = float_image

        print 'Make prediction from test set ...'
        images_test = test_set.astype(np.float32).values.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))
        logits_valid_set = sess.run(logits, feed_dict={images: images_test})
        predict = logits_valid_set.argmax(axis=1)

        print 'Make submission file ...'
        submission = pd.DataFrame({
            'ImageId': range(1, images_test.shape[0] + 1),
            'Label': predict
        })
        submission.to_csv('predict.csv', index=False)
