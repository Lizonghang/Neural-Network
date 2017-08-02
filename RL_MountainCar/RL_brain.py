# coding=utf-8
import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.experience_observations = []
        self.experience_actions = []
        self.experience_rewards = []

        self._build_net()
        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter("logs/", self.sess.graph) if output_graph else None
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        # self.saver.restore(self.sess, '/tmp/train/model.ckpt')

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_observation = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_actions = tf.placeholder(tf.int32, [None, ], name="actions")
            self.tf_rewards = tf.placeholder(tf.float32, [None, ], name="rewards")

        """
        # full-connect-layer-1
        self.layer_1 = tf.layers.dense(
            inputs=self.tf_observation,
            units=10,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )

        # full-connect-layer-2
        self.all_act = tf.layers.dense(
            inputs=self.layer_1,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )
        """

        # full-connect-layer-1
        with tf.name_scope('fc1'):
            self.w1 = tf.Variable(tf.random_normal([self.n_features, 10], stddev=0.3), name='w1')
            self.b1 = tf.Variable(tf.constant(0.1), name='b1')
            self.l1 = tf.nn.tanh(tf.matmul(self.tf_observation, self.w1) + self.b1)

            # ———————————— summary ————————————
            # tf.summary.histogram('w1', self.w1)
            # tf.summary.histogram('b1', self.b1)
            # tf.summary.histogram('l1', self.l1)
            # —————————————————————————————————

        # full-connect-layer-2
        with tf.name_scope('fc2'):
            self.w2 = tf.Variable(tf.random_normal([10, self.n_actions], stddev=0.3), name='w2')
            self.b2 = tf.Variable(tf.constant(0.1), name='b2')
            self.all_act = tf.matmul(self.l1, self.w2) + self.b2

            # ———————————— summary ————————————
            # tf.summary.histogram('w2', self.w2)
            # tf.summary.histogram('b2', self.b2)
            # tf.summary.histogram('all_act', self.all_act)
            # —————————————————————————————————

        # softmax-output
        with tf.name_scope('softmax_output'):
            self.all_act_prob = tf.nn.softmax(self.all_act, name='action_probability')

            # ———————————— summary ————————————
            # tf.summary.histogram('all_act_prob', self.all_act_prob)
            # —————————————————————————————————

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)
            # or in this way:
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_actions, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_rewards)

            # ———————————— summary ————————————
            # tf.summary.scalar('loss', loss)
            # —————————————————————————————————

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_observation: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def store_transition(self, s, a, r):
        self.experience_observations.append(s)
        self.experience_actions.append(a)
        self.experience_rewards.append(r)

    def learn(self):
        discounted_experience_rewards_norm = self._discount_and_norm_rewards()

        _, summary = self.sess.run([self.train_op, tf.summary.merge_all()], feed_dict={
             self.tf_observation: np.vstack(self.experience_observations),  # shape=[None, n_obs]
             self.tf_actions: np.array(self.experience_actions),  # shape=[None, ]
             self.tf_rewards: discounted_experience_rewards_norm,  # shape=[None, ]
        })

        self.writer.add_summary(summary)

        self.experience_observations, self.experience_actions, self.experience_rewards = [], [], []

        return discounted_experience_rewards_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_experience_rewards = np.zeros_like(self.experience_rewards)
        running_add = 0
        for t in reversed(range(0, len(self.experience_rewards))):
            running_add = running_add * self.gamma + self.experience_rewards[t]
            discounted_experience_rewards[t] = running_add

        # normalize episode rewards
        discounted_experience_rewards -= np.mean(discounted_experience_rewards)
        discounted_experience_rewards /= np.std(discounted_experience_rewards)
        return discounted_experience_rewards
