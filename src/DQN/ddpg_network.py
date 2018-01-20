#DDPG based on Morvan

import tensorflow as tf
import numpy as np

#MAX_EPISODES = 200
#MAX_EP_STEPS = 200
#LR_A = 0.001    # learning rate for actor
#LR_C = 0.002    # learning rate for critic
#GAMMA = 0.9     # reward discount

#MEMORY_CAPACITY = 10000
#BATCH_SIZE = 32

#RENDER = False
#ENV_NAME = 'Pendulum-v0'

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, input_width, input_height, net_width, net_height, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, nesterov_momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule,
                 batch_accumulator, rng, input_scale=255.0):
        #self.memory = np.zeros((MEMORY_CAPACITY, self.input_width * 2 + num_actions + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        #self.num_actions, self.self.input_width,  = num_actions, self.input_width,
        self.input_width = input_width
        self.input_height = input_height
        self.network_width = net_width
        self.num_actions = num_actions
        self.discount = discount
        self.batch_size = batch_size
        self.LR_A = learning_rate
        self.LR_C = learning_rate
        self.lr  = learning_rate
        self.num_frames = num_frames
        self.TAU = 0.01      # soft replacement

        self.S = tf.placeholder(tf.float32, [None, self.input_width], 's')
        self.S_ = tf.placeholder(tf.float32, [None, self.input_width], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - self.TAU) * ta + self.TAU * ea), tf.assign(tc, (1 - self.TAU) * tc + self.TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + self.discount * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(self.LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, _ ,s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def train(self, bs, ba, br, bs_,terminals):
        # soft target replacement
        self.sess.run(self.soft_replace)

        # indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        # bt = self.memory[indices, :]
        # bs = bt[:, :self.self.input_width]
        # ba = bt[:, self.self.input_width: self.self.input_width + self.num_actions]
        # br = bt[:, -self.self.input_width - 1: -self.self.input_width]
        # bs_ = bt[:, -self.self.input_width:]

        bs = np.reshape(bs,(self.batch_size,self.input_width))
        bs_ = np.reshape(bs_,(self.batch_size,self.input_width))

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, self.network_width, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.num_actions, activation=tf.nn.tanh, name='a', trainable=trainable)
            return a #tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = self.network_width
            w1_s = tf.get_variable('w1_s', [self.input_width, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.num_actions, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)