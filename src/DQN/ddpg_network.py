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
        self.LR_C = learning_rate*2
        self.lr  = learning_rate
        self.num_frames = num_frames
        self.rng = rng
        self.clip_delta = clip_delta
        self.TAU = 0.02      # soft replacement


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
        diff = q - q_target
        quadratic_part = tf.minimum(abs(diff), self.clip_delta)
        linear_part = abs(diff) - quadratic_part
        loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        td_error = loss
        self.td_error = td_error
        #td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(self.LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.a_loss = a_loss
        self.q = q
        self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s, epsilon):
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)
        s = s.reshape(self.input_width)
        action_list = self.sess.run([self.a], {self.S: s[np.newaxis, :]})[0]
        print(action_list)
        if np.count_nonzero(action_list == 0) == 5:
            return self.rng.randint(0, self.num_actions)
        return np.argmax(action_list) 

    def train(self, bs, ba, br, bs_, terminals):
        # soft target replacement
        self.sess.run(self.soft_replace)

        # indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        # bt = self.memory[indices, :]
        # bs = bt[:, :self.input_width]
        # ba = bt[:, self.input_width: selfinput_width + self.num_actions]
        # br = bt[:, -self.input_width - 1: -self.input_width]
        # bs_ = bt[:, -self.input_width:]

        bs = np.reshape(bs,(self.batch_size,self.input_width))
        bs_ = np.reshape(bs_,(self.batch_size,self.input_width))
        temp_a  = np.zeros((self.batch_size, self.num_actions))
        ba = ba.reshape(-1)
        temp_a[np.arange(self.batch_size), ba] = 1
        ba = temp_a
        #print(br)
        #print(ba)
        self.sess.run(self.atrain, {self.S: bs})
        _, td_error,q = self.sess.run([self.ctrain,self.td_error,self.q], {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        #loss = self.sess.run(self.a_loss)
        print(td_error[:5])
        return 0#a_loss

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, self.network_width, activation=tf.nn.relu, name='l1', trainable=trainable)
            bn = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=trainable,scope='bn')
            a = tf.layers.dense(bn, self.num_actions, activation=tf.nn.relu, name='a', trainable=trainable)
            return a #tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = self.network_width
            w1_s = tf.get_variable('w1_s', [self.input_width, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.num_actions, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
