import tensorflow as tf
import numpy as np

class controller:
    """Class used to create RNN controller for OpenAI envs"""

    @property
    def input_size(self):
        return self.obs_space_dim

    @property
    def output_size(self):
        return self.act_space_dim

    @property
    def state_size(self):
        return self.nodes

    @property
    def num_params(self):
        return (self.input_size * self.state_size) + \
               (self.state_size * self.state_size) + \
               (self.state_size * self.output_size)

    def __init__(self, obs_space_dim, act_space_dim, nodes):
        """
        Randomly sets the weights of the network to normally distributed
        values with a standard dev of 0.1.


        The size of the state = obs_space_dim + act_space_dim + nodes

        TODO: May need to update value of dt, not to be between calls


        :param obs_space_dim: Number of input nodes
        :param act_space_dim: Number of output nodes
        :param nodes: Number of unattached nodes
        :param dt: time between calls, single scalar number
        """

        # Input, state, output weights
        self.obs_space_dim = obs_space_dim
        self.act_space_dim = act_space_dim
        self.nodes = nodes

        self.sess = tf.InteractiveSession()

        x = self.make_rnn()

        self.inputs, self.state, self.output, self.next_state, self.W1, \
            self.W2, self.W3 = x

        # Initialize all vars
        self._init_vars()

    def make_rnn(self):
        """

        :return: x, input placeholder. h_t, previous state placeholder. y,
                 output. h_t1, next state.
        """
        # Todo: may need to make the placeholder shape [obs_space_dim, 1]

        # with tf.variable_scope('rnn'):
        # ----------------------------------------------------------------------
        #                               Inputs
        # ----------------------------------------------------------------------
        x = tf.placeholder(tf.float32, (self.input_size, 1),
                           'observation')
        W_hx = tf.get_variable("W_hx",
                             (self.state_size, self.input_size),
                             dtype=tf.float32, trainable=False)

        # ----------------------------------------------------------------------
        #                         previous state
        # ----------------------------------------------------------------------
        h_t = tf.placeholder(tf.float32, [self.state_size, 1],
                             'previous_state')
        W_hh = tf.get_variable("W_hh", [self.state_size, self.state_size],
                               dtype=tf.float32, trainable=False)

        # ----------------------------------------------------------------------
        #                               State
        # ----------------------------------------------------------------------
        # Hidden bias
        b_h = tf.zeros([self.nodes, 1], dtype=tf.float32)
        # Hidden state
        h_t1 = tf.tanh(tf.matmul(W_hx, x,name='inputs') + tf.matmul(W_hh,
                                                                   h_t,
                                                      name='hidden') + b_h)

        # ----------------------------------------------------------------------
        #                              Output
        # ----------------------------------------------------------------------
        W_yh = tf.get_variable("W_yh", (self.output_size, self.state_size),
                               dtype=tf.float32, trainable=False)
        b_y = tf.zeros([self.output_size, 1], dtype=tf.float32)
        y = tf.tanh(tf.matmul(W_yh, h_t1, name='output') + b_y) * 2

        # input, prev state, output, next state, input_weights, hidden_weights,
        # out_weights
        return x, h_t, y, h_t1, W_hx, W_hh, W_yh

    def _init_vars(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def get_weights(self):
        """ Returns the weights of the controller unrolled into a single
        dimension.

        Length of the array = (obs_space_dim + act_space_dim + nodes)^2
        Length of the array = state_size^2

        :return: numpy.ndarray
        """
        weights = np.concatenate((self.W1.eval().reshape((-1)),
                           self.W2.eval().reshape((-1)),
                           self.W3.eval().reshape((-1))), axis=0)
        return weights

    def set_weights(self,weights):
        """ Set the weights of the controller
        :param weights: A single array of length state_size^2
        :return:
        """
        one = np.prod(self.W1.shape)
        two = np.prod(self.W2.shape)
        three = np.prod(self.W3.shape)
        w_shape = (one+two+three,)
        if weights.shape != w_shape:
            raise RuntimeError('Shape of weights given not correct shape. '
                               'Given {} , expecting ({},)'.format(
                                                                  weights.shape,
                                                                  w_shape))

        x1 = np.reshape(weights[0:one], self.W1.shape)
        x2 = np.reshape(weights[one:one+two], self.W2.shape)
        x3 = np.reshape(weights[one+two:], self.W3.shape)
        set_w1 = self.W1.assign(x1)
        set_w2 = self.W2.assign(x2)
        set_w3 = self.W3.assign(x3)

        self.sess.run([set_w1, set_w2, set_w3])

    def percieve(self, obs, _state):
        """Pass an observation from the environment and the last state
        If it is the first call, randomly generate a state of of shape
        (state_size, 1)
        """
        action, _next_state = self.sess.run([self.output, self.next_state],
                                            feed_dict={self.inputs: obs,
                                                       self.state: _state})
        return action, _next_state
