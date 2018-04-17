import tensorflow as tf
import numpy as np
from pprint import pprint

from tensorflow.contrib.rnn import RNNCell

# If using seed to replicate results.
# np.random.seed(22)

# dt = 0.05  # time between each time step in the simulation.
# num_inputs = 3  # Number of Values output by the simulation
# num_outputs = 1  # OpenAI env takes a single controller input, thus the controller network must output only one.
# num_nodes = num_inputs + 1  # Number of nodes in the network (must be >=nump_inputs)
# num_epochs = 10
# initial_state = np.random.randn(num_nodes,
#                                 1)  # tf.zeros([num_nodes,1],dtype=tf.float32)


# class D_RNN():
#     def __init__(self):
#         self.create_network()

# TODO: set_weights method may be totally redundant as golbal variable initializer sets all weights
def set_weights(shape, std_dev=0.1):
    """
    Returns normally distributed tensor

    Args:
        shape - dimensions of the nodes
        std_dev - standard_deviation
    """
    return tf.random_normal(shape, mean=0.0, stddev=std_dev, dtype=tf.float32,
                            seed=None, name=None)


class Basic_rnn:
    """Class used to create RNN controller for OpenAI envs"""

    @property
    def input_size(self):
        return self.obs_space_dim

    @property
    def output_size(self):
        return self.act_space_dim

    @property
    def state_size(self):
        return self.act_space_dim + self.obs_space_dim + self.nodes

    @property
    def num_params(self):
        return self.state_size ** 2

    def __init__(self, obs_space_dim, act_space_dim, nodes, dt,
                 init_weights=True):
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
        if init_weights:
            self.weights = tf.Variable(
            set_weights([obs_space_dim + act_space_dim + nodes,
                         obs_space_dim + act_space_dim + nodes]),
                        dtype=tf.float32, name="node_weights",
                        trainable=False)
        else:
            self.weights = tf.get_variable("node_weights",
                                           (self.state_size, self.state_size),
                                           dtype=tf.float32,
                                           trainable=False)

        self.bias = tf.zeros([obs_space_dim + act_space_dim + nodes, 1],
                            dtype=tf.float32)
        self.dt = dt
        self.sess = tf.InteractiveSession()
        x = self.make_rnn()
        self.inputs, self.state, self.output, self.next_state = x

        # Initialize all vars
        self._init_vars()

    def make_rnn(self):
        """
        Inputs added to the first obs_space_dim inputs


        :param inputs:
        :param state:
        :return:
        """
        with tf.name_scope('inputs'):
            inputs = tf.placeholder(tf.float32, [self.obs_space_dim, 1],
                                    'observation')
            state = tf.placeholder(tf.float32, [self.obs_space_dim +
                                                self.act_space_dim +
                                                self.nodes, 1],
                                   'observation')
        _input_buffer = tf.zeros([self.state_size - self.input_size, 1],
                                 dtype=tf.float32)
        with tf.name_scope('rnn'):
            Wx_b = tf.matmul(self.weights, state) + self.bias
            # print('states', state.shape)
            # print('weights', self.weights.shape)
            # print('bias', self.bias.shape)
            # print(Wx_b.shape)

            I = tf.concat([inputs, _input_buffer], 0)
            # print(I.shape)

            Wx_b_I = Wx_b+ I
            # print('added',Wx_b_I.shape)

            next_state = state + self.dt * (-state + tf.tanh(Wx_b_I))
            # print(next_state.shape)

            output = next_state[-self.act_space_dim:]
            # print(output.shape)

            return inputs, state, output, next_state

    def _init_vars( self ):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def get_weights(self):
        """ Returns the weights of the controller unrolled into a single
        dimension.

        Length of the array = (obs_space_dim + act_space_dim + nodes)^2
        Length of the array = state_size^2

        :return: numpy.ndarray
        """
        return self.weights.eval()

    def set_weights(self,weights):
        """ Set the weights of the controller
        :param weights: A single array of length state_size^2
        :return:
        """
        if weights.shape != (self.state_size,self.state_size) and \
            weights.shape != (self.state_size**2,):
            raise RuntimeError('Shape of weights given not correct shape. '
                               'Given {} , expecting {} or {}'.format(
                weights.shape, (self.state_size,self.state_size),
                (self.state_size**2)
            ))

        x = np.reshape(weights,(self.state_size,self.state_size))
        set_w = self.weights.assign(x)
        self.sess.run(set_w)

    def percieve(self, obs, _state):
        """Pass an observation from the environment and the last state
        If it is the first call, randomly generate a state of of shape
        (state_size, 1)
        """
        action, _next_state = self.sess.run([self.output, self.next_state],
                                            feed_dict={self.inputs: obs,
                                                       self.state: _state})
        return action, _next_state


class FullyConnectedRNN:
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


if __name__ == '__main__':
    # --------------------------------------------------------------------------
    #                       All code below used for testing
    # --------------------------------------------------------------------------

    print(np.random.rand())

    c = Basic_rnn(3, 1, 1, 0.05, False)
    x = c.get_weights()
    # print(x)
    # print(type(x))
    # print(np.reshape(x,-1))

    print(x)
    c.set_weights(np.ones((5,5),dtype=np.float32))
    print(c.get_weights())

    # # inputs, state, outputs, next_state = c.make_rnn()
    # num_epochs = 100
    #
    # out_list = list()
    # _in = np.zeros([num_inputs, 1], dtype=np.float32)
    # _next_state = np.ones([c.state_size, 1], dtype=np.float32)
    # for epoch in range(num_epochs):
    #     _, _next_state = c.percieve(_in,_next_state)
    #     out_list.append(_)
    # pprint(out_list)