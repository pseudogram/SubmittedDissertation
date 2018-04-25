from abc import ABC, abstractmethod
import numpy as np
import cma

class Agent(ABC):

    @property
    @abstractmethod
    def input_size(self):
        pass

    @property
    @abstractmethod
    def output_size(self):
        pass

    @property
    @abstractmethod
    def num_params(self):
        pass

    @abstractmethod
    def percieve(self, obs):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def set_weights(self, weights):
        pass


class Linear(Agent):
    """Class used to create Linear controller for OpenAI envs"""

    @property
    def input_size(self):
        return self.in_dim

    @property
    def output_size(self):
        return self.out_dim

    @property
    def num_params(self):
        return np.prod(self.weights.shape) + 1 # plus bias

    def __init__(self, in_dim, out_dim):

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights = np.random.rand(self.input_size, self.output_size) * 2 - 1
        self.bias = np.random.rand() * 2 - 1


    def get_weights(self):
        """ Returns the weights of the controller unrolled into a single
        dimension.

        Length of the array = (in_dim + out_dim + nodes)^2
        Length of the array = state_size^2

        :return: numpy.ndarray
        """

        return np.concatenate([self.weights.reshape(-1), self.bias])

    def set_weights(self, weights):
        """ Set the weights of the controller
        :param weights: A single array of length state_size^2
        :return:
        """
        self.bias = weights[-1]
        self.weights = weights[:-1].reshape(self.weights.shape)

    def percieve(self, obs):
        """Pass an observation from the environment and the last state
        If it is the first call, randomly generate a state of of shape
        (state_size, 1)
        """
        a = np.matmul(obs, self.weights) + self.bias
        return np.tanh(a)


class MLP(Agent):
    """Class used to create Linear controller for OpenAI envs"""

    @property
    def input_size(self):
        return self.in_dim

    @property
    def output_size(self):
        return self.out_dim

    @property
    def num_params(self):
        return np.sum([np.prod(w.shape) for w in self.weights]) + len(self.bias)

    def __init__( self, in_dim, out_dim, layers=None):
        """layers = Nodes in each layer"""

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = layers

        self.weights = []
        self.bias = np.random.rand(len(layers) + 1) * 2 - 1
        prev_layer = in_dim
        for layer in layers:
            self.weights.append(np.random.rand(prev_layer, layer)*2-1)
            prev_layer = layer

        self.weights.append(np.random.rand(prev_layer, out_dim) * 2 - 1)

    def get_weights(self):
        """ Returns the weights of the controller unrolled into a single
        dimension.

        Length of the array = (in_dim + out_dim + nodes)^2
        Length of the array = state_size^2

        :return: numpy.ndarray
        """
        rollout = [w.reshape(-1) for w in self.weights] + [self.bias]
        return np.concatenate(rollout)

    def set_weights(self, weights):
        """ Set the weights of the controller
        :param weights: A single array of length state_size^2
        :return:
        """
        start = 0
        fin = 0

        for w in range(len(self.weights)):
            fin += np.prod(self.weights[w].shape)
            self.weights[w] = weights[start:fin].reshape(self.weights[w].shape)
            start = fin
        self.bias = weights[fin:]

    def percieve(self, obs):
        """Pass an observation from the environment and the last state
        If it is the first call, randomly generate a state of of shape
        (state_size, 1)
        """
        prev_node = obs
        for i in range(len(self.weights)):
            prev_node = np.matmul(prev_node, self.weights[i]) + self.bias[i]
            prev_node = np.tanh(prev_node)
        return prev_node

class rnn(Agent):
    """Class used to create RNN controller for OpenAI envs"""

    @property
    def input_size(self):
        return self.obs_space_dim

    @property
    def output_size(self):
        return self.act_space_dim

    @property
    def num_params(self):
        return np.sum([np.prod(w.shape) for w in self.weights])

    def __init__(self, obs_space_dim, act_space_dim, nodes=None ):
        """layers = Nodes in each layer"""

        self.obs_space_dim = obs_space_dim
        self.act_space_dim = act_space_dim
        self.nodes = nodes

        self.weights = []

        # input
        self.weights.append(np.zeros([obs_space_dim, nodes]))
        self.weights.append(np.zeros([nodes + 1, nodes]))
        self.weights.append(np.zeros([nodes + 1, act_space_dim]))
        self.state = np.zeros([1, self.nodes])

    def get_weights( self ):
        """ Returns the weights of the controller unrolled into a single
        dimension.

        Length of the array = (in_dim + out_dim + nodes)^2
        Length of the array = state_size^2

        :return: numpy.ndarray
        """
        rollout = [w.reshape(-1) for w in self.weights]
        return np.concatenate(rollout)

    def set_weights( self, weights ):
        """ Set the weights of the controller
        :param weights: A single array of length state_size^2
        :return:
        """
        start = 0
        fin = 0
        for w in self.weights:
            fin += np.prod(w.shape)
            w = weights[start:fin].reshape(w.shape)
            start = fin

        self.state = np.random.rand(1, self.nodes) * 2 - 1


    def percieve(self, obs):
        """pass a tuple, ob observations and states
        """
        state = np.concatenate([self.state, np.ones([1, 1])], 1)
        n_1 = np.matmul(obs, self.weights[0])
        n_2 = np.matmul(state, self.weights[1])

        self.state = np.tanh(np.add(n_1, n_2))
        next_state = np.concatenate([self.state, np.ones([1, 1])], 1)
        out = np.tanh(np.matmul(next_state, self.weights[2]))

        return out.reshape(-1)

from core.narx import define_scope, lazy_property
import tensorflow as tf

class RNN(Agent):
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
        return int(np.sum([
                    np.prod(self.w1.shape),
                    np.prod(self.w1_state.shape),
                    np.prod(self.b1.shape),
                    np.prod(self.w2.shape),
                    np.prod(self.b2.shape)
        ]))


    def __init__(self, obs_space_dim, act_space_dim, nodes, sess=None):
        """
        Randomly sets the weights of the network to normally distributed
        values with a standard dev of 0.1.


        The size of the state = in_dim + out_dim + nodes

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

        if sess is None:
            self.sess = tf.InteractiveSession()
        else:
            self.sess = sess

        self.x = tf.placeholder(tf.float32, [None, obs_space_dim],
                                'x')
        self.state = tf.placeholder(tf.float32, [None, nodes],
                                    'state')

        self.w1 = tf.get_variable('w1', [obs_space_dim, nodes],
                                    dtype=tf.float32, trainable=False)
        self.w1_state = tf.get_variable('w1_state', [nodes, nodes],
                                    dtype=tf.float32, trainable=False)
        self.b1 = tf.get_variable('b1', [nodes], dtype=tf.float32,
                                  trainable=False)

        self.w2 = tf.get_variable("w2", [nodes, act_space_dim],
                                  dtype=tf.float32, trainable=False)
        self.b2 = tf.get_variable('b2', [nodes], dtype=tf.float32,
                                  trainable=False)

        self.predict

        self.sess.run(tf.global_variables_initializer())

        # Initialises next state to prepare for ....
        self.next_state = np.random.rand(1, self.state_size)*2-1

    @define_scope
    def predict(self):
        """

        :return: x, input placeholder. h_t, previous state placeholder. y,
                 output. h_t1, next state.
        """

        # Hidden state
        state = tf.tanh(tf.matmul(self.x, self.w1) +
                        tf.matmul(self.state, self.w1_state) + self.b1)

        output = tf.tanh(tf.matmul(state, self.w2) + self.b2)

        return output, state

    def get_weights(self):
        """ Returns the weights of the controller unrolled into a single
        dimension.

        Length of the array = (in_dim + out_dim + nodes)^2
        Length of the array = state_size^2

        :return: numpy.ndarray
        """
        a, b, c, d, e = self.w1.eval(), self.w1_state.eval(), self.b1.eval(),\
                        self.w2.eval(), self.b2.eval()

        weights = np.concatenate([a.reshape((-1)), b.reshape((-1)),
                                  c.reshape((-1)), d.reshape((-1)),
                                  e.reshape((-1))], axis=0)
        return weights

    def set_weights(self,weights):
        """ Set the weights of the controller
        :param weights: A single array of length state_size^2
        :return:
        """
        a = np.prod(self.w1.shape)
        b = np.prod(self.w1_state.shape) + a
        c = np.prod(self.b1.shape) + b
        d = np.prod(self.w2.shape) + c
        e = np.prod(self.b2.shape) + d

        w_shape = (e,)
        if weights.shape != w_shape:
            raise RuntimeError('Shape of weights given not correct shape. '
                               'Given {} , expecting ({},)'.format(
                                                                  weights.shape,
                                                                  w_shape))

        x1 = np.reshape(weights[0:a], self.w1.shape)
        x2 = np.reshape(weights[a:b], self.w1_state.shape)
        x3 = np.reshape(weights[b:c], self.b1.shape)
        x4 = np.reshape(weights[c:d], self.w2.shape)
        x5 = np.reshape(weights[d:], self.b2.shape)

        set_w1 = self.w1.assign(x1)
        set_w1_state = self.w1_state.assign(x2)
        set_b1 = self.b1.assign(x3)
        set_w2 = self.w2.assign(x4)
        set_b2 = self.b2.assign(x5)

        self.sess.run([set_w1, set_w1_state, set_b1, set_w2, set_b2])

    def init_state(self):
        self.next_state = np.random.rand(1, self.state_size)*2-1

    def percieve(self, obs):
        """Pass an observation from the environment and the last state
        If it is the first call, randomly generate a state of of shape
        (state_size, 1)
        """
        action, self.next_state = self.sess.run(self.predict, feed_dict={
                                   self.x: obs, self.state: self.next_state})
        return action

if __name__ == "__main__":
    rnn = RNN(3,1,4)
    w = rnn.get_weights()
    print(w.shape)

    rnn.set_weights(w)
    print(rnn.num_params)

    w2 = rnn.get_weights()

    print(np.ma.allequal(w, w2))

    a = rnn.percieve(np.array([1,1,1])[None])
