from abc import ABC, abstractmethod
import numpy as np

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
        return self.obs_space_dim

    @property
    def output_size(self):
        return self.act_space_dim

    @property
    def num_params(self):
        return np.prod(self.weights.shape)

    def __init__(self, obs_space_dim, act_space_dim):

        self.obs_space_dim = obs_space_dim
        self.act_space_dim = act_space_dim
        self.weights = np.random.rand(self.input_size, self.output_size)*2-1


    def get_weights(self):
        """ Returns the weights of the controller unrolled into a single
        dimension.

        Length of the array = (obs_space_dim + act_space_dim + nodes)^2
        Length of the array = state_size^2

        :return: numpy.ndarray
        """

        return self.weights.reshape(-1)

    def set_weights(self, weights):
        """ Set the weights of the controller
        :param weights: A single array of length state_size^2
        :return:
        """
        self.weights = weights.reshape(self.weights.shape)

    def percieve(self, obs):
        """Pass an observation from the environment and the last state
        If it is the first call, randomly generate a state of of shape
        (state_size, 1)
        """
        action = np.matmul(obs, self.weights)
        return np.tanh(action)


class MLP(Agent):
    """Class used to create Linear controller for OpenAI envs"""

    @property
    def input_size(self):
        return self.obs_space_dim

    @property
    def output_size(self):
        return self.act_space_dim

    @property
    def num_params(self):
        return np.sum([np.prod(w.shape) for w in self.weights])

    def __init__(self, obs_space_dim, act_space_dim, layers=None):
        """layers = Nodes in each layer"""

        self.obs_space_dim = obs_space_dim
        self.act_space_dim = act_space_dim
        self.layers = layers

        self.weights = []
        prev_layer = self.input_size
        for layer in layers:
            self.weights.append(np.random.rand(prev_layer, layer)*2-1)
            prev_layer = layer

        self.weights.append(np.random.rand(prev_layer, self.output_size)*2-1)


    def get_weights(self):
        """ Returns the weights of the controller unrolled into a single
        dimension.

        Length of the array = (obs_space_dim + act_space_dim + nodes)^2
        Length of the array = state_size^2

        :return: numpy.ndarray
        """
        rollout = [w.reshape(-1) for w in self.weights]
        return np.concatenate(rollout)

    def set_weights(self, weights):
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

    def percieve(self, obs):
        """Pass an observation from the environment and the last state
        If it is the first call, randomly generate a state of of shape
        (state_size, 1)
        """
        prev_node = obs
        for w in self.weights:
            prev_node = np.matmul(prev_node, w)
            prev_node = np.tanh(prev_node)
        return prev_node

if __name__ == "__main__":
    a = MLP(3,1,[4])
    b = a.get_weights()
    a.set_weights(b)
    c = a.get_weights()
    print(b)
    print(c)
    print(np.equal(b,c))