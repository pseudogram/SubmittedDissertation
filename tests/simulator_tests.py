import tensorflow as tf
import os
import tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import unittest
from collections import deque
import pickle
import gym
from Simulator import Simulator
import numpy as np

class MyTestCase(unittest.TestCase):
    """
    hello
    """
    def setUp(self):
        self.history = 4

        with open('./data/replay_buffer.pkl', 'rb') as file:
            self.replay_buffer = pickle.load(file)

        env_to_use = 'Pendulum-v0'
        env = gym.make(env_to_use)
        self.history = 4
        self.simulator = Simulator(self.history, env)


    def test_format_buffer(self):
        # Separate into prev_observations
        # previous actions
        # next observations
        # and reward for next observation
        p_obs, p_a, n_o, p_r = self.simulator._format_buffer(
            self.replay_buffer)
        first_obs = np.array([self.replay_buffer[i][0] for i in range(
            self.history-1, -1, -1)])
        print(first_obs)
        print(p_obs.shape)


        self.assert_(np.array_equal(first_obs, p_obs[0]))
        # for i in range(len(self.replay_buffer)):
        #     if self.replay_buffer[i][4]:
        #         print(i)
        #         break

        pos = 200
        second_obs = np.array([self.replay_buffer[i][0] for i in range(
            pos+self.history-1, pos-1, -1)])

        self.assert_(np.array_equal(second_obs, p_obs[197]))

        pos = 400
        second_obs = np.array([self.replay_buffer[i][0] for i in range(
            pos + self.history-1, pos-1, -1)])

        self.assert_(np.array_equal(second_obs, p_obs[394]))

    def test_train(self):


if __name__ == '__main__':
    unittest.main()
