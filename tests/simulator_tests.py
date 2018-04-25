import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
from collections import deque
import pickle
import gym
from core.simulator import Simulator
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

        # Check prev Observations
        first_obs = np.array([self.replay_buffer[i][0] for i in range(
            self.history-1, -1, -1)]).reshape(-1)
        self.assert_(np.array_equal(first_obs, p_obs[0]))

        pos = 200
        second_obs = np.array([self.replay_buffer[i][0] for i in range(
            pos+self.history-1, pos-1, -1)]).reshape(-1)

        self.assert_(np.array_equal(second_obs, p_obs[197]))

        pos = 400
        third_obs = np.array([self.replay_buffer[i][0] for i in range(
            pos + self.history-1, pos-1, -1)]).reshape(-1)
        self.assert_(np.array_equal(third_obs, p_obs[394]))

        # Check prev Actions
        first_act = np.array([self.replay_buffer[i][1] for i in range(
            self.history - 1, -1, -1)]).reshape(-1)
        self.assert_(np.array_equal(first_act, p_a[0]))

        pos = 200
        second_act = np.array([self.replay_buffer[i][1] for i in range(
            pos + self.history - 1, pos - 1, -1)]).reshape(-1)

        self.assert_(np.array_equal(second_act, p_a[197]))

        pos = 400
        third_act = np.array([self.replay_buffer[i][1] for i in range(
            pos + self.history - 1, pos - 1, -1)]).reshape(-1)
        self.assert_(np.array_equal(third_act, p_a[394]))

        # Check next observation
        first_no = self.replay_buffer[self.history - 1][2]
        print(first_no.shape, n_o[0].shape)
        self.assert_(np.array_equal(first_no, n_o[0]))

        pos = 200
        second_no = self.replay_buffer[pos+self.history - 1][2]
        self.assert_(np.array_equal(second_no, n_o[197]))

        pos = 400
        third_no = self.replay_buffer[pos+self.history - 1][2]
        self.assert_(np.array_equal(third_no, n_o[394]))

        # Check reward
        first_r = self.replay_buffer[self.history - 1][3]
        print(first_r.shape, n_o[0].shape)
        self.assert_(np.array_equal(first_r, p_r[0]))

        pos = 200
        second_r = self.replay_buffer[pos+self.history - 1][3]
        self.assert_(np.array_equal(second_r, p_r[197]))

        pos = 400
        third_r = self.replay_buffer[pos+self.history - 1][3]
        self.assert_(np.array_equal(third_r, p_r[394]))

    def test_train(self):
        pass


if __name__ == '__main__':
    unittest.main()
