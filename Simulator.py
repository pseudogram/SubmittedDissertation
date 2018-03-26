import gym
import tensorflow as tf
import numpy as np
from narx import NARX


class Simulator(gym.Env):
    def _step(self, action):
        pass

    def _reset(self):
        pass

    def __init__(self, history, env):
        assert isinstance(env, gym.wrappers.time_limit.TimeLimit)
        self.history = history  # how many prev observations to consider

        self.x1 = tf.placeholder(tf.float32,
                                 dtype=np.array(env.action_space.shape)[None],
                                 name='motor_input')
        self.x2 = tf.placeholder(tf.float32,
                                 dtype=np.array(env.observation_space.shape)[
                                     None],
                                 name='observation_input')
        self.y = tf.placeholder(tf.float32,
                                 dtype=np.array(env.observation_space.shape)[
                                     None],
                                 name='observation_output')

        # Neural net
        self.sess = tf.InteractiveSession()

        self.narx = NARX(self.x1, self.x2, self.y, history, )


        pass

    def train(self, replay_buffer):
        """replay_buffer = history of (prev_observation, action, observation,
        reward, done"""

        # Format data
        pass

    def step(self):
        pass

    def reset(self):
        pass
