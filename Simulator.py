import random

import gym
import tensorflow as tf
import numpy as np
from narx import NARX
import math

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Simulator(gym.Env):
    def _step(self, action):
        pass

    def _reset(self):
        pass

    def __init__(self, history, env):
        assert isinstance(env, gym.wrappers.time_limit.TimeLimit)
        self.history = history  # how many prev observations to consider

        # Assuming continuous action space
        action_dim = np.prod(np.array(env.action_space.shape))
        self.x1 = tf.placeholder(tf.float32, shape=[None, action_dim],
                                 name='motor_input')

        # Get total number of dimensions in state
        state_dim = np.prod(np.array(env.observation_space.shape))
        self.x2 = tf.placeholder(tf.float32, shape=[None, state_dim],
                                 name='observation_input')
        self.y = tf.placeholder(tf.float32, shape=[None, state_dim],
                                 name='observation_output')

        # Neural net
        self.sess = tf.InteractiveSession()

        hidden_layers = [dict(units=20, activation=tf.nn.tanh, use_bias=True,
                              reuse=tf.AUTO_REUSE),
                         dict(units=20, activation=tf.nn.tanh, use_bias=True,
                              reuse=tf.AUTO_REUSE)]

        self.narx = NARX(self.x1, self.x2, self.y, history, hidden_layers,
                         learning_rate=0.001,)

        self.trainWriter = tf.summary.FileWriter('tboard/train')
        self.testWriter = tf.summary.FileWriter('tboard/test')

        self.merged = tf.summary.merge_all()

        self.sess.run(tf.global_variables_initializer())
        pass

    def _format_buffer(self, replay_buffer):
        rb = np.array(replay_buffer)

        start = 0
        for i in range(self.history-1):
            if replay_buffer[i][4]:
                start = i+1
                print(i)
                break

        p_observations = []
        p_actions = []
        n_observation = []
        rewards = []
        i = 0
        while i < len(replay_buffer):
            # if done is within prev history, skip position of pointer
            if i > len(replay_buffer) - self.history + 1:
                break
            if replay_buffer[i+self.history-2][4]:
                i += self.history - 1

                continue
            else:
                p_observations.append(
                    np.array([replay_buffer[i + m][0] for m in range(
                        self.history-1,-1,-1)]).reshape(-1))

                p_actions.append(
                    np.array([replay_buffer[i + m][1] for m in range(
                        self.history-1,-1,-1)]).reshape(-1))

                n_observation.append(replay_buffer[i + self.history-1][2])

                rewards.append(replay_buffer[i + self.history-1][3])
            i += 1

        p_observations = np.array(p_observations)
        p_actions = np.array(p_actions)
        n_observation = np.array(n_observation)
        rewards = np.array(rewards)

        return p_observations, p_actions, n_observation, rewards


    def train(self, replay_buffer, batch_size=32, epochs=1, seed=None,
              verbose=0):
        """replay_buffer = history of (prev_observation, action, observation,
        reward, done"""

        prev_obs, prev_act, next_obs, next_reward = \
            self._format_buffer(replay_buffer)

        random.seed(seed)
        # Create batch list
        shuffle = []
        buff_inds = range(prev_obs.shape[0])
        if len(replay_buffer) >= batch_size:
            shuffle += random.sample(buff_inds, len(buff_inds))
        remainder = len(replay_buffer) % batch_size

        print('training data %s' % len(shuffle))
        print('remainder %s' % remainder)
        if remainder:
            shuffle += random.sample(buff_inds, batch_size - remainder)


        x = np.concatenate([prev_act,prev_obs],1)
        print('size of x %s' % str(x.shape))
        y = next_obs

        # train model
        model = self.narx
        iterations = int((len(shuffle) / batch_size)-1)
        print('training data %s' % len(shuffle))
        print('batchsize %s' % batch_size)
        print('iterations %s' % iterations)

        for epoch in range(epochs):
            for batch in range(iterations):
                start = batch * batch_size
                fin = (batch+1) * batch_size
                inds = shuffle[start:fin]
                print(max(shuffle))
                print(inds)
                tboard, _ = self.sess.run([self.merged, self.narx.optimizer],
                                          feed_dict={model.x_0: x[inds, :],
                                                     model.y: y[inds, :]})
                self.trainWriter.add_summary(tboard, batch + iterations * epoch)

        if verbose:
            print('Done finished sleep cycle')

    def angle_normalize( x ):
        return (((x + np.pi) % (2 * np.pi)) - np.pi)

    def _reward(self, observation, u):
        """

        :param observation: last observation output by network
        :param u: action
        :return: reward
        """
        cos_theta, sin_theta, thdot = observation
        c_theta = np.arccos(cos_theta)
        s_theta = np.arcsin(sin_theta)
        th = np.average([c_theta, s_theta])

        cost = self.angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        return -cost

    def reset(self):
        pass

    def normalise(self, action=None, observation=None):
        if action is not None:
            action = action / np.array([2])
        if observation is not None:
            observation = observation / np.array([1, 1, 8])
        return action, observation

    def denormalise(self, action=None, observation=None):
        if action is not None:
            action = action * np.array([2])
        if observation is not None:
            observation = observation * np.array([1, 1, 8])
        return action, observation

    def load_starting_queue(self):
        """Load the first 5 iterations from a real environment and then
        continue"""
        pass

    def step(self, action, iteration):
        if self.steps_passed > 200:
            done = False
        elif self.steps_passed > 201:
            raise Exception('Simulation has ended')
        else:
            done = True

        reward = self._reward(self.prev_ob, action)

        nn = self.narx
        action, observation = self.normalise(action=action,
                                             observation=self.prev_ob)

        _ = self.sess.run(self.narx.enqueue, feed_dict={
            nn.x_1:observation, nn.x_2:action})

        next_observation = self.sess.run(nn.predict_para)
        self.prev_ob = next_observation
        self.steps_passed += 1

        return next_observation, reward, done

    def reset(self):
        """must feed prev ob back into netowrk"""
        high = np.array([np.pi, 1])
        # state = self.np_random.uniform(low=-high, high=high)
        th, thdot = np.random.uniform(-high,high)
        c_th = np.cos(th)
        s_th = np.sin(th)
        start_ob = np.array([c_th, s_th, thdot])[None]
        no_motor = np.array([0])[None]
        self.sess.run(self.narx.enqueue,feed_dict={self.narx.x_1:start_ob,
                                                   self.narx.x_2:no_motor})
        self.prev_ob = start_ob
        self.steps_passed = 0
        return start_ob
