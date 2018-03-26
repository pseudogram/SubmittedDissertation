from collections import deque
import tensorflow as tf
import numpy as np
import gym
import json,sys,os
from os import path

################################################################################
## Algorithm

# A simple recurrent neural network to solve the Pendulum V0 problem

################################################################################

## Setup

env_to_use = 'Pendulum-v0'

# HYPERPARAMETERS

# OpenAI
num_episodes = 15000  # number of episodes
max_steps_ep = 20000  # default max number of steps per episode (unless env
# has a lower hardcoded limit)

# Agent Parameters


# Simulator Parameters
prev_steps = 3
batch_size = 49

# Game Parameters
env = gym.make(env_to_use)
state_dim = np.prod(np.array(env.observation_space.shape)) 	# Get total number of dimensions in state
action_dim = np.prod(np.array(env.action_space.shape))		# Assuming continuous action space


# info to be recorded
info = {}
info['env_id'] = env.spec.id
info['params'] = dict(
    num_episodes=num_episodes,
    max_steps_ep=max_steps_ep
)



# prepare monitorings
outdir = 'tmp/narx-agent-results'
best_score = -(sys.maxsize-1)
# env = wrappers.Monitor(env, outdir, force=True)
def writefile(fname, s):
    with open(path.join(outdir, fname), 'w') as fh: fh.write(s)










































# hyperparameters
# gamma = 0.99				# reward discount factor
# h1_actor = 8				# hidden layer 1 size for the actor
# h2_actor = 8				# hidden layer 2 size for the actor
# h3_actor = 8				# hidden layer 3 size for the actor
# h1_critic = 8				# hidden layer 1 size for the critic
# h2_critic = 8				# hidden layer 2 size for the critic
# h3_critic = 8				# hidden layer 3 size for the critic
# lr_actor = 1e-3				# learning rate for the actor
# lr_critic = 1e-3			# learning rate for the critic
# lr_decay = 1				# learning rate decay (per episode)
# l2_reg_actor = 1e-6			# L2 regularization factor for the actor
# l2_reg_critic = 1e-6		# L2 regularization factor for the critic
# dropout_actor = 0			# dropout rate for actor (0 = no dropout)
# dropout_critic = 0			# dropout rate for critic (0 = no dropout)
# num_episodes = 15000		# number of episodes
# max_steps_ep = 10000	# default max number of steps per episode (unless env has a lower hardcoded limit)
# tau = 1e-2				# soft target update rate
# train_every = 1			# number of steps to run the policy (and collect experience) before updating network weights
# replay_memory_capacity = int(1e5)	# capacity of experience replay memory
# minibatch_size = 1024	# size of minibatch from experience replay memory for updates
# initial_noise_scale = 0.1	# scale of the exploration noise process (1.0 is the range of each action dimension)
# noise_decay = 0.99		# decay rate (per episode) of the scale of the exploration noise process
# exploration_mu = 0.0	# mu parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt + sigma*dWt
# exploration_theta = 0.15 # theta parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt + sigma*dWt
# exploration_sigma = 0.2	# sigma parameter for the exploration noise process: dXt = theta*(mu-Xt	)*dt + sigma*dWt
#
# # game parameters
# env = gym.make(env_to_use)
# state_dim = np.prod(np.array(env.observation_space.shape)) 	# Get total number of dimensions in state
# action_dim = np.prod(np.array(env.action_space.shape))		# Assuming continuous action space
#
# # set seeds to 0
# env.seed(0)
# np.random.seed(0)
#
# # prepare monitorings
# outdir = '/tmp/ddpg-agent-results'
# env = wrappers.Monitor(env, outdir, force=True)
# def writefile(fname, s):
#     with open(path.join(outdir, fname), 'w') as fh: fh.write(s)
# info = {}
# info['env_id'] = env.spec.id
# info['params'] = dict(
# 	gamma = gamma,
# 	h1_actor = h1_actor,
# 	h2_actor = h2_actor,
# 	h3_actor = h3_actor,
# 	h1_critic = h1_critic,
# 	h2_critic = h2_critic,
# 	h3_critic = h3_critic,
# 	lr_actor = lr_actor,
# 	lr_critic = lr_critic,
# 	lr_decay = lr_decay,
# 	l2_reg_actor = l2_reg_actor,
# 	l2_reg_critic = l2_reg_critic,
# 	dropout_actor = dropout_actor,
# 	dropout_critic = dropout_critic,
# 	num_episodes = num_episodes,
# 	max_steps_ep = max_steps_ep,
# 	tau = tau,
# 	train_every = train_every,
# 	replay_memory_capacity = replay_memory_capacity,
# 	minibatch_size = minibatch_size,
# 	initial_noise_scale = initial_noise_scale,
# 	noise_decay = noise_decay,
# 	exploration_mu = exploration_mu,
# 	exploration_theta = exploration_theta,
# 	exploration_sigma = exploration_sigma
# )
#
# np.set_printoptions(threshold=np.nan)
#
# replay_memory = deque(maxlen=replay_memory_capacity)			# used for O(1) popleft() operation
#
# def add_to_memory(experience):
# 	replay_memory.append(experience)
#
# def sample_from_memory(minibatch_size):
# 	return random.sample(replay_memory, minibatch_size)


# Finalize and upload results, saving info with the best score.
writefile('info_{}.json'.format(best_score), json.dumps(info))
env.close()
# gym.upload(outdir)