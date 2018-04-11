"""
Creates a replay buffer of the Pendulum-v0 environment using a deque and
pickles it. Each entry in the queue is of the format:
    (prev_observation, action, observation, reward, done)
"""

from collections import deque
import gym
import pickle


replay_buffer = deque()

env = gym.make('Pendulum-v0')

for i_episode in range(20):
    prev_observation = env.reset()

    for t in range(400):
        #         env.render()
        #         print(prev_observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        replay_buffer.append(
            (prev_observation, action, observation, reward, done))
        prev_observation = observation
        if done:
            #             print("Episode finished after {} timesteps".format(t+1))
            break

env.close()



with open('./tests/data/replay_buffer.pkl', 'wb') as file:
    pickle.dump(replay_buffer, file)

# with open('./tests/data/replay_buffer.pkl', 'rb') as file:
#     new_buff = pickle.load(file)