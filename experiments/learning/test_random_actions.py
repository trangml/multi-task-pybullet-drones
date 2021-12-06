import gym
import numpy as np
import pandas as pd
import random
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary
from gym_pybullet_drones.envs.single_agent_rl.TuneAviary import TuneAviary
from gym_pybullet_drones.envs.single_agent_rl.NavigateMazeAviary import NavigateMazeAviary
from gym_pybullet_drones.envs.single_agent_rl.NavigateObstaclesAviary import NavigateObstaclesAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)

matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt


# Set random seed
np.random.seed(42)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

class policy_estimator():
    def __init__(self, env):
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_outputs),
            nn.Softmax(dim=-1))

    def predict(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs

def main():
    env = gym.make("obstacles-aviary-v0")
    # env.GUI = True
    env = NavigateObstaclesAviary(gui=True, record=False)
    # num_actions = env.action_space.n
    num_obs = env.observation_space

    num_episodes = 10
    steps_per_episode = 10 * 240
    plotting_freq = 10
    # step size
    alpha = 0.9
    epsilon = 0.0
    # Discount factor
    gamma = 1

    total_rewards = []
    x_axis = np.arange(0, num_episodes)

    # The environment has 4 rows and 12 columns. Thus, 48 states
    # q_table = np.random.random((num_obs, num_actions))
    for episode in range(num_episodes):
        timestep = 0
        observations = env.reset()
        rewards = 0
        done = False

        while not done and timestep < steps_per_episode:
            # Select actions using a fixed stochastic policy
            # 0-right 1-down 2-left 3-up
            # take best action from q_table or random
            if np.random.random() > epsilon:
                action = env.action_space.sample()
            else:
                action = env.action_space.sample()
            env.render()
            next_observations, reward, done, _ = env.step(action)
            # next_action = np.argmax(q_table[next_observations])
            # q_table[observations][action] = q_table[observations][action] + alpha * (
            #     reward
            #     + gamma * q_table[next_observations][next_action]
            #     - q_table[observations][action]
            # )

            observations = next_observations
            rewards += reward
            # print(f"Timestep:  {timestep}  Reward: {reward}")
            timestep = timestep + 1
        print(f"\nTotal reward: {rewards}")
        total_rewards.append(rewards)

    # Now we plot things
    fig, ax = plt.subplots()
    ax.plot(x_axis, total_rewards, label="Total Rewards")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")
    ax.set_title(r"Q-Learning for Takeoff Environment")
    ax.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    main()
