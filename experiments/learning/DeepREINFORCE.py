import os
import time
import argparse
from datetime import datetime
from sys import platform
import subprocess
import pdb
import math
import numpy as np
import pybullet as p
import pickle
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.multi_agent_rl import map_name_to_multi_env
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import (
    BaseMultiagentAviary,
)
from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import (
    LeaderFollowerAviary,
)
from gym_pybullet_drones.envs.multi_agent_rl.MeetupAviary import MeetupAviary
from gym_pybullet_drones.envs.multi_agent_rl.MultiCrossObstaclesAviary import (
    MultiCrossObstaclesAviary,
)
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)
from gym_pybullet_drones.utils.Logger import Logger

import shared_constants


class DeepREINFORCE:
    def __init__(self, args):
        env_name = "multicrossobs-aviary-v0"
        self.env = gym.make(
                env_name,
                num_drones=args.num_drones,
                aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                obs=args.obs,
                act=args.act,
            )

        self.num_obs = self.env.observation_space

        self.num_episodes = 200
        self.batch_size = 16
        self.steps_per_episode = 200
        self.plotting_freq = 10


    def train():
        pass
def reinforce(env, policy_estimator, num_episodes=2000,
              batch_size=10, gamma=0.99):

    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1

    # Define optimizer
    optimizer = optim.Adam(policy_estimator.network.parameters(),
                           lr=0.01)

    action_space = np.arange(env.action_space.n)
    for ep in range(num_episodes):
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        complete = False
        while complete == False:
            # Get actions and convert to numpy array
            action_probs = policy_estimator.predict(s_0).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)
            s_1, r, complete, _ = env.step(action)

            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

            # If complete, batch data
            if complete:
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                # If batch is complete, update network
                if batch_counter == batch_size:
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(batch_rewards)
                    # Actions are used as indices, must be LongTensor
                    action_tensor = torch.LongTensor(batch_actions)

                    # Calculate loss
                    logprob = torch.log(
                        policy_estimator.predict(state_tensor))
                    selected_logprobs = reward_tensor * \
                        logprob[np.arange(len(action_tensor)), action_tensor]
                    loss = -selected_logprobs.mean()

                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1

                # Print running average
                print("\rEp: {} Average of last 10: {:.2f}".format(
                    ep + 1, np.mean(total_rewards[-10:])), end="")

    return total_rewards

if __name__ == "__main__":
    drl = DeepREINFORCE()
    rewards = reinforce(env, pe)
    window = 10
    smoothed_rewards = [np.mean(rewards[i-window:i+1]) if i > window
                        else np.mean(rewards[:i+1]) for i in range(len(rewards))]

    plt.figure(figsize=(12,8))
    plt.plot(rewards)
    plt.plot(smoothed_rewards)
    plt.ylabel('Total Rewards')
    plt.xlabel('Episodes')
    plt.show()
