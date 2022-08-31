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
from gym_pybullet_drones.envs.single_agent_rl.NavigateMazeAviary import (
    NavigateMazeAviary,
)
from gym_pybullet_drones.envs.single_agent_rl.NavigateObstaclesAviary import (
    NavigateObstaclesAviary,
)
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)
import shared_constants

matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt


# Set random seed
np.random.seed(42)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


class Policy:
    def __init__(self, env):
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.shape[0]

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_outputs),
            nn.Softmax(dim=-1),
        )

    def predict(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs


def softmax_param(theta, action, state):
    """
    Calculate the softmax parameter.
    """
    return np.exp(theta[state, action] - np.max(theta[state])) / np.sum(
        np.exp(theta[state, :] - np.max(theta[state]))
    )


def get_dist(theta, state):
    """
    Calculate the softmax for all actions
    """
    probs = [softmax_param(theta, action, state) for action in range(len(theta[state]))]
    probs = probs / np.sum(probs)
    return probs


def generate_trajectory(env, theta, max_steps=1000):
    """
    Generate a trajectory of a policy.
    """
    observations = env.reset()
    trajectory = []
    for _ in range(max_steps):
        p_dist = get_dist(theta, observations)
        action = np.random.choice(np.arange(len(p_dist)), p=p_dist)
        # action = np.argmax(policy[observations])
        next_observations, reward, done, _ = env.step(action)
        trajectory.append((observations, action, reward))
        if done:
            break
        observations = next_observations
    return trajectory


def discount_rewards(trajectory, gamma=0.99):
    """
    Calculate the discounted rewards for a trajectory.
    """
    r = np.array([gamma ** i * trajectory[i] for i in range(len(trajectory))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def gradient_of_log_policy(state, action, policy, G):
    """
    Calculate the gradient of the log policy.
    """
    return policy[state][action] * (G - policy[state][action])


def main():
    # r_comp = [ "EnterAreaReward": {"scale": 1,  "area": [[4, 5], [-1, 1]]}, "OrientationReward":{"scale": 0.01}, "IncreaseXReward":{"scale": 0.1}]
    env = gym.make(
        "cross-obstacles-aviary-v0",
        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
        obs=ObservationType["KIN"],
        act=ActionType["RPM"],
        initial_xyzs=[[-0.5, 0, 0.5]],
        reward_components=[],
    )
    # env.GUI = True
    # env = NavigateObstaclesAviary(gui=True, record=False)
    # num_actions = env.action_space.n
    # num_actions = env.action_space.n
    num_obs = env.observation_space

    num_episodes = 200
    batch_size = 16
    steps_per_episode = 200
    plotting_freq = 10
    # step size
    alphas = [0.1]
    for alpha in alphas:
        # Discount factor
        gamma = 0.9

        policy_estimator = Policy(env)
        total_rewards = []
        batch_rewards = []
        batch_actions = []
        batch_states = []
        batch_counter = 1

        optimizer = optim.Adam(policy_estimator.network.parameters(), lr=alpha)
        action_space = np.arange(env.action_space.shape[0])

        x_axis = np.arange(0, num_episodes)

        # The environment has 4 rows and 12 columns. Thus, 48 states
        # theta = np.random.random((48, 4))
        for episode in range(num_episodes):
            s_0 = env.reset()
            states = []
            rewards = []
            actions = []
            done = False
            while done == False:
                # Get actions and convert to numpy array
                action = policy_estimator.predict(s_0).detach().numpy()
                # action = np.random.choice(action_space, p=action_probs)
                s_1, r, done, _ = env.step(action)

                states.append(s_0)
                rewards.append(r)
                actions.append(action)
                s_0 = s_1

                # If done, batch data
                if done:
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
                        # Actions are used as indices, must be
                        # LongTensor
                        action_tensor = torch.LongTensor(batch_actions)

                        # Calculate loss
                        logprob = torch.log(policy_estimator.predict(state_tensor))
                        # selected_logprobs = (
                        #     reward_tensor
                        #     * torch.gather(logprob, 1, action_tensor).squeeze()
                        # )

                        selected_logprobs = (
                            reward_tensor
                            * logprob[np.arange(len(action_tensor)), action_tensor]
                        )
                        loss = -selected_logprobs.mean()

                        # Calculate gradients
                        loss.backward()
                        # Apply gradients
                        optimizer.step()

                        batch_rewards = []
                        batch_actions = []
                        batch_states = []
                        batch_counter = 1

                    avg_rewards = np.mean(total_rewards[-100:])
                    # Print running average
                    print(
                        "\rEp: {} Average of last 100:"
                        + "{:.2f}".format(episode + 1, avg_rewards),
                        end="",
                    )
        fig, ax = plt.subplots()
        ax.plot(x_axis, total_rewards, label="Total Rewards")
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Reward")
        ax.set_title(r"Reinforce for Drone")
        ax.legend(loc="best")
        plt.show()


if __name__ == "__main__":
    main()
