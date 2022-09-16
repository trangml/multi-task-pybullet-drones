# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from gym_pybullet_drones.envs.single_agent_rl import CrossObstaclesAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)
from omegaconf import OmegaConf


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="cross-obstacles-aviary-v0",
        help="the id of the environment")
    parser.add_argument("--env-config", type=str, default="config/cross-obstacles-aviary.yaml",
        help="config file for the env")
    parser.add_argument("--load-path", type=str, default="/storage/results_v0/cross-obstacles-aviary-v0__ddpg__3__1662045840/agent.pt",
        help="config file for the env")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name, env_config):
    def thunk():
        # cfg = {"env_kwargs": None}
        # with open(env_config, "r") as f:
        #     cfg = yaml.safe_load(f)
        cfg = OmegaConf.load(env_config)
        env = gym.make(
            env_id,
            gui=True,
            act=ActionType.RPM,
            obs=ObservationType.KIN,
            **cfg["env_kwargs"],
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"results/videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.FloatTensor(
                (env.action_space.high - env.action_space.low).reshape((1, 4)) / 2.0
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.FloatTensor(
                (env.action_space.high + env.action_space.low).reshape((1, 4)) / 2.0
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup

    cfg = OmegaConf.load(args.env_config)
    env = gym.make(
        args.env_id,
        gui=True,
        act=ActionType.RPM,
        obs=ObservationType.KIN,
        **cfg["env_kwargs"],
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)

    actor = Actor(env).to(device)

    env.observation_space.dtype = np.float32
    start_time = time.time()

    def load(
        path, agent,
    ):
        """Restores model and experiment given checkpoint path.
        """
        state = torch.load(path)
        # Restore policy.
        agent.load_state_dict(state)
        # obs_normalizer.load_state_dict(state["obs_normalizer"])
        # reward_normalizer.load_state_dict(state["reward_normalizer"])
        # Restore experiment state.

    load(args.load_path, actor)
    # TRY NOT TO MODIFY: start the game
    obs = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.1)
        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            actions = actor(torch.Tensor(obs).to(device))
            actions += torch.normal(actor.action_bias, actor.action_scale)
            actions = (
                actions.cpu().numpy().clip(env.action_space.low, env.action_space.high)
            )

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, done, infos = env.step(actions)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

    env.close()
