# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import argparse
import os
from platform import java_ver
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import pybullet_envs  # noqa
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from gym_pybullet_drones.envs.single_agent_rl import CrossObstaclesAviary
from gym_pybullet_drones.envs.multi_agent_rl import MultiCrossObstaclesAviary
from gym_pybullet_drones.envs.multi_agent_rl.wrappers.MultiRecordEpisodeStatistics import (
    MultiRecordEpisodeStatistics,
)
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)
from omegaconf import OmegaConf
from experiments.learning.utils.model_averager import average_weights


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
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="multicrossobs-aviary-v0",
        help="the id of the environment")
    parser.add_argument("--env-config", type=str, default="config/multi-cross-obstacles-aviary.yaml",
        help="config file for the env")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=25e3,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--shared-policy", type=bool, default=True,
        help="Whether to use a shared policy or not")
    parser.add_argument('--num-models', type=int, default=10,
        help='the number of models saved')
    parser.add_argument('--num-agents', type=int, default=2,
        help='the number of agents used')
    args = parser.parse_args()
    # fmt: on
    args.save_frequency = min(
        args.total_timesteps, int(args.total_timesteps // args.num_models)
    )  # save either 1 or num_model times
    return args


def make_env(env_id, seed, idx, capture_video, run_name, env_config, num_agents):
    # cfg = {"env_kwargs": None}
    # with open(env_config, "r") as f:
    #     cfg = yaml.safe_load(f)
    cfg = OmegaConf.load(env_config)
    env = gym.make(
        env_id,
        act=ActionType.RPM,
        obs=ObservationType.KIN,
        num_drones=num_agents,
        **cfg.env_kwargs,
    )
    env = MultiRecordEpisodeStatistics(env)
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # if capture_video:
    #     if idx == 0:
    #         env = gym.wrappers.RecordVideo(env, f"results/videos/{run_name}")
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, key):
        super().__init__()
        # Check reshape whatever the input is into a box.
        self.obs_dim = env.observation_space[key].shape[0]
        self.act_dim = env.action_space[key].shape[0]

        # self.fc1 = nn.Linear(
        #     np.array(env.observation_space.shape).prod()
        #     + np.prod(env.action_space.shape),
        #     256,
        # )
        self.fc1 = nn.Linear(self.obs_dim + self.act_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env, key):
        super().__init__()
        obs_dim = np.array(env.observation_space[key].shape).prod()
        act_dim = np.prod(env.action_space[key].shape)
        # self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc1 = nn.Linear(obs_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        # self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_mu = nn.Linear(256, act_dim)
        # action rescaling
        # env.action_space is also a dictionary
        self.register_buffer(
            "action_scale",
            torch.FloatTensor(
                (env.action_space[key].high - env.action_space[key].low) / 2.0
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.FloatTensor(
                (env.action_space[key].high + env.action_space[key].low) / 2.0
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


def save(path, agents, training, total_steps, obs):
    """Saves model params and experiment state to checkpoint path.
    """
    path_dir = os.path.dirname(path)
    os.makedirs(path_dir, exist_ok=True)
    state_dict = {}
    for ix in range(len(agents)):
        state_dict[f"agent_{ix}"] = agents[ix].state_dict()
        # "obs_normalizer": self.obs_normalizer.state_dict(),
        # "reward_normalizer": self.reward_normalizer.state_dict(),

    if training:
        exp_state = {
            "total_steps": total_steps,
            "obs": obs,
            # "random_state": get_random_state(),
            # "env_random_state": self.env.get_env_random_state(),
        }
        state_dict.update(exp_state)
    torch.save(state_dict, path)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = make_env(
        args.env_id,
        args.seed,
        0,
        args.capture_video,
        run_name,
        args.env_config,
        args.num_agents,
    )

    # assert isinstance(
    #     envs.action_space, gym.spaces.Box
    # ), "only continuous action space is supported"
    actors = {}
    qf1s = {}
    qf1_targets = {}
    target_actors = {}
    q_optimizers = {}
    actor_optimizers = {}
    best_per_agent = {}

    # we know that the observation space is a dictionary
    for key, space in envs.observation_space.spaces.items():
        actors[key] = Actor(envs, key).to(device)

        qf1s[key] = QNetwork(envs, key).to(device)
        qf1_targets[key] = QNetwork(envs, key).to(device)
        target_actors[key] = Actor(envs, key).to(device)
        target_actors[key].load_state_dict(actors[key].state_dict())
        qf1_targets[key].load_state_dict(qf1s[key].state_dict())

        q_optimizers[key] = optim.Adam(
            list(qf1s[key].parameters()), lr=args.learning_rate
        )
        actor_optimizers[key] = optim.Adam(
            list(actors[key].parameters()), lr=args.learning_rate
        )
        best_per_agent[key] = {"timestep": 0, "reward": -np.inf}

    best_per_agent["average"] = {"timestep": 0, "reward": -np.inf}
    episode_rwds = [0.0 for _ in range(args.num_agents)]

    if args.shared_policy:
        # now that we've updated the policies, average them together
        ave_actor_policy = average_weights(actors)
        for actor in actors.values():
            actor.load_state_dict(ave_actor_policy)
    # envs.observation_space.dtype = np.float32

    rb = [
        ReplayBuffer(
            args.buffer_size,
            envs.observation_space[0],
            envs.action_space[0],
            device,
            handle_timeout_termination=True,
        )
        for _ in range(args.num_agents)
    ]
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    initial_save = True
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = envs.action_space.sample()

        else:
            with torch.no_grad():
                actions = [
                    actors[ix](torch.Tensor(obs[ix]).to(device)) for ix in obs.keys()
                ]

                actions += [
                    torch.normal(
                        actors[ix].action_bias,
                        actors[ix].action_scale * args.exploration_noise,
                    )
                    for ix in obs.keys()
                ]
                actions = {
                    ix: (
                        actions[ix]
                        .cpu()
                        .numpy()
                        .clip(envs.action_space[0].low, envs.action_space[0].high,)
                    )
                    for ix in obs.keys()
                }

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # we need to check if we finished the game
        if dones["__all__"]:
            next_obs = envs.reset()
            ave_episode_rwd = np.mean(episode_rwds)
            if ave_episode_rwd > best_per_agent["average"]["reward"]:
                print(
                    f"New best average reward! Timestep:{global_step}, Reward:{ave_episode_rwd}"
                )
                best_per_agent["average"]["reward"] = ave_episode_rwd
                best_per_agent["average"]["timestep"] = global_step
                save(
                    path=f"results/{run_name}/best_agent_average.pt",
                    agents=actors,
                    training=True,
                    total_steps=global_step,
                    obs=obs,
                )
                with open(f"results/{run_name}/best_agent_average.txt", "w") as f:
                    f.write(f"timestep={global_step}, reward={ave_episode_rwd}\n")
            episode_rwds = [0.0 for _ in range(args.num_agents)]

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # TODO: Get this working, either update the info dict, or write a custom monitoring wrapper
        for ix in range(args.num_agents):
            info = infos[ix]
            if "episode" in info.keys():
                episode_rwds[ix] = info["episode"]["r"]
                print(
                    f"agent={ix}, global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                writer.add_scalar(
                    f"charts/episodic_return_{ix}", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    f"charts/episodic_length_{ix}", info["episode"]["l"], global_step
                )
                # TODO: using this value here isn't the best, should really run an eval episode with
                # no random noise to see if the agent is actually good and not just lucky
                if info["episode"]["r"] > best_per_agent[ix]["reward"]:
                    best_per_agent[ix]["reward"] = info["episode"]["r"]
                    best_per_agent[ix]["timestep"] = global_step
                    save(
                        path=f"results/{run_name}/best_agent_{ix}.pt",
                        agents=actors,
                        training=True,
                        total_steps=global_step,
                        obs=obs,
                    )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in dones.items():
            if d:
                if idx != "__all__":
                    real_next_obs[idx] = infos[idx]["terminal_observation"]
        [
            rb[i].add(
                obs[i], real_next_obs[i], actions[i], rewards[i], dones[i], [infos[i]]
            )
            for i in range(args.num_agents)
        ]

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            datas = [rb[ix].sample(args.batch_size) for ix in range(args.num_agents)]
            with torch.no_grad():
                next_state_actions = [
                    target_actors[ix](datas[ix].next_observations) for ix in obs.keys()
                ]
                qf1_next_target = [
                    qf1_targets[ix](datas[ix].next_observations, next_state_actions[ix])
                    for ix in obs.keys()
                ]
                next_q_value = [
                    datas[ix].rewards.flatten()
                    + (1 - datas[ix].dones.flatten())
                    * args.gamma
                    * (qf1_next_target[ix]).view(-1)
                    for ix in obs.keys()
                ]

            qf1_a_values = [
                qf1s[ix](datas[ix].observations, datas[ix].actions).view(-1)
                for ix in obs.keys()
            ]
            qf1_loss = [
                F.mse_loss(qf1_a_values[ix], next_q_value[ix]) for ix in obs.keys()
            ]

            # optimize the model
            [q_optimizers[ix].zero_grad() for ix in obs.keys()]
            [qf1_loss[ix].backward() for ix in obs.keys()]
            [q_optimizers[ix].step() for ix in obs.keys()]
            actor_loss = {}
            if global_step % args.policy_frequency == 0:
                for ix in range(args.num_agents):
                    qf1, actor, target_actor, actor_optimizer, qf1_target, data = (
                        qf1s[ix],
                        actors[ix],
                        target_actors[ix],
                        actor_optimizers[ix],
                        qf1_targets[ix],
                        datas[ix],
                    )
                    actor_loss[ix] = -qf1(
                        data.observations, actor(data.observations)
                    ).mean()
                    actor_optimizer.zero_grad()
                    actor_loss[ix].backward()
                    actor_optimizer.step()

                    # update the target network
                    for param, target_param in zip(
                        actor.parameters(), target_actor.parameters()
                    ):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        qf1.parameters(), qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )
                if args.shared_policy:
                    # now that we've updated the policies, average them together
                    ave_actor_policy = average_weights(actors)
                    for actor in actors.values():
                        actor.load_state_dict(ave_actor_policy)

            if global_step % 100 == 0:
                for ix in range(args.num_agents):
                    writer.add_scalar(
                        f"losses/qf1_loss_{ix}", qf1_loss[ix].item(), global_step
                    )
                    writer.add_scalar(
                        "losses/actor_loss", actor_loss[ix].item(), global_step
                    )
                    writer.add_scalar(
                        "losses/qf1_values", qf1_a_values[ix].mean().item(), global_step
                    )
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

            if global_step % args.save_frequency == 0 or initial_save:
                initial_save = False
                if not os.path.exists(f"results/{run_name}"):
                    os.makedirs(f"results/{run_name}")
                save(
                    path=f"results/{run_name}/agent.pt",
                    agents=actors,
                    training=True,
                    total_steps=global_step,
                    obs=obs,
                )
                save(
                    path=f"results/{run_name}/{global_step}_agent.pt",
                    agents=actors,
                    training=True,
                    total_steps=global_step,
                    obs=obs,
                )
                if args.track:
                    wandb.save(
                        f"results/{run_name}/agent.pt",
                        base_path=f"results/{run_name}",
                        policy="now",
                    )
    envs.close()
    writer.close()
