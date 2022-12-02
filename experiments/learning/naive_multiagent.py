"""Learning script for multi agent problems but all in individual simulations.

Agents are based on `stable_baselines3`'s implementation of A2C, PPO SAC, TD3, DDPG.

Example
-------
To run the script, type in a terminal:

    $ python singleagent.py --env <env> --algo <alg> --obs <ObservationType> --act <ActionType> --cpu <cpu_num>

Notes
-----
Use:

    $ tensorboard --logdir ./results/save-<env>-<algo>-<obs>-<act>-<time-date>/tb/

To check the tensorboard results at:

    http://localhost:6006/

"""
import collections
import copy
import os
import pprint
import random
import subprocess
from datetime import datetime
from sys import platform

import gym
import hydra
import numpy as np
from gym_pybullet_drones.envs.single_agent_rl.callbacks.StopTrainingRunningAverageRewardThreshold import (
    StopTrainingRunningAverageRewardThreshold,
)
from stable_baselines3.common.evaluation import evaluate_policy
import shared_constants
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import (  # StopTrainingOnMaxEpisodes,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticCnnPolicy as a2cppoCnnPolicy
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.common.policies import (
    MultiInputActorCriticPolicy as a2cppoMultiInputPolicy,
)
from stable_baselines3.common.vec_env import (
    VecCheckNan,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
)
from stable_baselines3.sac import CnnPolicy as sacCnnPolicy
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
from stable_baselines3.td3 import CnnPolicy as td3ddpgCnnPolicy
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy

from gym_pybullet_drones.envs.single_agent_rl import map_name_to_env
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)
from gym_pybullet_drones.envs.single_agent_rl.callbacks.CustomCallback import (
    CustomCallback,
)
from gym_pybullet_drones.envs.single_agent_rl.callbacks.CustomCheckpointCallback import (
    CustomCheckpointCallback,
)
from gym_pybullet_drones.envs.single_agent_rl.callbacks.CustomEvalCallback import (
    CustomEvalCallback,
)

"""float: Reward threshold to halt the script."""

MAX_EPISODES = 10000  # Upperbound: number of episodes

DEFAULT_OUTPUT_FOLDER = "results"


@hydra.main(version_base=None, config_path="config", config_name="naive_multiagent")
def train_loop(cfg: DictConfig = None):
    # cfg = OmegaConf.load(ARGS.config)
    pprint.pprint(cfg)

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)

    EPISODE_REWARD_THRESHOLD = getattr(cfg, "max_reward", 1000)

    #### Save directory ########################################
    filename = (
        os.path.dirname(os.path.abspath(__file__))
        + "/results/save-"
        + cfg.env
        + "-"
        + cfg.algo
        + "-"
        + cfg.obs
        + "-"
        + cfg.act
        + "-"
        + cfg.tag
        + "-"
        + datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    )
    if not os.path.exists(filename):
        os.makedirs(filename + "/")

    #### Print out current git commit hash #####################
    if (platform == "linux" or platform == "darwin") and (
        "GITHUB_ACTIONS" not in os.environ.keys()
    ):
        git_commit = subprocess.check_output(["git", "describe", "--tags"]).strip()
        with open(filename + "/git_commit.txt", "w+") as f:
            f.write(str(git_commit))
    num_agents = cfg.num_agents

    train_envs = []
    eval_envs = []
    models = []
    callbacks = []

    for ix in range(num_agents):
        if cfg.algo in ["sac", "td3", "ddpg"] and cfg.cpu != 1:
            print(
                "[ERROR] The selected algorithm does not support multiple environments"
            )
            exit()

        with open(filename + "/config.yaml", "w") as f:
            OmegaConf.save(cfg, f)
        #### Uncomment to debug slurm scripts ######################
        # exit()

        env_name = cfg.env + "-aviary-v0"
        with open_dict(cfg.env_kwargs):
            cfg.env_kwargs["difficulty"] = ix
        sa_env_kwargs = dict(
            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
            obs=ObservationType[cfg.obs],
            act=ActionType[cfg.act],
            **cfg.env_kwargs,
        )
        n_envs = cfg.cpu

        # train_env = gym.make(env_name, aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
        # obs=ObservationType[cfg.obs], act=ActionType[cfg.act]) # single environment instead of a
        # vectorized one
        envAviary = map_name_to_env(env_name)
        train_env = make_vec_env(
            envAviary, env_kwargs=sa_env_kwargs, n_envs=cfg.cpu, seed=cfg.seed
        )
        # check_env(train_env, warn=True, skip_render_check=True)
        ### Load the saved model if specified #################
        if cfg.exp != "none":
            #### Load the model from file ##############################
            # algo = cfg.exp.split("-")[2]
            # NOTE: if the model is loaded, then the number of cpus must be the same
            algo = cfg.algo

            if os.path.isfile(cfg.exp + "/success_model.zip"):
                path = cfg.exp + "/success_model.zip"
                print("Loading success model")
            elif os.path.isfile(cfg.exp + "/best_model.zip"):
                path = cfg.exp + "/best_model.zip"
                print("Loading the best training model")
            else:
                print("[ERROR]: no model under the specified path", cfg.exp)
                return 1
            if os.path.isfile(cfg.exp + "/vecnormalize_best_model.pkl"):
                vec_norm_path = cfg.exp + "/vecnormalize_best_model.pkl"
                print("Loading vecnormalize")

            # TODO: Fix tb log path for loading models
            if algo == "a2c":
                model = A2C.load(path, tensorboard_log=filename + "/tb_log")
            if algo == "ppo":
                model = PPO.load(path, tensorboard_log=filename + "/tb_log")
            if algo == "sac":
                model = SAC.load(path, tensorboard_log=filename + "/tb_log")
            if algo == "td3":
                model = TD3.load(path, tensorboard_log=filename + "/tb_log")
            if algo == "ddpg":
                model = DDPG.load(path, tensorboard_log=filename + "/tb_log")

            if vec_norm_path is not None:
                train_env = VecNormalize.load(vec_norm_path, train_env)
            else:
                train_env = VecNormalize(
                    train_env, norm_obs=True, norm_reward=True, clip_obs=10.0
                )

            if ObservationType[cfg.obs] != ObservationType.KIN:
                train_env = VecTransposeImage(train_env)
                # train_env = VecCheckNan(train_env, raise_exception=True)
            print("[INFO] Action space:", train_env.action_space)
            print("[INFO] Observation space:", train_env.observation_space)
            model.set_env(train_env)
        else:
            # train_env = VecNormalize(
            #     train_env, norm_obs=True, norm_reward=True, clip_obs=10.0
            # )
            if ObservationType[cfg.obs] != ObservationType.KIN:
                train_env = VecTransposeImage(train_env)
                # train_env = VecCheckNan(train_env, raise_exception=True)
                # train_env = VecFrameStack(train_env, n_stack=4)
                # train_env = VecNormalize(train_env)
            print("[INFO] Action space:", train_env.action_space)
            print("[INFO] Observation space:", train_env.observation_space)
            #### On-policy algorithms ##################################
            # default policy
            p_kwargs = {
                "onpolicy_kwargs": dict(
                    activation_fn=torch.nn.ReLU,
                    net_arch=[512, 512, 256, dict(vf=[256, 128], pi=[256, 128])],
                )  # or None
            }
            # onpolicy_kwargs = cfg.policy_kwargs if cfg.policy_kwargs else onpolicy_kwargs
            if cfg.algo == "a2c":
                if cfg.a2c != None:
                    p_kwargs = hydra.utils.instantiate(cfg.a2c, _convert_="partial")
                if ObservationType[cfg.obs] == ObservationType.KIN:
                    policy = a2cppoMlpPolicy
                elif ObservationType[cfg.obs] == ObservationType.RGB:
                    policy = a2cppoCnnPolicy
                else:
                    policy = a2cppoMultiInputPolicy

                model = A2C(
                    policy,
                    train_env,
                    # policy_kwargs=onpolicy_kwargs,
                    tensorboard_log=filename + "/tb/",
                    verbose=1,
                    seed=cfg.seed,
                    **p_kwargs,
                )

            if cfg.algo == "ppo":
                if cfg.ppo != None:
                    p_kwargs = hydra.utils.instantiate(cfg.ppo, _convert_="partial")
                if ObservationType[cfg.obs] == ObservationType.KIN:
                    policy = a2cppoMlpPolicy
                elif ObservationType[cfg.obs] == ObservationType.RGB:
                    policy = a2cppoCnnPolicy
                else:
                    policy = a2cppoMultiInputPolicy

                model = PPO(
                    policy,
                    train_env,
                    # policy_kwargs=onpolicy_kwargs,
                    tensorboard_log=filename + f"/tb{ix}/",
                    verbose=1,
                    seed=cfg.seed,
                    **p_kwargs,
                )

            #### Off-policy algorithms #################################
            offpolicy_kwargs = dict(
                activation_fn=torch.nn.ReLU, net_arch=[512, 512, 256, 128]
            )  # or None # or dict(net_arch=dict(qf=[256, 128, 64, 32], pi=[256, 128, 64, 32]))
            if cfg.algo == "sac":
                model = (
                    SAC(
                        sacMlpPolicy,
                        train_env,
                        policy_kwargs=offpolicy_kwargs,
                        tensorboard_log=filename + f"/tb{ix}/",
                        seed=cfg.seed,
                        verbose=1,
                    )
                    if ObservationType[cfg.obs] == ObservationType.KIN
                    else SAC(
                        sacCnnPolicy,
                        train_env,
                        policy_kwargs=offpolicy_kwargs,
                        tensorboard_log=filename + "/tb/",
                        seed=cfg.seed,
                        verbose=1,
                    )
                )
            if cfg.algo == "td3":
                model = (
                    TD3(
                        td3ddpgMlpPolicy,
                        train_env,
                        policy_kwargs=offpolicy_kwargs,
                        tensorboard_log=filename + "/tb/",
                        seed=cfg.seed,
                        verbose=1,
                    )
                    if ObservationType[cfg.obs] == ObservationType.KIN
                    else TD3(
                        td3ddpgCnnPolicy,
                        train_env,
                        policy_kwargs=offpolicy_kwargs,
                        tensorboard_log=filename + "/tb/",
                        seed=cfg.seed,
                        verbose=1,
                    )
                )
            if cfg.algo == "ddpg":
                model = (
                    DDPG(
                        td3ddpgMlpPolicy,
                        train_env,
                        policy_kwargs=offpolicy_kwargs,
                        tensorboard_log=filename + "/tb/",
                        seed=cfg.seed,
                        verbose=1,
                    )
                    if ObservationType[cfg.obs] == ObservationType.KIN
                    else DDPG(
                        td3ddpgCnnPolicy,
                        train_env,
                        policy_kwargs=offpolicy_kwargs,
                        tensorboard_log=filename + "/tb/",
                        seed=cfg.seed,
                        verbose=1,
                    )
                )

        #### Create evaluation environment #########################
        if ObservationType[cfg.obs] == ObservationType.KIN:
            eval_env = gym.make(env_name, **sa_env_kwargs)
            # eval_env = VecNormalize(
            #     eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0, training=False
            # )
        else:
            n_envs = 1
            evalAviary = map_name_to_env(env_name)
            eval_env = make_vec_env(
                evalAviary, env_kwargs=sa_env_kwargs, n_envs=1, seed=cfg.seed
            )
            # eval_env = VecNormalize(
            #     eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0, training=False
            # )
            eval_env = VecTransposeImage(eval_env)
            # eval_env = VecFrameStack(eval_env, n_stack=4)
            # eval_env = VecNormalize(eval_env)

        #### Train the model #######################################
        checkpoint_callback = CustomCheckpointCallback(
            save_freq=max(100000 // n_envs, 1),
            save_path=filename + f"/logs{ix}/",
            name_prefix="rl_model",
            verbose=2,
        )
        # callback_on_best = StopTrainingOnRewardThreshold(
        #     reward_threshold=EPISODE_REWARD_THRESHOLD, verbose=1
        # )
        # stop_callback = StopTrainingOnNoModelImprovement(
        #     max_no_improvement_evals=(
        #         cfg.stop_after_no_improvement
        #         if cfg.stop_after_no_improvement is not None
        #         else cfg.n_steps
        #     ),
        #     min_evals=100,
        #     verbose=1,
        # )
        stop_callback = StopTrainingRunningAverageRewardThreshold(
            reward_threshold=EPISODE_REWARD_THRESHOLD, eval_rollback_len=10, verbose=1
        )
        eval_callback = CustomEvalCallback(
            eval_env,
            # callback_on_new_best=callback_on_best,
            callback_after_eval=stop_callback,
            verbose=1,
            best_model_save_path=filename + f"/best_{ix}/",
            log_path=filename + f"/best_{ix}/",
            eval_freq=int(2000 / cfg.cpu),
            deterministic=True,
            render=False,
        )
        custom_callback = CustomCallback()
        training_callback = CallbackList(
            [checkpoint_callback, eval_callback, custom_callback]
        )

        train_envs.append(train_env)
        eval_envs.append(eval_env)
        models.append(model)
        callbacks.append(training_callback)

    training_duration = cfg.training_duration
    training_steps = int(cfg.n_steps / training_duration)
    training_rewards = []
    best_average_reward = -np.inf
    for steps in range(training_steps):
        rewards = []
        for ix in range(num_agents):
            models[ix].learn(
                total_timesteps=int(training_duration),
                callback=callbacks[ix],
                log_interval=int(training_steps / 5),
                reset_num_timesteps=False,
            )
            rewards.append(callbacks[ix].callbacks[1].last_mean_reward)

        # average the model policies
        policies = []
        for ix in range(num_agents):
            policies.append(models[ix].get_parameters())
        avg_policy = average_policies(policies, num_agents)
        for ix in range(num_agents):
            models[ix].set_parameters(avg_policy)

        # Test the new policy on all environments
        mean_rewards = []
        for ix in range(num_agents):
            mean_reward, std_reward = evaluate_policy(
                models[ix],
                eval_envs[ix],
                n_eval_episodes=5,
                render=False,
                deterministic=False,
            )
            mean_rewards.append(mean_reward)
            models[ix].logger.record("ave_eval/reward_{ix}", mean_reward)
        average_reward = np.mean(mean_rewards)
        mean_rewards.append(average_reward)
        training_rewards.append(mean_rewards)
        all_successful = True

        for ix, r in enumerate(mean_rewards):
            if r < EPISODE_REWARD_THRESHOLD:
                all_successful = False
            print(f"Reward for agent {ix} : {r}")

        # save the model with some frequency
        if steps % 100 == 0:
            save_path = filename + "/ave_logs/"
            model_path = save_path + f"rl_model_{steps}_steps.zip"
            # model_path = self._checkpoint_path(extension="zip")
            models[0].save(model_path)
            print(f"Saving model checkpoint to {model_path}")

            # vec_normalize_path = save_path + f"rl_model_vecnormalize_{steps}_steps.pkl"
            # models[0].get_vec_normalize_env().save(vec_normalize_path)
            # print(f"Saving model VecNormalize to {vec_normalize_path}")

        # save the model if we have the new best average reward
        if average_reward > best_average_reward:
            best_average_reward = average_reward
            print(f"New best average reward: {best_average_reward} at step {steps}")

            save_path = filename
            model_path = save_path + f"/best_model.zip"
            # model_path = self._checkpoint_path(extension="zip")
            models[0].save(model_path)

            log_path = save_path + f"/best_model_log.txt"
            with open(log_path, "a",) as file_handler:
                file_handler.write(
                    f"New best average reward: {best_average_reward} at step {steps}\n"
                )
                for ix, r in enumerate(mean_rewards):
                    file_handler.write(f"Reward for agent {ix} : {r}\n")

            print(f"Saving model checkpoint to {model_path}")

            # vec_normalize_path = save_path + f"/vecnormalize_best_model.pkl"
            # models[0].get_vec_normalize_env().save(vec_normalize_path)
            # print(f"Saving model VecNormalize to {vec_normalize_path}")

        if all_successful:
            print("All agents successful, stopping training")
            break

    #### Save the model ########################################
    save_path = filename
    model_path = save_path + f"/success_model.zip"
    # model_path = self._checkpoint_path(extension="zip")
    models[0].save(model_path)
    print(f"Saving model checkpoint to {model_path}")

    # vec_normalize_path = save_path + f"/vecnormalize_success_model.pkl"
    # models[0].get_vec_normalize_env().save(vec_normalize_path)
    # print(f"Saving model VecNormalize to {vec_normalize_path}")

    for ix in range(num_agents):
        #### Print training progression ############################
        print("Log for Agent {}".format(ix))
        with np.load(filename + f"/best_{ix}/evaluations.npz") as data:
            for j in range(data["timesteps"].shape[0]):
                try:
                    print(str(data["timesteps"][j]) + "," + str(data["results"][j][0]))
                except Exception as ex:
                    print("oops")
                    raise ValueError("Could not print training progression") from ex
    return training_rewards


def average_policies(policies, num_agents):
    avg_policy = copy.deepcopy(policies[0])
    for param in avg_policy.keys():
        # if the type is a dict, then once again iterate over the keys and then average the
        # tensors
        if param == "policy.optimizer":
            # don't average the optimizer
            pass
        elif isinstance(avg_policy[param], collections.abc.Mapping):
            avg_policy[param] = average_policies(
                [policies[i][param] for i in range(num_agents)], num_agents
            )
        elif isinstance(avg_policy[param], list):
            avg_policy[param] = average_policies(
                [policies[i][param] for i in range(num_agents)], num_agents
            )
        elif isinstance(avg_policy[param], torch.Tensor):
            avg_policy[param][0] = (
                sum([policies[i][param][0] for i in range(num_agents)]) / num_agents
            )

    return avg_policy


if __name__ == "__main__":
    train_loop()
