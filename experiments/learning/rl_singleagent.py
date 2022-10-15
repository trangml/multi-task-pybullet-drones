"""Learning script for single agent problems.

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
import os
import pprint
import random
import subprocess
from datetime import datetime
from sys import platform

import gym
import hydra
import numpy as np
import shared_constants
import torch
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import (  # StopTrainingOnMaxEpisodes,
    CallbackList, CheckpointCallback, EvalCallback,
    StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import \
    ActorCriticCnnPolicy as a2cppoCnnPolicy
from stable_baselines3.common.policies import \
    ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.common.policies import \
    MultiInputActorCriticPolicy as a2cppoMultiInputPolicy
from stable_baselines3.common.vec_env import (VecCheckNan, VecFrameStack,
                                              VecNormalize, VecTransposeImage)
from stable_baselines3.sac import CnnPolicy as sacCnnPolicy
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
from stable_baselines3.td3 import CnnPolicy as td3ddpgCnnPolicy
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy

from gym_pybullet_drones.envs.single_agent_rl import map_name_to_env
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType, ObservationType)
from gym_pybullet_drones.envs.single_agent_rl.callbacks.CustomCallback import \
    CustomCallback
from gym_pybullet_drones.envs.single_agent_rl.callbacks.CustomCheckpointCallback import \
    CustomCheckpointCallback
from gym_pybullet_drones.envs.single_agent_rl.callbacks.CustomEvalCallback import \
    CustomEvalCallback
from gym_pybullet_drones.envs.single_agent_rl.callbacks.StopTrainingRunningAverageRewardThreshold import \
    StopTrainingRunningAverageRewardThreshold

DEFAULT_OUTPUT_FOLDER = "results"


@hydra.main(version_base=None, config_path="config", config_name="rl_singleagent")
def train_loop(cfg: DictConfig = None):
    # cfg = OmegaConf.load(ARGS.config)
    pprint.pprint(cfg)

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)

    EPISODE_REWARD_THRESHOLD = getattr(cfg, "episode_reward_threshold", 1000)

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

    if cfg.algo in ["sac", "td3", "ddpg"] and cfg.cpu != 1:
        print("[ERROR] The selected algorithm does not support multiple environments")
        exit()

    with open(filename + "/config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    #### Uncomment to debug slurm scripts ######################
    # exit()

    env_name = cfg.env + "-aviary-v0"
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
        envAviary, env_kwargs=sa_env_kwargs, n_envs=cfg.cpu, seed=0
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

        if algo == "a2c":
            model = A2C.load(path, tensorboard_log=filename + "/tb_log")
        if algo == "ppo":
            p_kwargs = hydra.utils.instantiate(cfg.ppo, _convert_="partial")
            p_kwargs["seed"] = cfg.seed
            model = PPO.load(
                path, tensorboard_log=filename + "/tb_log", kwargs=p_kwargs
            )
            model.seed = cfg.seed
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
        train_env = VecNormalize(
            train_env, norm_obs=True, norm_reward=True, clip_obs=10.0
        )
        if ObservationType[cfg.obs] != ObservationType.KIN:
            train_env = VecTransposeImage(train_env)
            # train_env = VecCheckNan(train_env, raise_exception=True)
            # train_env = VecFrameStack(train_env, n_stack=4)
            # train_env = VecNormalize(train_env)
        print("[INFO] Action space:", train_env.action_space)
        print("[INFO] Observation space:", train_env.observation_space)
        #### On-policy algorithms ##################################
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
                tensorboard_log=filename + "/tb/",
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
                    tensorboard_log=filename + "/tb/",
                    verbose=1,
                )
                if ObservationType[cfg.obs] == ObservationType.KIN
                else SAC(
                    sacCnnPolicy,
                    train_env,
                    policy_kwargs=offpolicy_kwargs,
                    tensorboard_log=filename + "/tb/",
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
                    verbose=1,
                )
                if ObservationType[cfg.obs] == ObservationType.KIN
                else TD3(
                    td3ddpgCnnPolicy,
                    train_env,
                    policy_kwargs=offpolicy_kwargs,
                    tensorboard_log=filename + "/tb/",
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
                    verbose=1,
                )
                if ObservationType[cfg.obs] == ObservationType.KIN
                else DDPG(
                    td3ddpgCnnPolicy,
                    train_env,
                    policy_kwargs=offpolicy_kwargs,
                    tensorboard_log=filename + "/tb/",
                    verbose=1,
                )
            )

    #### Create evaluation environment #########################
    if ObservationType[cfg.obs] == ObservationType.KIN:
        eval_env = gym.make(env_name, **sa_env_kwargs)
        eval_env = VecNormalize(
            eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0
        )
    else:
        n_envs = 1
        evalAviary = map_name_to_env(env_name)
        eval_env = make_vec_env(evalAviary, env_kwargs=sa_env_kwargs, n_envs=1, seed=0)
        eval_env = VecNormalize(
            eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0
        )
        eval_env = VecTransposeImage(eval_env)
        # eval_env = VecFrameStack(eval_env, n_stack=4)
        # eval_env = VecNormalize(eval_env)

    #### Train the model #######################################
    checkpoint_callback = CustomCheckpointCallback(
        save_freq=max(100000 // n_envs, 1),
        save_path=filename + "/logs/",
        name_prefix="rl_model",
        verbose=2,
        save_vecnormalize=True,
    )
    # callback_on_best = StopTrainingOnRewardThreshold(
    #     reward_threshold=EPISODE_REWARD_THRESHOLD, verbose=1
    # )
    callback_on_best = StopTrainingRunningAverageRewardThreshold(
        reward_threshold=EPISODE_REWARD_THRESHOLD, eval_rollback_len=5, verbose=1
    )
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=(
            cfg.stop_after_no_improvement
            if cfg.stop_after_no_improvement is not None
            else cfg.n_steps
        ),
        min_evals=100,
        verbose=1,
    )
    eval_callback = CustomEvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        callback_after_eval=stop_callback,
        verbose=1,
        best_model_save_path=filename + "/",
        log_path=filename + "/",
        eval_freq=int(2000 / cfg.cpu),
        deterministic=True,
        render=False,
        save_vecnormalize=True,
    )
    custom_callback = CustomCallback()
    training_callback = CallbackList(
        [checkpoint_callback, eval_callback, custom_callback]
    )

    model.learn(
        total_timesteps=int(cfg.n_steps), callback=training_callback, log_interval=1000,
    )
    reward = eval_callback.last_mean_reward

    #### Save the model ########################################
    model.save(filename + "/success_model.zip")
    print(filename)

    #### Print training progression ############################
    with np.load(filename + "/evaluations.npz") as data:
        for j in range(data["timesteps"].shape[0]):
            try:
                print(str(data["timesteps"][j]) + "," + str(data["results"][j][0]))
            except Exception as ex:
                print("oops")
                raise ValueError("Could not print training progression") from ex
    return reward


if __name__ == "__main__":
    train_loop()
