"""Test script for single agent problems.

This scripts runs the best model found by one of the executions of `singleagent.py`

Example
-------
To run the script, type in a terminal:

    $ python test_singleagent.py --exp ./results/save-<env>-<algo>-<obs>-<act>-<time_date>

"""
import os
import time
from datetime import datetime
import argparse
import re
import numpy as np
import gym
import torch
from omegaconf import OmegaConf
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import DDPG
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy as a2cppoCnnPolicy
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
from stable_baselines3.sac import CnnPolicy as sacCnnPolicy
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from stable_baselines3.td3 import CnnPolicy as td3ddpgCnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)

import shared_constants

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description="Single agent reinforcement learning example script using TakeoffAviary"
    )
    parser.add_argument(
        "--exp",
        type=str,
        help="The experiment folder written as ./results/save-<env>-<algo>-<obs>-<act>-<time_date>",
        metavar="",
    )
    parser.add_argument(
        "--landing_zone",
        default="3.5, 3.5, 0.0625",
        type=str,
        help="Landing Zone XYZ location, comma separated (default: 3.5, 3.5, 0.0625)",
        metavar="",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="if included, a video of the simulation is recorded",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="if included, a video of the simulation is recorded",
    )
    parser.add_argument(
        "--early_done",
        action="store_true",
        help="if included, the render will finish on done in the sim",
    )
    parser.add_argument(
        "--override_path",
        type=str,
        help="The specific zip file to run. This bypasses the logic of selecting either the best of the success",
        metavar="",
    )
    ARGS = parser.parse_args()

    if os.path.isfile(ARGS.exp + "/config.yaml"):
        with open(ARGS.exp + "/config.yaml", "r") as f:
            exp = ARGS.exp
            additional_args = OmegaConf.load(f)
            ARGS.__dict__ = {**ARGS.__dict__, **additional_args}
            ARGS.exp = exp
    print(ARGS)
    #### Load the model from file ##############################
    algo = ARGS.algo

    if ARGS.override_path is not None:
        path = ARGS.override_path
    else:
        if ARGS.latest:
            logs = os.listdir(ARGS.exp + "/logs")
            if len(logs) < 1:
                print("[ERROR]: no latest model under the specified path", ARGS.exp)
            else:
                path = ARGS.exp + "/logs/" + logs[-1]
            if os.path.isfile(ARGS.exp + "/success_model.zip"):
                path = ARGS.exp + "/success_model.zip"
        else:
            # if we aren't using the latest model, then we want to use either the success_model or the
            # current best_model
            if os.path.isfile(ARGS.exp + "/success_model.zip"):
                path = ARGS.exp + "/success_model.zip"
            elif os.path.isfile(ARGS.exp + "/best_model.zip"):
                path = ARGS.exp + "/best_model.zip"
            else:
                print("[ERROR]: no model under the specified path", ARGS.exp)

    if algo == "a2c":
        model = A2C.load(path)
    if algo == "ppo":
        model = PPO.load(path)
    if algo == "sac":
        model = SAC.load(path)
    if algo == "td3":
        model = TD3.load(path)
    if algo == "ddpg":
        model = DDPG.load(path)

    #### Parameters to recreate the environment ################
    env_name = ARGS.env + "-aviary-v0"
    OBS = ObservationType[ARGS.obs]
    ACT = ActionType[ARGS.act]

    #### Evaluate the model ####################################
    print(ARGS.landing_zone)
    eval_env = gym.make(
        env_name,
        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
        obs=OBS,
        act=ACT,
        **ARGS.env_kwargs,
    )

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    #### Show, record a video, and log the model's performance #
    test_env = gym.make(
        env_name,
        gui=True,
        record=ARGS.record,
        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
        obs=OBS,
        act=ACT,
        **ARGS.env_kwargs,
    )

    logger = Logger(
        logging_freq_hz=int(test_env.SIM_FREQ / test_env.AGGR_PHY_STEPS),
        num_drones=1,
        num_rewards=len(test_env.reward_dict),
        rewards_names=list(test_env.reward_dict.keys()),
    )
    obs = test_env.reset()
    start = time.time()
    for i in range(
        test_env.EPISODE_LEN_SEC * int(test_env.SIM_FREQ / test_env.AGGR_PHY_STEPS)
    ):  # Up to 6''
        action, _states = model.predict(
            obs, deterministic=True
        )  # OPTIONAL 'deterministic=False'
        obs, reward, done, info = test_env.step(action)
        test_env.render()
        time.sleep(0.05)
        if OBS == ObservationType.KIN:
            logger.log(
                drone=0,
                timestamp=i / test_env.SIM_FREQ,
                state=test_env.env._getDroneStateVector(0),
                # state=np.hstack(
                #     [obs[0:3], np.zeros(4), obs[3:15], np.resize(action, (4))]
                # ),
                control=np.zeros(12),
                reward=list(test_env.reward_dict.values()),
                done=done,
            )
        sync(np.floor(i * test_env.AGGR_PHY_STEPS), start, test_env.TIMESTEP)
        # if done:
        #     obs = test_env.reset()  # OPTIONAL EPISODE HALT
        if ARGS.early_done and done:
            break  # OPTIONAL EPISODE Break
    test_env.close()
    # logger.save_as_csv("sa")  # Optional CSV save
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")
    logger.plot()
    logger.plot_rewards()

    # with np.load(ARGS.exp+'/evaluations.npz') as data:
    #     print(data.files)
    #     print(data['timesteps'])
    #     print(data['results'])
    #     print(data['ep_lengths'])
