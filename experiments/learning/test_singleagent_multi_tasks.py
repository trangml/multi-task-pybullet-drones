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
import numpy as np
import gym
import torch
import random
from omegaconf import OmegaConf
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (
    VecTransposeImage,
    VecFrameStack,
    VecNormalize,
)
from gym_pybullet_drones.envs.single_agent_rl.common.NotVecNormalize import (
    NotVecNormalize,
)
from stable_baselines3.common.env_util import make_vec_env
from gym_pybullet_drones.envs.single_agent_rl import map_name_to_env

from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)
from gym_pybullet_drones.utils.utils import sync, str2bool

import shared_constants

DEFAULT_GUI = True
DEFAULT_PLOT = True
DEFAULT_OUTPUT_FOLDER = "results"


def run(
    exp,
    gui=DEFAULT_GUI,
    plot=DEFAULT_PLOT,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    **ARGS,
):
    """
    Runs testing for a single agent RL problem.

    Parameters
    ----------
    exp : _type_
        _description_
    gui : _type_, optional
        _description_, by default DEFAULT_GUI
    plot : _type_, optional
        _description_, by default DEFAULT_PLOT
    output_folder : _type_, optional
        _description_, by default DEFAULT_OUTPUT_FOLDER
    """
    #### Load the model from file ##############################

    if os.path.isfile(exp + "/config.yaml"):
        with open(exp + "/config.yaml", "r") as f:
            additional_args = OmegaConf.load(f)
            ARGS = OmegaConf.create({**ARGS, **additional_args})
            ARGS.exp = exp
    else:
        raise ValueError("No config.yaml found in {}".format(exp))
    print(ARGS)

    # TRY NOT TO MODIFY: seeding
    print(ARGS.seed)
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)

    #### Load the model from file ##############################
    algo = ARGS.algo
    vec_wrapped = False
    vec_norm_pth = exp + "/vec_normalize_best_model.pkl"
    if ARGS.override_path is not None:
        path = ARGS.override_path
        idx = ARGS.override_path.rindex("_", 0, len(ARGS.override_path) - 10)
        vec_norm_pth = (
            ARGS.override_path[:idx]
            + "_vecnormalize"
            + ARGS.override_path[idx:-3]
            + "pkl"
        )
    else:
        if ARGS.latest:
            logs = os.listdir(ARGS.exp + "/logs")
            zips = [a for a in logs if a.endswith("zip")]
            pkls = [a for a in logs if a.endswith("pkl")]
            if len(zips) < 1:
                print("[ERROR]: no latest model under the specified path", ARGS.exp)
            else:
                path = ARGS.exp + "/logs/" + logs[-1]
            if len(pkls) >= 1:
                vec_wrapped = True
                vec_norm_pth = ARGS.exp + "/logs/" + pkls[-1]
                print("vecnorm found")
        else:
            # if we aren't using the latest model, then we want to use either the success_model or the
            # current best_model
            if os.path.isfile(exp + "/success_model.zip") and ARGS.best == False:
                path = exp + "/success_model.zip"
            elif os.path.isfile(exp + "/best_model.zip"):
                path = exp + "/best_model.zip"
            else:
                print("[ERROR]: no model under the specified path", exp)

            if (
                os.path.isfile(exp + "/vecnormalize_success_model.pkl")
                and ARGS.best is False
            ):
                vec_wrapped = True
                vec_norm_pth = exp + "/vecnormalize_success_model.pkl"
                print("vecnorm found")
            elif os.path.isfile(exp + "/vecnormalize_best_model.pkl"):
                vec_wrapped = True
                vec_norm_pth = exp + "/vecnormalize_best_model.pkl"
                print("vecnorm found")
            else:
                print("no vecnorm found")

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
    diff = ARGS.env_kwargs.difficulty
    if ARGS.all_difficulties:
        if ARGS.env == "room":
            diff_range = range(0, 3)
        elif ARGS.env == "cross-obstacles":
            if diff > 10 or (ARGS.alternate_difficulty and diff < 10):
                diff_range = [0, *range(11, 17)]
            else:
                diff_range = range(0, 7)
        else:
            raise ValueError("Test not implemented for this environment")
    else:
        if ARGS.env == "room":
            diff_range = range(0, 3)
        elif ARGS.env == "cross_obstacles":
            if diff > 10 or (ARGS.alternate_difficulty and diff > 10):
                diff_range = [0, *range(11, diff + 1)]
            else:
                diff_range = range(0, diff + 1)
        else:
            raise ValueError("Test not implemented for this environment")
    #### Evaluate the model ####################################
    cum_results = []
    cum_stds = []
    test_results = []
    for difficulty in diff_range:
        ARGS.env_kwargs.difficulty = difficulty
        env_kwargs = dict(
            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
            obs=OBS,
            act=ACT,
            **ARGS.env_kwargs,
        )
        envAviary = map_name_to_env(env_name)
        if vec_wrapped:
            eval_env = make_vec_env(envAviary, env_kwargs=env_kwargs, n_envs=1)
            eval_env = VecNormalize.load(vec_norm_pth, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
            if ObservationType[ARGS.obs] != ObservationType.KIN:
                eval_env = VecTransposeImage(eval_env)
        else:
            eval_env = gym.make(env_name, **env_kwargs,)
        eval_env.reset()
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=5, deterministic=False
        )
        cum_results.append(mean_reward)
        cum_stds.append(std_reward)

        print(
            "Evaluated model on {}-difficulty {}-env with mean reward {:.2f} +/- {:.2f}".format(
                difficulty, ARGS.env, mean_reward, std_reward
            )
        )
        #### Show, record a video, and log the model's performance #
        exp_start = exp.index("save")
        env_kwargs = dict(
            gui=gui,
            record=ARGS.record,
            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
            obs=OBS,
            act=ACT,
            tag=exp[exp_start:],
            **ARGS.env_kwargs,
        )
        if vec_wrapped:
            test_env = make_vec_env(envAviary, env_kwargs=env_kwargs, n_envs=1)
            test_env = VecNormalize.load(vec_norm_pth, test_env)
            test_env.training = False
            test_env.norm_reward = False
            if ObservationType[ARGS.obs] != ObservationType.KIN:
                test_env = VecTransposeImage(test_env)
        else:
            test_env = gym.make(env_name, **env_kwargs,)

        total_reward = 0
        if vec_wrapped:
            logger = Logger(
                logging_freq_hz=int(
                    test_env.get_attr("SIM_FREQ")[0]
                    / test_env.get_attr("AGGR_PHY_STEPS")[0]
                ),
                num_drones=1,
                num_rewards=len(test_env.get_attr("reward_dict")[0]),
                rewards_names=list(test_env.get_attr("reward_dict")[0].keys()),
                # done_names=list(test_env.term_dict.keys()),
                output_folder=test_env.get_attr("CURR_OUTPUT_FOLDER")[0],
            )

            obs = test_env.reset()
            start = time.time()
            steps = 0
            for i in range(
                test_env.get_attr("EPISODE_LEN_SEC")[0]
                * int(
                    test_env.get_attr("SIM_FREQ")[0]
                    / test_env.get_attr("AGGR_PHY_STEPS")[0]
                )
            ):  # Up to 6''
                action, _states = model.predict(
                    obs, deterministic=True
                )  # OPTIONAL 'deterministic=False'
                obs, reward, done, info = test_env.step(action)
                total_reward += reward
                test_env.render()
                time.sleep(0.05)
                if OBS == ObservationType.KIN:
                    logger.log(
                        drone=0,
                        timestamp=i / test_env.get_attr("SIM_FREQ")[0],
                        state=test_env.get_attr("env")[0]._getDroneStateVector(0),
                        # state=np.hstack(
                        #     [obs[0:3], np.zeros(4), obs[3:15], np.resize(action, (4))]
                        # ),
                        control=np.zeros(12),
                        reward=list(test_env.get_attr("reward_dict")[0].values()),
                        done=done,
                    )
                else:
                    logger.log(
                        drone=0,
                        timestamp=i / test_env.get_attr("SIM_FREQ")[0],
                        state=test_env.get_attr("env")[0]._getDroneStateVector(0),
                        # state=np.hstack(
                        #     [obs[0:3], np.zeros(4), obs[3:15], np.resize(action, (4))]
                        # ),
                        control=np.zeros(12),
                        reward=list(test_env.get_attr("reward_dict")[0].values()),
                        done=done,
                    )
                sync(
                    np.floor(i * test_env.get_attr("AGGR_PHY_STEPS")[0]),
                    start,
                    test_env.get_attr("TIMESTEP")[0],
                )
                # if done:
                #     obs = test_env.reset()  # OPTIONAL EPISODE HALT
                steps += 1
                if ARGS.early_done and done:
                    break  # OPTIONAL EPISODE Break
        else:
            logger = Logger(
                logging_freq_hz=int(test_env.SIM_FREQ / test_env.AGGR_PHY_STEPS),
                num_drones=1,
                num_rewards=len(test_env.reward_dict),
                rewards_names=list(test_env.reward_dict.keys()),
                # done_names=list(test_env.term_dict.keys()),
                output_folder=output_folder,
            )

            obs = test_env.reset()
            start = time.time()
            steps = 0
            for i in range(
                test_env.EPISODE_LEN_SEC
                * int(test_env.SIM_FREQ / test_env.AGGR_PHY_STEPS)
            ):  # Up to 6''
                action, _states = model.predict(
                    obs, deterministic=True
                )  # OPTIONAL 'deterministic=False'
                obs, reward, done, info = test_env.step(action)
                total_reward += reward
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
                else:
                    logger.log(
                        drone=0,
                        timestamp=i / test_env.SIM_FREQ,
                        # TODO: figure out how to log the camera images
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
                steps += 1
                if ARGS.early_done and done:
                    break  # OPTIONAL EPISODE Break
        test_env.close()
        # logger.save_as_csv("sa")  # Optional CSV save
        print(
            "Evaluated model on {}-difficulty {}-env with mean reward {:.2f} +/- {:.2f}".format(
                difficulty, ARGS.env, mean_reward, std_reward
            )
        )
        print("Total Timesteps: ", steps)
        print("Total Reward: ", total_reward)
        test_results.append(total_reward)
        # logger.save_as_csv("sa")  # Optional CSV save
        if plot:
            logger.plot()
            logger.plot_rewards()
            logger.save_rewards(exp, mean_reward, std_reward)
    print("Render Results")
    print(test_results)
    print("Eval Results, rewards then stds")
    print(cum_results)
    print(cum_stds)


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
        "--gui",
        default=DEFAULT_GUI,
        type=str2bool,
        help="Whether to use PyBullet GUI (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--plot",
        default=DEFAULT_PLOT,
        type=str2bool,
        help="Whether to plot the simulation results (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--output_folder",
        default=DEFAULT_OUTPUT_FOLDER,
        type=str,
        help='Folder where to save logs (default: "results")',
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
    parser.add_argument(
        "--best",
        action="store_true",
        help="if included will run the best model, otherwise will run success if it exists",
    )
    parser.add_argument(
        "--seed", type=int, help="The random seed to use", metavar="",
    )
    parser.add_argument(
        "--all_difficulties",
        action="store_true",
        help="if included, the agent will be tested on all difficulty levels",
    )
    parser.add_argument(
        "--lower_difficulties",
        action="store_true",
        help="if included, the agent will be tested on only its difficulty level and lower",
    )
    parser.add_argument(
        "--alternate_difficulty",
        action="store_true",
        help="if included, the agent will be tested on the difficulty version it wasn't trained on, ie wall->no wall",
    )
    # parser.add_argument(
    #     "--env_kwargs", type=str, default="difficulty:", help="string of env", metavar="",
    # )
    ARGS = parser.parse_args()

    run(**vars(ARGS))
