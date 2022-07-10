#!/bin/bash

#python rl_singleagent.py env_kwargs.reward_components.BoundsReward.scale=1 tag=bounds_scale_1
python rl_singleagent.py env_kwargs.reward_components.BoundsReward.scale=5 tag=bounds_scale_5
python rl_singleagent.py env_kwargs.reward_components.BoundsReward.scale=2 env_kwargs.reward_components.BoundsReward.use_time_scaling=true tag=bounds_scale_2_use_time_scaling
python rl_singleagent.py env_kwargs.reward_components.BoundsReward.scale=10 env_kwargs.reward_components.BoundsReward.use_time_scaling=true tag=bounds_scale_10_use_time_scaling