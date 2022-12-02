#!/bin/bash

#python rl_singleagent.py env_kwargs.reward_components.BoundsReward.scale=1 tag=bounds_scale_1
for i in {11..16}; do
    python rl_singleagent.py --config-name=norm_reward_cross_obstacles.yaml tag=norm_reward_inc_difficulty_${i} seed=0 env_kwargs.difficulty=$i exp=none
done