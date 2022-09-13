#!/bin/bash

#python rl_singleagent.py env_kwargs.reward_components.BoundsReward.scale=1 tag=bounds_scale_1
for i in {0..4}; do
    python rl_singleagent.py --config-name=cross_obstacles.yaml tag=inc_difficulty_15_from_5 seed=$i env_kwargs.difficulty=15 exp=/home/mtrang/Documents/vt/research/multiagent-pybullet-drones/experiments/learning/results/save-cross-obstacles-ppo-BOTH-RPM-inc_difficulty_5-09.11.2022_21.39.19
done