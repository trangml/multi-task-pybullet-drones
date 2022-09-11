#!/bin/bash

#python rl_singleagent.py env_kwargs.reward_components.BoundsReward.scale=1 tag=bounds_scale_1
for i in {0..4}; do
    python rl_singleagent.py --config-name=cross_obstacles.yaml tag=inc_difficulty_3 seed=$i env_kwargs.difficulty=3 exp=/home/mtrang/Documents/vt/research/multiagent-pybullet-drones/experiments/learning/results/save-cross-obstacles-ppo-BOTH-RPM-inc_difficulty_2-09.05.2022_12.33.52
done