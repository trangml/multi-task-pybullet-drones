#!/bin/bash

#python rl_singleagent.py env_kwargs.reward_components.BoundsReward.scale=1 tag=bounds_scale_1
for i in {0..4}; do
    python rl_singleagent.py --config-name=cross_obstacles.yaml tag=inc_difficulty_11_collision_3_seed_${i} seed=$i env_kwargs.difficulty=11 exp=results/save-cross-obstacles-ppo-BOTH-RPM-inc_difficulty_0_collision_seed_3-10.10.2022_03.01.10
done