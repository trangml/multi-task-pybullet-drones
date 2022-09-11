#!/bin/bash

#python rl_singleagent.py env_kwargs.reward_components.BoundsReward.scale=1 tag=bounds_scale_1
for i in {0..4}; do
    python rl_singleagent.py --config-name=cross_obstacles.yaml tag=inc_difficulty_1 seed=$i
done