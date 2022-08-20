import math
import numpy as np

from gym_pybullet_drones.envs.single_agent_rl.rewards.utils import (
    Bounds,
    within_bounds,
    POSITIVE_REWARD,
    NEGATIVE_REWARD,
    ZERO_REWARD,
)
from gym_pybullet_drones.envs.single_agent_rl.rewards.Reward import (
    DenseReward,
    SparseReward,
)


class BoundsReward(SparseReward):
    """Checks rewards for out of bounds"""

    def __init__(self, scale, bounds, use_time_scaling=False):
        super().__init__(scale)
        # Defined as [[x_high, y_high, z_high], [x_low, y_low, z_low]]
        self.bounds = bounds
        self.XYZ_IDX = [0, 1, 2]
        self.use_time_scaling = use_time_scaling
        self.step = 0

    def _calculateReward(self, state):
        self.step += 1
        position = state[0:3]

        for dim_idx in self.XYZ_IDX:
            if (
                state[dim_idx] > self.bounds[0][dim_idx]
                or state[dim_idx] < self.bounds[1][dim_idx]
            ):
                # self.aviary.completeEpisode = True
                if self.use_time_scaling:
                    return NEGATIVE_REWARD * (1 - 1 / np.sqrt(self.step))
                return NEGATIVE_REWARD
        return 0
