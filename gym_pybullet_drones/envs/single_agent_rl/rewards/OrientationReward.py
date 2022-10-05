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


class OrientationReward(SparseReward):
    """Calculate the reward for staying level based off roll and pitch."""

    def __init__(self, scale):
        super().__init__(scale)
        # 1 radian is about 57 degrees
        self.bounds = Bounds(min=-1, max=1)

    def _calculateReward(self, state, drone_id=0):
        rp = state[7:9]
        in_bounds = all(within_bounds(self.bounds, field) for field in rp)
        if in_bounds:
            return POSITIVE_REWARD
        else:
            return NEGATIVE_REWARD

    ################################################################################
