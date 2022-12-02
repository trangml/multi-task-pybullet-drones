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


class SpeedReward(SparseReward):
    """Calculate the landing reward."""

    def __init__(self, scale, max_speed):
        super().__init__(scale)
        self.max_speed = max_speed

    def _calculateReward(self, state, drone_id):
        velocity = state[10:13]
        vel = np.linalg.norm(velocity)

        if vel > self.max_speed:
            return -1
        else:
            return 0

    ################################################################################
