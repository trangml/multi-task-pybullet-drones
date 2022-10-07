import copy
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


class DistanceReward(DenseReward):
    """Calculate the dense distance reward."""

    def __init__(self, scale, landing_zone_xyz):
        super().__init__(scale)
        self.landing_zone_xyz = landing_zone_xyz

    def _calculateReward(self, state, drone_id):
        position = state[0:3]
        target_position = self.landing_zone_xyz
        pos_dist = np.linalg.norm(position[0:2] - target_position[0:2])

        # max_dist = np.linalg.norm(target_position) + 1
        reward = min(POSITIVE_REWARD - ((pos_dist) / 5) ** 0.5, POSITIVE_REWARD)
        return reward

    ################################################################################


class DistanceRewardV2(DenseReward):
    """Calculate the dense distance reward."""

    def __init__(self, scale, landing_zone_xyz):
        super().__init__(scale)
        self.landing_zone_xyz = np.array(landing_zone_xyz)
        self.target_position = copy.deepcopy(self.landing_zone_xyz)
        DRONE_MID_Z = 0.01347
        self.target_position[2] = 2 * self.target_position[2] + DRONE_MID_Z

    def _calculateReward(self, state, drone_id):
        position = state[0:3]
        pos_dist = np.linalg.norm(position - self.target_position)
        reward = min(POSITIVE_REWARD - ((pos_dist) / 5) ** 0.5, POSITIVE_REWARD)
        return reward
