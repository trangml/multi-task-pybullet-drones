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


class SlowdownReward(DenseReward):
    """Calculate the dense slowdown reward.

    This reward is supposed to help encourage the drone to slow down when approaching the landing zone

    """

    def __init__(self, scale, landing_zone_xyz, slowdown_dist):
        super().__init__(scale)
        self.landing_zone_xyz = landing_zone_xyz
        self.slowdown_dist = slowdown_dist

    def _calculateReward(self, state, drone_id=0):
        position = state[0:3]
        # only consider the x and y coordinates, as y can still change to get us closer
        velocity = state[10:12]
        target_position = self.landing_zone_xyz
        target_velocity = np.asarray([0, 0])
        pos_dist = np.linalg.norm(position - target_position)

        vel_dist = np.linalg.norm(velocity - target_velocity)

        if pos_dist < self.slowdown_dist:
            reward = min(POSITIVE_REWARD - (vel_dist / 5) ** 0.5, POSITIVE_REWARD)
        else:
            reward = ZERO_REWARD

        return reward

    ################################################################################
