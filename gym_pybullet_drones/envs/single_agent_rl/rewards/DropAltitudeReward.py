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


class DropAltitudeReward(DenseReward):
    """Calculate the dense drop altitude reward

    Reward = +1 if we are lower or the same z height than the last step, -1 otherwise

    """

    def __init__(self, scale, landing_zone_xyz, start_dist):
        super().__init__(scale)
        self.landing_zone_xyz = np.array(landing_zone_xyz)
        self.target_position = copy.deepcopy(self.landing_zone_xyz)
        DRONE_MID_Z = 0.01347
        self.target_position[2] = 2 * self.target_position[2] + DRONE_MID_Z
        self.start_dist = start_dist
        self.reward = False
        self.prev_alt = 1

    def reset(self):
        self.reward = False
        self.prev_alt = 1

    def _calculateReward(self, state, drone_id):
        # get the actual state, not the obs
        position = state[0:3]
        pos_dist = np.linalg.norm(position[0:2] - self.target_position[0:2])
        if pos_dist < self.start_dist:
            if self.reward:
                if position[-1] <= self.prev_alt + 1e-6:
                    self.prev_alt = position[-1]
                    return POSITIVE_REWARD
                else:
                    self.prev_alt = position[-1]
                    return NEGATIVE_REWARD
            else:
                self.reward = True
                self.prev_alt = position[-1]
                return POSITIVE_REWARD
        else:
            self.reward = False

        return ZERO_REWARD
