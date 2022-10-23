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
from gym_pybullet_drones.envs.single_agent_rl.rewards.Reward import DenseReward


class DeltaDistanceReward(DenseReward):
    """Calculate the dense delta distance reward.

    Reward = +1 if we are closer than the last step, -1 if we are further away or the same

    Reward is always 1 if we are closer than 0.1

    """

    def __init__(self, scale, landing_zone_xyz):
        super().__init__(scale)
        self._initial_step = True
        self.landing_zone_xyz = np.array(landing_zone_xyz)
        self.target_position = copy.deepcopy(self.landing_zone_xyz)
        DRONE_MID_Z = 0.01347
        self.target_position[2] = 2 * self.target_position[2] + DRONE_MID_Z
        self.last_pos_dist = 0

    def reset(self):
        self._initial_step = True
        self.last_pos_dist = 0

    def _calculateReward(self, state, drone_id):
        # get the actual state, not the obs
        position = state[0:3]
        pos_dist = np.linalg.norm(position[0:2] - self.target_position[0:2])
        if pos_dist < 0.1:
            return POSITIVE_REWARD
        if not self._initial_step:
            # we want the distance to decrease at each step, thus negative dist_delta
            dist_delta = pos_dist - self.last_pos_dist
            self.last_pos_dist = pos_dist
            if dist_delta < -0.01:
                return POSITIVE_REWARD
            else:
                return NEGATIVE_REWARD
        else:
            self.last_pos_dist = pos_dist
            self._initial_step = False
            return ZERO_REWARD


class DeltaDistanceRewardV2(DenseReward):
    """Calculate the dense delta distance reward.

    Reward = +1 if we are closer than the last step, -1 if we are further away or the same

    Reward is always 1 if we are closer than 0.1

    """

    def __init__(self, scale, landing_zone_xyz):
        super().__init__(scale)
        self._initial_step = True
        self.landing_zone_xyz = np.array(landing_zone_xyz)
        self.target_position = copy.deepcopy(self.landing_zone_xyz)
        DRONE_MID_Z = 0.01347
        self.target_position[2] = 2 * self.target_position[2] + DRONE_MID_Z
        self.last_pos_dist = 0

    def reset(self):
        self._initial_step = True
        self.last_pos_dist = 0

    def _calculateReward(self, state, drone_id):
        # get the actual state, not the obs
        position = state[0:3]
        pos_dist = np.linalg.norm(position - self.target_position)
        if pos_dist < 0.099:
            return POSITIVE_REWARD
        if not self._initial_step:
            # we want the distance to decrease at each step, thus negative dist_delta
            dist_delta = pos_dist - self.last_pos_dist
            self.last_pos_dist = pos_dist
            if dist_delta < -0.01:
                return POSITIVE_REWARD
            else:
                return NEGATIVE_REWARD
        else:
            self.last_pos_dist = pos_dist
            self._initial_step = False
            return ZERO_REWARD


class SafeDeltaDistanceReward(DenseReward):
    """Calculate the dense delta distance reward.

    Reward = +1 if we are closer than the last step, 0 otherwise

    Reward is always 1 if we are closer than 0.1

    """

    def __init__(self, scale, landing_zone_xyz, delta=0.2):
        super().__init__(scale)
        self._initial_step = True
        self.landing_zone_xyz = np.array(landing_zone_xyz)
        self.target_position = copy.deepcopy(self.landing_zone_xyz)
        DRONE_MID_Z = 0.01347
        self.target_position[2] = 2 * self.target_position[2] + DRONE_MID_Z
        self.last_pos_dist = 0
        self.delta = delta

    def reset(self):
        self._initial_step = True
        self.last_pos_dist = 0

    def _calculateReward(self, state, drone_id):
        # get the actual state, not the obs
        position = state[0:3]
        pos_dist = np.linalg.norm(position - self.target_position)
        if pos_dist < 0.099:
            return POSITIVE_REWARD

        if not self._initial_step:
            # we want the distance to decrease at each step, thus negative dist_delta
            dist_delta = self.last_pos_dist - pos_dist
            if dist_delta > 0.2:
                self.last_pos_dist = pos_dist
                return POSITIVE_REWARD
            else:
                return ZERO_REWARD
        else:
            self.last_pos_dist = pos_dist
            self._initial_step = False
            return ZERO_REWARD
