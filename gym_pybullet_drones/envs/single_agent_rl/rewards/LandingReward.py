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


class LandingReward(SparseReward):
    """Calculate the landing reward."""

    def __init__(self, scale, landing_zone_xyz):
        super().__init__(scale)
        self.landing_zone_xyz = landing_zone_xyz
        self.landing_frames = 0

    def _calculateReward(self, state):
        position = state[0:3]

        # only consider x and y
        target_position = self.landing_zone_xyz
        pos_dist = np.linalg.norm(position[0:2] - target_position[0:2])

        if pos_dist < 0.15:
            self.landing_frames += 1
            if self.landing_frames >= 10:
                # self.aviary.completeEpisode = True
                return 10 * POSITIVE_REWARD
            else:
                y_dist = np.linalg.norm(position[2] - (target_position[2] - 0.1))
                return POSITIVE_REWARD * (1 - y_dist)
        else:
            return 0

    ################################################################################


class LandingRewardV2(SparseReward):
    """Calculate the landing reward."""

    def __init__(self, scale, landing_zone_xyz):
        super().__init__(scale)
        self.landing_zone_xyz = np.array(landing_zone_xyz)
        self.target_position = copy.deepcopy(self.landing_zone_xyz)
        self.landing_frames = 0
        self.rp_bounds = Bounds(-1, 1)
        DRONE_MID_Z = 0.01347
        self.goal_height = landing_zone_xyz[-1] + DRONE_MID_Z
        self.is_landed = False

    def reset(self):
        self.landing_frames = 0
        self.is_landed = False

    def _calculateReward(self, state):
        position = state[0:3]

        # only consider x and y
        target_position = self.landing_zone_xyz
        pos_dist = np.linalg.norm(position[0:2] - target_position[0:2])

        rp = state[7:9]
        in_rp_bounds = all(within_bounds(self.rp_bounds, field) for field in rp)
        above_goal = pos_dist < 0.099
        if above_goal and in_rp_bounds:
            self.landing_frames += 1
            if math.isclose(position[2], self.goal_height, rel_tol=1e-2):
                self.is_landed = True
                return POSITIVE_REWARD
            else:
                z_dist = position[2] - (self.goal_height)
                max_dist = (
                    1 - self.goal_height
                )  # 1 here assumes that we are bounding above 1 as OOB

                # TODO: Test squaring this quantity
                return POSITIVE_REWARD * ((max_dist - z_dist) / max_dist)
        else:
            return ZERO_REWARD

    ################################################################################
