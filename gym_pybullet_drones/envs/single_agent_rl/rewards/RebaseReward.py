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


class RebaseReward(DenseReward):
    """Calculate the field coverage reward."""

    def __init__(self, scale, landing_zone_xyz, interval_secs):
        super().__init__(scale)
        self.landing_zone_xyz = landing_zone_xyz
        self.interval_secs = interval_secs
        self.step_counter = 0
        self.SIM_FREQ = 240 #TODO: may have to change this

    def _calculateReward(self, state):
        position = state[0:3]
        self.step_counter += 1
        # if we are at a time when we need to rebase, then reward that
        if self.step_counter / self.SIM_FREQ % self.interval_secs == 0:
            if np.linalg.norm(position - self.landing_zone_xyz) < 0.125:
                return POSITIVE_REWARD
            else:
                return NEGATIVE_REWARD
        else:
            return ZERO_REWARD

    ################################################################################
