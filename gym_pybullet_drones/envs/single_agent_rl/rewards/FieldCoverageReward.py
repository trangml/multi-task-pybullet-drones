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


class FieldCoverageReward(DenseReward):
    """Calculate the field coverage reward."""

    def __init__(self, scale, field):
        super().__init__(scale)
        self.field = field

    def reset(self):
        self.field.reset()

    def _calculateReward(self, state):
        position = state[0:3]
        if self.field.checkIsCovered((position[0], position[1])):
            return 1
        else:
            return 0

    ################################################################################
