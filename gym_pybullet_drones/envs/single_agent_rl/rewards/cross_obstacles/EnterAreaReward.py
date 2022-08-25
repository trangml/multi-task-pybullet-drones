import copy
import math
import numpy as np
from typing import Tuple, List

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


class EnterAreaReward(SparseReward):
    """Calculate the sparse entered area reward."""

    def __init__(self, scale, area: List[float]):
        """
        Generate

        _extended_summary_

        Parameters
        ----------
        scale : _type_
            _description_
        area : Tuple[Bounds, Bounds]
            x bounds, and y bounds
        """
        super().__init__(scale)
        self.area = (Bounds(area[0][0], area[0][1]), Bounds(area[1][0], area[1][1]))

    def _calculateReward(self, state):
        # For now, ignore height.
        position = state[0:2]
        in_area = True
        for bound, pos in zip(self.area, position):
            in_area = in_area and within_bounds(bound, pos)

        if in_area:
            return POSITIVE_REWARD
        return ZERO_REWARD

    ################################################################################
