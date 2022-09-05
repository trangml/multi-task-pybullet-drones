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
from gym_pybullet_drones.envs.single_agent_rl.terminations import Terminations


class EnterAreaTerm(Terminations):
    """Calculate the sparse entered area reward."""

    def __init__(self, area: List[float]):
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
        super().__init__()
        self.area = (Bounds(area[0][0], area[0][1]), Bounds(area[1][0], area[1][1]))
        self.steps_in_area = 0

    def reset(self):
        self.steps_in_area = 0
        return super().reset()

    def _calculateTerm(self, state):
        # For now, ignore height.
        position = state[0:2]
        in_area = True
        for bound, pos in zip(self.area, position):
            in_area = in_area and within_bounds(bound, pos)
            self.steps_in_area += 1

        return self.steps_in_area > 20

    ################################################################################
