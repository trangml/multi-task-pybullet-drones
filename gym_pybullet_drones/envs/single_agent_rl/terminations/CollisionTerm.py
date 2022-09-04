import copy
import math
import numpy as np
import pybullet as p
from typing import Tuple, List

from gym_pybullet_drones.envs.single_agent_rl.rewards.utils import (
    Bounds,
    within_bounds,
    POSITIVE_REWARD,
    NEGATIVE_REWARD,
    ZERO_REWARD,
)
from gym_pybullet_drones.envs.single_agent_rl.terminations import Terminations


class CollisionTerm(Terminations):
    """Calculate the sparse entered area reward."""

    def __init__(self, area, drone_id, client):
        """
        Generate

        _extended_summary_

        Parameters
        ----------
        scale : _type_
            _description_
        area : Tuple[Bounds, Bounds]
            safe/landing zone area
        """
        super().__init__()
        self.drone_id = drone_id
        self.CLIENT = client
        self.area = (Bounds(area[0][0], area[0][1]), Bounds(area[1][0], area[1][1]))

    def _calculateTerm(self, state):
        # For now, ignore height.
        contact_pts = p.getContactPoints(
            bodyA=self.drone_id, physicsClientId=self.CLIENT
        )
        if len(contact_pts) > 0:
            position = state[0:2]
            in_area = True
            for bound, pos in zip(self.area, position):
                in_area = in_area and within_bounds(bound, pos)

            if not in_area:
                return True
        return False

    ################################################################################
