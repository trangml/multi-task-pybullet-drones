import copy
import math
from typing import List, Tuple

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.single_agent_rl.rewards.utils import (
    NEGATIVE_REWARD,
    POSITIVE_REWARD,
    ZERO_REWARD,
    Bounds,
    within_bounds,
)
from gym_pybullet_drones.envs.single_agent_rl.terminations import Terminations


class CollisionTerm(Terminations):
    """Calculate the sparse entered area reward."""

    def __init__(self, area, client=0):
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
        self.CLIENT = client
        self.area = (Bounds(area[0][0], area[0][1]), Bounds(area[1][0], area[1][1]))

    def setClient(self, client):
        self.CLIENT = client

    def _calculateTerm(self, state, drone_id):
        # For now, ignore height.
        contact_pts = p.getContactPoints(bodyA=drone_id, physicsClientId=self.CLIENT)
        if len(contact_pts) > 0:
            position = state[0:2]
            in_area = True
            for bound, pos in zip(self.area, position):
                in_area = in_area and within_bounds(bound, pos)

            if not in_area:
                return True
        return False

    ################################################################################
