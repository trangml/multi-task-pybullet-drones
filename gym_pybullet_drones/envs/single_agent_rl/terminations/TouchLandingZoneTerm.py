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


class TouchLandingZoneTerm(Terminations):
    """Calculate the touch landing zone termination function.

    """

    def __init__(
        self, client=0,
    ):
        """
        Generate

        _extended_summary_

        Parameters
        ----------
        scale : _type_
            _description_
        """
        super().__init__()
        self.CLIENT = client
        self.landing_zone = None

    def setClient(self, client):
        self.CLIENT = client

    def defineBody(self, body_id: int):
        self.landing_zone = body_id

    def _calculateTerm(self, state, drone_id):
        # For now, ignore height.
        contact_pts = p.getContactPoints(
            bodyA=drone_id, bodyB=self.landing_zone, physicsClientId=self.CLIENT
        )
        # if there are contact points
        if len(contact_pts) > 0:
            # check the excluded contact points
            return True
        return False

    ################################################################################
