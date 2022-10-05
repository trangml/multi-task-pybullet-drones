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


class CollisionReward(SparseReward):
    """Calculate the sparse collision reward, which is a punishment."""

    def __init__(self, scale, area, client):
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
        super().__init__(scale)
        self.CLIENT = client
        self.area = (Bounds(area[0][0], area[0][1]), Bounds(area[1][0], area[1][1]))

    def _calculateReward(self, state, drone_id=0):
        # For now, ignore height.
        contact_pts = p.getContactPoints(bodyA=drone_id, physicsClientId=self.CLIENT)
        if len(contact_pts) > 0:
            # TODO: Update with things which can be collided without penalty ie, landing zone
            position = state[0:2]
            in_area = True
            for bound, pos in zip(self.area, position):
                in_area = in_area and within_bounds(bound, pos)

            if not in_area:
                return NEGATIVE_REWARD
        return ZERO_REWARD

    ################################################################################
