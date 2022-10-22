import copy
import math
from typing import List, Tuple

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.single_agent_rl.rewards.Reward import (
    DenseReward,
    SparseReward,
)
from gym_pybullet_drones.envs.single_agent_rl.rewards.utils import (
    NEGATIVE_REWARD,
    POSITIVE_REWARD,
    ZERO_REWARD,
    Bounds,
    within_bounds,
)


class NoCollisionReward(SparseReward):
    """Calculate the dense collision avoidance reward, which is a constant award for not crashing."""

    def __init__(self, scale, area=None, client=0):
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
        self.area = (
            (Bounds(area[0][0], area[0][1]), Bounds(area[1][0], area[1][1]))
            if area
            else None
        )
        self.checkArea = True if area else False
        self.exclude = []

    def setClient(self, client):
        self.CLIENT = client

    def excludeBody(self, body_id: int):
        self.exclude.append(body_id)

    def _getExcludedContacts(self, drone_id):
        excluded_contact_pts = []
        for body_id in self.exclude:
            excluded_contact_pts.extend(
                p.getContactPoints(
                    bodyA=drone_id, bodyB=body_id, physicsClientId=self.CLIENT
                )
            )
        return excluded_contact_pts

    def _calculateTerm(self, state, drone_id):
        # For now, ignore height.
        contact_pts = p.getContactPoints(bodyA=drone_id, physicsClientId=self.CLIENT)
        # if there are contact points
        if len(contact_pts) > 0:
            # check the excluded contact points
            excluded_contact_pts = self._getExcludedContacts(drone_id)
            if len(excluded_contact_pts) != len(contact_pts):
                position = state[0:2]
                if self.checkArea:
                    # check if we're in a safe area
                    in_area = True
                    for bound, pos in zip(self.area, position):
                        in_area = in_area and within_bounds(bound, pos)

                    if not in_area:
                        return NEGATIVE_REWARD
                else:
                    # if we don't have a safe area, we're done
                    return NEGATIVE_REWARD
        return POSITIVE_REWARD

