import math
import os
from typing import Type
import numpy as np
from gym_pybullet_drones.envs.single_agent_rl.rewards.utils import Bounds, within_bounds


class Terminations:
    """
    Base Terminal class

    """

    def __init__(self):
        pass

    def reset(self):
        """Resets the terminal function"""
        pass

    def _calculateTerm(self, state, drone_id):
        # if this method is not overriden, then try returning the version which only takes in a state
        return False

    def calculateTerm(self, state, drone_id=1):
        return self._calculateTerm(state, drone_id)


def getTermDict(term):
    """Converts a term to a dictionary"""
    termDict = {}
    for ter in term:
        termDict[ter.__class__.__name__] = False
    termDict["total"] = False
    return termDict


class BoundsTerm(Terminations):
    """Checks terminals for out of bounds"""

    def __init__(self, bounds):
        super().__init__()
        # Defined as [[x_high, y_high, z_high], [x_low, y_low, z_low]]
        self.bounds = bounds
        self.XYZ_IDX = [0, 1, 2]

    def _calculateTerm(self, state, drone_id):
        for dim_idx in self.XYZ_IDX:
            if (
                state[dim_idx] > self.bounds[0][dim_idx]
                or state[dim_idx] < self.bounds[1][dim_idx]
            ):
                return True
        return False


class OrientationTerm(Terminations):
    """Checks rewards for out of bounds"""

    def __init__(self):
        super().__init__()
        # Defined as [[x_high, y_high, z_high], [x_low, y_low, z_low]]
        self.bounds = Bounds(min=-1, max=1)

    def _calculateTerm(self, state, drone_id):
        """
        Calculates if an agent fails the orientation terminal condition

        Parameters
        ----------
        state : np.ndarray
            the states

        Returns
        -------
        bool
            true if the agent is not within orientation bounds
        """
        rp = state[7:9]
        in_bounds = all(within_bounds(self.bounds, field) for field in rp)
        return not in_bounds
