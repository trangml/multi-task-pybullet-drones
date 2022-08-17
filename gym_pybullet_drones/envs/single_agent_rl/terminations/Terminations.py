import math
import os
import numpy as np
from gym_pybullet_drones.envs.single_agent_rl.rewards.utils import Bounds

class Terminations:
    def __init__(self):
        pass
    def _calculateTerm(self, state):
        return False
    def calculateTerm(self, state):
        return self._calculateTerm(state)

def getTermDict(term):
    termDict = {}
    for term in term:
        termDict[term.__class__.__name__] = False
    termDict["total"] = False
    return termDict

class BoundsTerm(Terminations):
    """Checks terminals for out of bounds"""

    def __init__(self, bounds):
        super().__init__()
        # Defined as [[x_high, y_high, z_high], [x_low, y_low, z_low]]
        self.bounds = bounds
        self.XYZ_IDX = [0, 1, 2]

    def _calculateTerm(self, state):
        position = state[0:3]

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
        self.bounds = Bounds(min = -1, max=1)

    def _calculateReward(self, state):
        rp = state[7:9]
        in_bounds = all(within_bounds(self.bounds, field) for field in rp)
        return not in_bounds