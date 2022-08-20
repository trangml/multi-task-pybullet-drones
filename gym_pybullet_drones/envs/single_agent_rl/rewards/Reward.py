import os
import numpy as np
import math
import abc

from gym import spaces
import pybullet as p
from gym_pybullet_drones.envs.single_agent_rl.obstacles.LandingZone import LandingZone
from gym_pybullet_drones.envs.single_agent_rl.rewards.utils import (
    Bounds,
    within_bounds,
    POSITIVE_REWARD,
    NEGATIVE_REWARD,
    ZERO_REWARD,
)


class Reward:
    def __init__(self):
        pass

    def calculateReward(self):
        return 0


def getRewardDict(rewards):
    rewardDict = {}
    for reward in rewards:
        rewardDict[reward.__class__.__name__] = 0
    rewardDict["total"] = 0
    return rewardDict


class DenseReward(Reward):
    """Dense reward for a drone.

    """

    def __init__(self, scale):
        self.scale = scale

    def _calculateReward(self, state):
        return 0

    def calculateReward(self, state):
        if self.scale == 0:
            return 0
        return self._calculateReward(state) * self.scale


class SparseReward(Reward):
    """Sparse reward for a drone.

    """

    def __init__(self, scale):
        self.scale = scale

    def _calculateReward(self, state):
        return 0

    def calculateReward(self, state):
        if self.scale == 0:
            return 0
        return self._calculateReward(state) * self.scale
