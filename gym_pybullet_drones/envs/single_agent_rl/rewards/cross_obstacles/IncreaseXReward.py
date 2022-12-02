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


class IncreaseXReward(DenseReward):
    """Calculate the sparse entered area reward."""

    def __init__(self, scale: float):
        """
        rewards the agent for moving in the positive x direction

        Parameters
        ----------
        scale : _type_
            _description_
        area : Tuple[Bounds, Bounds]
            x bounds, and y bounds
        """
        super().__init__(scale)

    # def reset(self):
    #     self.last_pos

    def _calculateReward(self, state, drone_id):
        # For now, reward based on just x position
        position = state[0]
        if position > 0:
            return position / 10
        return ZERO_REWARD

    ################################################################################


class IncreaseXRewardV2(DenseReward):
    """Calculate the dense increase x reward."""

    def __init__(self, scale: float, delta: float = 0.2):
        """
        rewards the agent for moving in the positive x direction

        Parameters
        ----------
        scale : _type_
            _description_
        area : Tuple[Bounds, Bounds]
            x bounds, and y bounds
        """
        super().__init__(scale)
        self.max_x = -1000
        self.delta = delta

    def reset(self):
        self.max_x = -1000

    def _calculateReward(self, state, drone_id):
        # For now, reward based on just x position
        position = state[0]
        if position > self.max_x + self.delta:
            self.max_x = position
            return POSITIVE_REWARD
        return ZERO_REWARD

    ################################################################################
