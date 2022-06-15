import os
import numpy as np
import math
import abc

from gym import spaces
import pybullet as p
from gym_pybullet_drones.envs.single_agent_rl.obstacles.LandingZone import LandingZone
from gym_pybullet_drones.envs.single_agent_rl import BaseSingleAgentAviary
from gym_pybullet_drones.envs.single_agent_rl.rewards.Reward import Reward


POSITIVE_REWARD = 1
NEGATIVE_REWARD = -1


class SparseReward(Reward):
    """Sparse reward for a drone."""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def _calculateReward(self, state):
        return 0

    def calculateReward(self, state):
        return self._calculateReward(state) * self.scale


class BoundsReward(SparseReward):
    """Checks rewards for out of bounds"""

    def __init__(self, scale, bounds, useTimeScaling=False):
        super().__init__(scale)
        # Defined as [[x_high, y_high, z_high], [x_low, y_low, z_low]]
        self.bounds = bounds
        self.XYZ_IDX = [0, 1, 2]
        self.useTimeScaling = useTimeScaling

    def _calculateReward(self, state):
        position = state[0:3]

        for dim_idx in self.XYZ_IDX:
            if (
                state[dim_idx] > self.bounds[0][dim_idx]
                or state[dim_idx] < self.bounds[1][dim_idx]
            ):
                # self.aviary.completeEpisode = True
                if self.useTimeScaling:
                    return NEGATIVE_REWARD * (
                        1
                        - (
                            (self.aviary.step_counter / self.aviary.SIM_FREQ)
                            / self.aviary.EPISODE_LEN_SEC
                        )
                    )
                return NEGATIVE_REWARD
        return 0


class LandingReward(SparseReward):
    """Calculate the landing reward."""

    def __init__(self, scale, landing_zone_xyz):
        super().__init__(scale)
        self.landing_zone_xyz = landing_zone_xyz
        self.landing_frames = 0

    def _calculateReward(self, state):
        position = state[0:3]

        # only consider x and y
        target_position = self.landing_zone_xyz
        pos_dist = np.linalg.norm(position[0:2] - target_position[0:2])

        if pos_dist < 0.15:
            self.landing_frames += 1
            if self.landing_frames >= 10:
                # self.aviary.completeEpisode = True
                return 2240
            else:
                y_dist = np.linalg.norm(position[2] - (target_position[2] - 0.1))
                return 80 + (1 - y_dist) * 10
        else:
            return 0

    ################################################################################


class SpeedReward(SparseReward):
    """Calculate the landing reward."""

    def __init__(self, scale, max_speed):
        super().__init__(scale)
        self.max_speed = max_speed

    def _calculateReward(self, state):
        velocity = state[10:13]
        vel = np.linalg.norm(velocity)

        if vel > self.max_speed:
            return -1
        else:
            return 0

    ################################################################################
