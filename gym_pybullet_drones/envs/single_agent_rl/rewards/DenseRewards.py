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


class DenseReward(Reward):
    """Dense reward for a drone.

    The reward is the distance to the closest obstacle.

    """

    def __init__(self, aviary, scale):
        self.aviary = aviary
        self.scale = scale

    def _calculateReward():
        return 0

    def _getDroneStateVector(self, drone_id=0):
        """Get the drone state vector.

        Parameters
        ----------
        drone_id : int
            Drone id.

        Returns
        -------
        np.array
            Drone state vector.

        """
        # state = p.getLinkState(drone_id, 0)
        state = self.aviary._getDroneStateVector(drone_id)
        return np.array(state)

    def calculateReward(self):
        return self._calculateReward() * self.scale


class DistanceReward(DenseReward):
    """Calculate the dense distance reward."""

    def __init__(self, aviary, scale, landing_zone_xyz):
        super().__init__(aviary, scale)
        self.landing_zone_xyz = landing_zone_xyz

    def _calculateReward(self):
        # get the actual state, not the obs
        state = self._getDroneStateVector(0)
        position = state[0:3]
        target_position = self.landing_zone_xyz
        pos_dist = np.linalg.norm(position[0:2] - target_position[0:2])
        # if pos_dist < 0.1:
        #     return POSITIVE_REWARD
        # if self.aviary.step_counter > 0:
        #     dist_delta = pos_dist - self.last_pos_dist
        #     self.last_pos_dist = pos_dist
        #     if dist_delta < 0:
        #         return POSITIVE_REWARD
        #     else:
        #         return NEGATIVE_REWARD
        # else:
        #     self.last_pos_dist = pos_dist
        #     return 0

        max_dist = np.linalg.norm(target_position) + 1
        reward = 1 - ((pos_dist) / 5) ** 0.5
        return reward

    ################################################################################


class SlowdownReward(DenseReward):
    """Calculate the dense slowdown reward.

    This reward is supposed to help encourage the drone to slow down when approaching the landing zone

    """

    def __init__(self, aviary, scale, landing_zone_xyz, slowdown_dist):
        super().__init__(aviary, scale)
        self.landing_zone_xyz = landing_zone_xyz
        self.slowdown_dist = slowdown_dist

    def _calculateReward(self):
        state = self._getDroneStateVector(0)
        position = state[0:3]
        # only consider the x and y coordinates, as y can still change to get us closer
        velocity = state[10:12]
        target_position = self.landing_zone_xyz
        target_velocity = np.asarray([0, 0])
        pos_dist = np.linalg.norm(position - target_position)

        vel_dist = np.linalg.norm(velocity - target_velocity)

        if pos_dist < self.slowdown_dist:
            reward = 1 - (vel_dist / 5) ** 0.5
        else:
            reward = 0

        return reward


class FieldCoverageReward(DenseReward):
    """Calculate the field coverage reward."""

    def __init__(self, aviary, scale, field):
        super().__init__(aviary, scale)
        self.field = field

    def _calculateReward(self):
        state = self._getDroneStateVector(0)
        position = state[0:3]
        if self.field.checkIsCovered((position[0], position[1])):
            return 1
        else:
            return -0.01

    ################################################################################
