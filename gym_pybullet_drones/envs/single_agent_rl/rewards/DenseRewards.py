import os
import numpy as np
import math
import abc

from gym import spaces
import pybullet as p
from gym_pybullet_drones.envs.single_agent_rl.obstacles.LandingZone import LandingZone
from gym_pybullet_drones.envs.single_agent_rl import BaseSingleAgentAviary


POSITIVE_REWARD = 1
NEGATIVE_REWARD = -1


class DenseReward:
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

    def calculateReward():
        return self._calculateReward() * self.scale


class BoundsReward(DenseReward):
    """Checks rewards for out of bounds"""

    def __init__(self, aviary, scale, bounds, useTimeScaling=False):
        super().__init__(aviary, scale)
        self.landing_zone = landing_zone
        # Defined as [[x_high, y_high, z_high], [x_low, y_low, z_low]]
        self.bounds = bounds
        self.XYZ_IDX = [0, 1, 2]
        self.useTimeScaling = useTimeScaling

    def _calculateReward(self):
        state = self._getDroneStateVector(0)
        position = state[0:3]

        for dim_idx in self.XYZ_IDX:
            if (
                state[dim_idx] > self.bounds[0, dim_idx]
                or state[dim_idx] < self.bounds[1, dim_idx]
            ):
                self.aviary.completeEpisode = True
                if self.useTimeScaling:
                    return NEGATIVE_REWARD * (
                        1
                        - (
                            (self.aviary.step_counter / self.aviary.SIM_FREQ)
                            / self.aviary.EPISODE_LEN_SEC
                        )
                    )
                return NEGATIVE_REWARD


class LandingReward(DenseReward):
    """Calculate the landing reward."""

    def __init__(self, aviary, scale, landing_zone: LandingZone):
        super().__init__(aviary, scale)
        self.landing_zone = landing_zone

    def _calculateReward(self):
        state = self._getDroneStateVector(0)
        position = state[0:3]
        velocity = state[10:13]
        target_position = self.landing_zone.xyz
        target_velocity = np.asarray([0, 0, 0])
        pos_dist = np.linalg.norm(position - target_position)
        vel_dist = np.linalg.norm(velocity - target_velocity)

        if pos_dist < 0.1 and vel_dist < 0.1:
            self.aviary.landing_frames += 1
            self.aviary.completeEpisode = True
            if self.aviary.landing_frames >= 10:
                return 2240
            else:
                return 80
        elif pos_dist < 1:
            if vel_dist < 1:
                inv_vel_dist = 2 - vel_dist
            else:
                inv_vel_dist = 1 / vel_dist

            return 2 - inv_vel_dist + 0.85 * inv_vel_dist
        # Penalize if out of bounds
        return 1 / pos_dist

    ################################################################################
