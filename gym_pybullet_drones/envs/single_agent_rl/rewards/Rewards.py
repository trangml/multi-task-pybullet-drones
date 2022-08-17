import os
from tracemalloc import start
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

    The reward is the distance to the closest obstacle.

    """

    def __init__(self, scale):
        self.scale = scale

    def _calculateReward(self, state):
        return 0

    def calculateReward(self, state):
        if self.scale == 0:
            return 0
        return self._calculateReward(state) * self.scale


class DistanceReward(DenseReward):
    """Calculate the dense distance reward."""

    def __init__(self, scale, landing_zone_xyz):
        super().__init__(scale)
        self.landing_zone_xyz = landing_zone_xyz

    def _calculateReward(self, state):
        position = state[0:3]
        target_position = self.landing_zone_xyz
        pos_dist = np.linalg.norm(position[0:2] - target_position[0:2])

        max_dist = np.linalg.norm(target_position) + 1
        reward = min(POSITIVE_REWARD - ((pos_dist) / 5) ** 0.5, POSITIVE_REWARD)
        return reward

    ################################################################################


class DistanceRewardV2(DenseReward):
    """Calculate the dense distance reward."""

    def __init__(self, scale, landing_zone_xyz):
        super().__init__(scale)
        self.landing_zone_xyz = landing_zone_xyz
        self.target_position = self.landing_zone_xyz
        DRONE_MID_Z = 0.01347
        self.target_position[2] = 2 * self.target_position[2] + DRONE_MID_Z

    def _calculateReward(self, state):
        position = state[0:3]
        pos_dist = np.linalg.norm(position - self.target_position)
        reward = min(POSITIVE_REWARD - ((pos_dist) / 5) ** 0.5, POSITIVE_REWARD)
        return reward


class DeltaDistanceReward(DenseReward):
    """Calculate the dense delta distance reward.

    Reward = +1 if we are closer than the last step, -1 if we are further away or the same

    Reward is always 1 if we are closer than 0.1

    """

    def __init__(self, scale, landing_zone_xyz):
        super().__init__(scale)
        self.landing_zone_xyz = landing_zone_xyz
        self._initial_step = True
        self.target_position = self.landing_zone_xyz
        DRONE_MID_Z = 0.01347
        self.target_position[2] = 2 * self.target_position[2] + DRONE_MID_Z

    def _calculateReward(self, state):
        # get the actual state, not the obs
        position = state[0:3]
        pos_dist = np.linalg.norm(position[0:2] - self.target_position[0:2])
        if pos_dist < 0.05:
            return POSITIVE_REWARD
        if not self._initial_step:
            # we want the distance to decrease at each step, thus negative dist_delta
            dist_delta = pos_dist - self.last_pos_dist
            self.last_pos_dist = pos_dist
            if dist_delta < -0.01:
                return POSITIVE_REWARD
            else:
                return NEGATIVE_REWARD
        else:
            self.last_pos_dist = pos_dist
            self._initial_step = False
            return ZERO_REWARD

class DropAltitudeReward(DenseReward):
    """Calculate the dense drop altitude reward

    Reward = +1 if we are lower or the same z height than the last step, -1 otherwise

    """

    def __init__(self, scale, landing_zone_xyz, start_dist):
        super().__init__(scale)
        self.landing_zone_xyz = landing_zone_xyz
        self.target_position = self.landing_zone_xyz
        DRONE_MID_Z = 0.01347
        self.target_position[2] = 2 * self.target_position[2] + DRONE_MID_Z
        self.start_dist = start_dist
        self.reward = False
        self.prev_alt = 1

    def _calculateReward(self, state):
        # get the actual state, not the obs
        position = state[0:3]
        pos_dist = np.linalg.norm(position[0:2] - self.target_position[0:2])
        if pos_dist < self.start_dist:
            if self.reward:
                if position[-1] <= self.prev_alt + 0.01:
                    self.prev_alt = position[-1]
                    return POSITIVE_REWARD
                else:
                    self.prev_alt = position[-1]
                    return NEGATIVE_REWARD
            else:
                self.reward = True
                self.prev_alt = position[-1]
                return POSITIVE_REWARD
        else:
            self.reward = False

        return ZERO_REWARD


class SlowdownReward(DenseReward):
    """Calculate the dense slowdown reward.

    This reward is supposed to help encourage the drone to slow down when approaching the landing zone

    """

    def __init__(self, scale, landing_zone_xyz, slowdown_dist):
        super().__init__(scale)
        self.landing_zone_xyz = landing_zone_xyz
        self.slowdown_dist = slowdown_dist

    def _calculateReward(self, state):
        position = state[0:3]
        # only consider the x and y coordinates, as y can still change to get us closer
        velocity = state[10:12]
        target_position = self.landing_zone_xyz
        target_velocity = np.asarray([0, 0])
        pos_dist = np.linalg.norm(position - target_position)

        vel_dist = np.linalg.norm(velocity - target_velocity)

        if pos_dist < self.slowdown_dist:
            reward = min(POSITIVE_REWARD - (vel_dist / 5) ** 0.5, POSITIVE_REWARD)
        else:
            reward = ZERO_REWARD

        return reward

    ################################################################################


class FieldCoverageReward(DenseReward):
    """Calculate the field coverage reward."""

    def __init__(self, scale, field):
        super().__init__(scale)
        self.field = field

    def _calculateReward(self, state):
        position = state[0:3]
        if self.field.checkIsCovered((position[0], position[1])):
            return 1
        else:
            return -0.01

    ################################################################################


class WaypointReward(DenseReward):
    """Calculate the waypoint reward."""

    def __init__(self, scale, waypoints):
        super().__init__(scale)
        self.waypoints = waypoints
        self.curr_waypoint = 0
        self.max_waypoint = len(waypoints)
        self.rp_bounds = Bounds(-1, 1)
        self.is_landed = False
        self.max_dist_to_waypoint = 0
        self._new_point = True

    def _calculateReward(self, state):
        position = state[0:3]
        target_position = self.waypoints[self.curr_waypoint]
        pos_dist = np.linalg.norm(position - target_position)
        if self._new_point:
            self.max_dist_to_waypoint = pos_dist
            self._new_point = False

        if math.isclose(pos_dist, 0, abs_tol=1e-01):
            if self.max_waypoint > self.curr_waypoint:
                self._new_point = True
                self.curr_waypoint += 1
            return POSITIVE_REWARD
        else:
            return POSITIVE_REWARD * (self.max_dist_to_waypoint - pos_dist)

    ################################################################################


class SparseReward(Reward):
    """Sparse reward for a drone."""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def _calculateReward(self, state):
        return 0

    def calculateReward(self, state):
        if self.scale == 0:
            return 0
        return self._calculateReward(state) * self.scale


class BoundsReward(SparseReward):
    """Checks rewards for out of bounds"""

    def __init__(self, scale, bounds, use_time_scaling=False):
        super().__init__(scale)
        # Defined as [[x_high, y_high, z_high], [x_low, y_low, z_low]]
        self.bounds = bounds
        self.XYZ_IDX = [0, 1, 2]
        self.use_time_scaling = use_time_scaling

    def _calculateReward(self, state):
        position = state[0:3]

        for dim_idx in self.XYZ_IDX:
            if (
                state[dim_idx] > self.bounds[0][dim_idx]
                or state[dim_idx] < self.bounds[1][dim_idx]
            ):
                # self.aviary.completeEpisode = True
                if self.use_time_scaling:
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
                return 10 * POSITIVE_REWARD
            else:
                y_dist = np.linalg.norm(position[2] - (target_position[2] - 0.1))
                return POSITIVE_REWARD * (1 - y_dist)
        else:
            return 0

    ################################################################################


class LandingRewardV2(SparseReward):
    """Calculate the landing reward."""

    def __init__(self, scale, landing_zone_xyz):
        super().__init__(scale)
        self.landing_zone_xyz = landing_zone_xyz
        self.landing_frames = 0
        self.rp_bounds = Bounds(-1, 1)
        DRONE_MID_Z = 0.01347
        self.goal_height = landing_zone_xyz[-1] + DRONE_MID_Z
        self.is_landed = False

    def _calculateReward(self, state):
        position = state[0:3]

        # only consider x and y
        target_position = self.landing_zone_xyz
        pos_dist = np.linalg.norm(position[0:2] - target_position[0:2])

        rp = state[7:9]
        in_rp_bounds = all(within_bounds(self.rp_bounds, field) for field in rp)
        above_goal = pos_dist < 0.099
        if above_goal and in_rp_bounds:
            self.landing_frames += 1
            if math.isclose(position[2], self.goal_height, rel_tol=1e-2):
                self.is_landed = True
                return POSITIVE_REWARD
            else:
                z_dist = position[2] - (self.goal_height)
                max_dist = (
                    1 - self.goal_height
                )  # 1 here assumes that we are bounding above 1 as OOB

                # TODO: Test squaring this quantity
                return POSITIVE_REWARD * ((max_dist - z_dist) / max_dist)
        else:
            return ZERO_REWARD

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


class OrientationReward(SparseReward):
    """Calculate the reward for staying level based off roll and pitch."""

    def __init__(self, scale):
        super().__init__(scale)
        # 1 radian is about 57 degrees
        self.bounds = Bounds(min=-1, max=1)

    def _calculateReward(self, state):
        rp = state[7:9]
        in_bounds = all(within_bounds(self.bounds, field) for field in rp)
        if in_bounds:
            return POSITIVE_REWARD
        else:
            return NEGATIVE_REWARD

    ################################################################################
