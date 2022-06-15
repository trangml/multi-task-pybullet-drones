import os
import numpy as np
import math
import abc
import ruamel.yaml

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

    def __init__(self, scale):
        self.scale = scale

    def _calculateReward(self, state):
        return 0

    def calculateReward(self, state):
        return self._calculateReward(state) * self.scale


class DistanceReward(DenseReward):
    """Calculate the dense distance reward."""

    def __init__(self, scale, landing_zone_xyz):
        super().__init__(scale)
        self.landing_zone_xyz = landing_zone_xyz

    @classmethod
    def from_yaml(cls, constructor, node):
        """Create a DistanceReward from YAML.

        Parameters
        ----------
        slx : Constructor
            YAML constructor.
        node : yaml.nodes.Node
            YAML node.

        Returns
        -------
        DistanceReward
            DistanceReward instance.

        """
        args = constructor.construct_yaml_map(node)
        return DistanceReward(**args)

    def _calculateReward(self, state):
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


class DeltaDistanceReward(DenseReward):
    """Calculate the dense delta distance reward.

    Reward = +1 if we are closer than the last step, -1 if we are further away or the same

    Reward is always 1 if we are closer than 0.1

    """

    def __init__(self, scale, landing_zone_xyz):
        super().__init__(scale)
        self.landing_zone_xyz = landing_zone_xyz
        self._initial_step = True

    def _calculateReward(self, state):
        # get the actual state, not the obs
        position = state[0:3]
        target_position = self.landing_zone_xyz
        pos_dist = np.linalg.norm(position[0:2] - target_position[0:2])
        if pos_dist < 0.1:
            return POSITIVE_REWARD
        if not self._initial_step:
            dist_delta = pos_dist - self.last_pos_dist
            self.last_pos_dist = pos_dist
            if dist_delta < 0:
                return POSITIVE_REWARD
            else:
                return NEGATIVE_REWARD
        else:
            self.last_pos_dist = pos_dist
            self._initial_step = False
            return 0


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
            reward = 1 - (vel_dist / 5) ** 0.5
        else:
            reward = 0

        return reward


class FieldCoverageReward(DenseReward):
    """Calculate the field coverage reward."""

    def __init__(self, scale, field):
        super().__init__(scale)
        self.field = field

    def _calculateReward(self):
        position = state[0:3]
        if self.field.checkIsCovered((position[0], position[1])):
            return 1
        else:
            return -0.01

    ################################################################################


def test_yaml_load():
    """Test the YAML load of the DistanceReward."""
    yaml_str = """
    !DistanceReward
    scale: 0.1
    landing_zone_xyz: [0, 0, 0]
    """
    yaml = ruamel.yaml.YAML(typ="safe")
    yaml.register_class(Item)
    yaml.register_class(Message)
    yaml_map = yaml.load(yaml_str)
    # with open('input.yaml') as fp:
    #     data = yaml.load(fp)


if __name__ == "__main__":
    test_yaml_load()
