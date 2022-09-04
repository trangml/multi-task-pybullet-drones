from typing import Dict, Tuple
import os
import numpy as np
import math

from gym import spaces
import pybullet as p


class LongObstacleRoom:
    """Obstacle Room"""

    def __init__(self, xyz, physics, difficulty, color=[0, 255, 0]):
        """
        Parameters
        ----------
        xyz : np.array
            x, y, z position
        wlh : np.array
            width, length, height of the field
        """
        self.xyz = xyz
        self.CLIENT = physics
        self.color = color
        self.difficulty = difficulty
        self.MASS = 100000

    def reset(self):
        pass

    def _addObstacles(self):
        """Add obstacles to the environment.

        """

        if self.difficulty == 0:
            # difficulty of 0 is no obstacles
            p.loadURDF(
                os.path.dirname(os.path.abspath(__file__))
                + "/../../../assets/long_room.urdf",
                [11 + self.xyz[0], -1 + self.xyz[1], 0 + self.xyz[2]],
                p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.CLIENT,
                globalScaling=1,
                useFixedBase=True,
            )
        elif self.difficulty == 1:
            p.loadURDF(
                os.path.dirname(os.path.abspath(__file__))
                + "/../../../assets/long_room_half_wall.urdf",
                [11 + self.xyz[0], -1 + self.xyz[1], 0 + self.xyz[2]],
                p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.CLIENT,
                globalScaling=1,
                useFixedBase=True,
            )

        elif self.difficulty == 2:
            # difficulty of 2 is many rows of obstacles
            p.loadURDF(
                os.path.dirname(os.path.abspath(__file__))
                + "/../../../assets/long_room_2_half_wall.urdf",
                [11 + self.xyz[0], -1 + self.xyz[1], 0 + self.xyz[2]],
                p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.CLIENT,
                globalScaling=1,
                useFixedBase=True,
            )

        elif self.difficulty == 3:
            # difficulty of 3 is many rows of staggered obstacles
            p.loadURDF(
                os.path.dirname(os.path.abspath(__file__))
                + "/../../../assets/long_room_2_window_wall.urdf",
                [11 + self.xyz[0], -1 + self.xyz[1], 0 + self.xyz[2]],
                p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.CLIENT,
                globalScaling=1,
                useFixedBase=True,
            )

