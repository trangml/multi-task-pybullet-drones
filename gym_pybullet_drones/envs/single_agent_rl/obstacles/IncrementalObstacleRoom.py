from typing import Dict, Tuple
import os
import numpy as np
import math

from gym import spaces
import pybullet as p


class IncrementalObstacleRoom:
    """Landing zone for a drone"""

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

        Add the landing zone box to the environment."""
        p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/../../../assets/room_w_pad.urdf",
            [5 + self.xyz[0], -1 + self.xyz[1], 0 + self.xyz[2]],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.CLIENT,
            globalScaling=1,
            useFixedBase=True,
        )

        if self.difficulty == 0:
            # difficulty of 0 is no obstacles
            pass
        elif self.difficulty == 1:
            # difficulty of 1 is placing a column right in front of the drone
            box_q = p.getQuaternionFromEuler([0, 1.57057, 0])
            box_pos = [2 + self.xyz[0], self.xyz[1], 0.5 + self.xyz[2]]
            p.loadURDF(
                "block.urdf",
                box_pos,
                box_q,
                physicsClientId=self.CLIENT,
                useFixedBase=True,
                globalScaling=10,
            )

        elif self.difficulty == 2:
            # difficulty of 2 is many rows of obstacles
            pass

        elif self.difficulty == 3:
            # difficulty of 3 is many rows of staggered obstacles
            pass

