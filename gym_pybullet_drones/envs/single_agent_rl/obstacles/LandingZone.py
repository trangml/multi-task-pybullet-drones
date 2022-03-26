import os
import numpy as np
import math

from gym import spaces
import pybullet as p


class LandingZone:
    """Landing zone for a drone"""

    def __init__(self, xyz, wlh, physics, color=[0, 0, 0, 1]):
        """
        Parameters
        ----------
        xyz : [type]
            [description]
        wlh : [type]
            [description]
        color : [type]
            [description]
        """
        self.xyz = xyz
        self.wlh = wlh
        self.color = color
        self.CLIENT = physics
        self.MASS = 10000

    def _addObstacles(self):
        """Add obstacles to the environment.

        Add the landing zone box to the environment."""
        colBoxId = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=self.wlh / 2, physicsClientId=self.CLIENT
        )
        box = p.createMultiBody(
            self.MASS, colBoxId, -1, self.xyz, [0, 0, 0, 1], physicsClientId=self.CLIENT
        )

        p.changeVisualShape(box ,colBoxId,rgbaColor=[255, 0,0,1])

        # self.landing_zone = p.loadURDF(
        #     "cube.urdf",
        #     self.xyz,
        #     physicsClientId=self.CLIENT,
        #     useFixedBase=True,
        #     globalScaling=0.2,
        # )
        # p.changeVisualShape(self.landing_zone, linkIndex, rgbaColor=self.color)
