import os
import numpy as np
import math

from gym import spaces
import pybullet as p


class Field:
    """Landing zone for a drone"""

    def __init__(self, xyz, wlh, physics, color=[0, 255, 0]):
        """
        Parameters
        ----------
        xyz : np.array
            x, y, z position
        wlh : np.array
            width, length, height of the field
        """
        self.xyz = xyz
        self.wlh = wlh
        self.CLIENT = physics
        self.color = color
        self.MASS = 100000
        self.covered_area = np.zeros((int(wlh[0]), int(wlh[1])))

    def _addObstacles(self):
        """Add obstacles to the environment.

        Add the landing zone box to the environment."""
        colBoxId = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=self.wlh / 2, physicsClientId=self.CLIENT
        )
        p.createMultiBody(
            self.MASS, colBoxId, -1, self.xyz, [0, 0, 0, 1], physicsClientId=self.CLIENT
        )

        # self.field = p.loadURDF(
        #     "cube.urdf",
        #     self.xyz,
        #     physicsClientId=self.CLIENT,
        #     useFixedBase=True,
        #     globalScaling=0.2,
        # )
        # p.changeVisualShape(self.landing_zone, linkIndex, rgbaColor=self.color)

    def checkIsCovered(self, new_spot):
        """Check if the new spot is covered by the field.

        Parameters
        ----------
        new_spot : np.array
            x, y, z position

        Returns
        -------
        bool
            True if the new spot is covered by the field.

        """
        x = int(new_spot[0] / self.wlh[0])
        y = int(new_spot[1] / self.wlh[1])
        if self.covered_area[x, y] == 1:
            self.updateCoveredArea(new_spot)
            return True
        else:
            return False

    def updateCoveredArea(self, new_spot):
        """Update the covered area of the field.

        Parameters
        ----------
        new_spot : np.array
            x, y, z position

        """
        x = int(new_spot[0] / self.wlh[0])
        y = int(new_spot[1] / self.wlh[1])
        self.covered_area[x, y] = 1

    def getTotalCoveredArea(self):
        """Get the total covered area of the field.

        Returns
        -------
        float
            Total covered area of the field.

        """
        return np.sum(self.covered_area)

    def isAllCovered(self):
        """Check if the field is covered.

        Returns
        -------
        bool
            True if the field is covered.

        """
        return np.sum(self.covered_area) == self.covered_area.size

