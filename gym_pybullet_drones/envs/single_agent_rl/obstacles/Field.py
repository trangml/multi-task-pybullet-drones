from typing import Dict, Tuple
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
        self.grid_dim = 1
        # grid is represented as (x, y)
        self.covered_area = np.zeros(
            (int(self.wlh[0] / self.grid_dim), int(self.wlh[1] / self.grid_dim))
        )
        self.total_covered_area = 0
        self.total_possible_area = self.covered_area.size

    def reset(self):
        self.covered_area = np.zeros(
            (int(self.wlh[0] / self.grid_dim), int(self.wlh[1] / self.grid_dim))
        )
        self.total_covered_area = 0

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

    def coordinates_to_grid(self, new_spot) -> Tuple[int, int]:
        """
        Convert simulation true coordinates to the field grid coordinates.

        Parameters
        ----------
        new_spot : _type_
            the sim true coordinates

        Returns
        -------
        Tuple[int, int]
            x, y locations for the 2d grid
        """
        x = new_spot[0]
        y = new_spot[1]
        x = x - self.xyz[0]
        y = y - self.xyz[1]
        x = int(x / self.grid_dim + (self.wlh[0] / 2) / self.grid_dim)
        y = int(y / self.grid_dim + (self.wlh[1] / 2) / self.grid_dim)
        return (x, y)

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
        # convert to the covered area array coordinates
        x, y = self.coordinates_to_grid(new_spot)

        if (
            x >= 0
            and y >= 0
            and x < self.covered_area.shape[0]
            and y < self.covered_area.shape[1]
        ):
            if self.covered_area[x, y] == 0:
                self.updateCoveredArea(x, y)
                return True
        return False

    def updateCoveredArea(self, x, y):
        """Update the covered area of the field.

        Parameters
        ----------
        new_spot : np.array
            x, y, z position

        """
        self.covered_area[x, y] = 1
        self.total_covered_area += 1

    def getTotalCoveredArea(self):
        """Get the total covered area of the field.

        Returns
        -------
        float
            Total covered area of the field.

        """
        return self.total_covered_area

    def isAllCovered(self):
        """Check if the field is covered.

        Returns
        -------
        bool
            True if the field is covered.

        """
        return self.total_covered_area == self.total_possible_area
