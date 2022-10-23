from typing import Dict, Tuple
import os
import numpy as np
import math

from gym import spaces
import pybullet as p
import pybullet_data as pd


class FurnitureRoom:
    """Landing zone for a drone"""

    def __init__(self, xyz, physics, version):
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
        self.MASS = 100000
        self.landing_zone = None
        self.version = version

    def reset(self):
        pass

    def version_1(self):
        self.landing_zone = p.loadURDF(
            "table_square/table_square.urdf", [6.5, 0, 0], physicsClientId=self.CLIENT
        )

        p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/../../../assets/wider_room.urdf",
            [7, -1.5, 0],
            physicsClientId=self.CLIENT,
            globalScaling=1,
        )
        p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/../../../assets/dresser.urdf",
            [2.1, 1.0, 0.1],
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.CLIENT,
            globalScaling=0.15,
        )
        p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/../../../assets/dresser.urdf",
            [2.1, 0.0, 0.1],
            baseOrientation=[0, 0, -1, 1],
            physicsClientId=self.CLIENT,
            globalScaling=0.15,
        )
        p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/../../../assets/tv_stand.urdf",
            [3, 0.5, 0],
            baseOrientation=[0, 0, 1, 1],
            physicsClientId=self.CLIENT,
            globalScaling=1,
        )
        p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/../../../assets/tv_stand.urdf",
            [1, 0.5, 0],
            baseOrientation=[0, 0, 1, 1],
            physicsClientId=self.CLIENT,
            globalScaling=1,
        )

        p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/../../../assets/lamp/lamp.urdf",
            [2.5, -0.7, 0.0],
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.CLIENT,
            globalScaling=0.8,
        )
        p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/../../../assets/lamp/lamp.urdf",
            [4.5, -0.7, 0.0],
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.CLIENT,
            globalScaling=0.8,
        )
        p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/../../../assets/lamp/lamp.urdf",
            [0.5, -0.7, 0.0],
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.CLIENT,
            globalScaling=0.8,
        )
        p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/../../../assets/lamp/lamp.urdf",
            [5.5, 0.0, 0.0],
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.CLIENT,
            globalScaling=0.8,
        )

    def version_0(self):
        self.landing_zone = p.loadURDF(
            "table_square/table_square.urdf", [6.5, 0, 0], physicsClientId=self.CLIENT,
        )

        p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/../../../assets/wider_room.urdf",
            [7, -1.5, 0],
            physicsClientId=self.CLIENT,
            globalScaling=1,
        )
        p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/../../../assets/chairs/table.urdf",
            [5.3, 0.5, 0],
            baseOrientation=[0, 0, 1, 1],
            physicsClientId=self.CLIENT,
            globalScaling=1.2,
        )
        p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/../../../assets/chairs/table.urdf",
            [1, 0.0, 0],
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.CLIENT,
            globalScaling=0.7,
        )
        p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/../../../assets/tv_stand.urdf",
            [3, 0.5, 0],
            physicsClientId=self.CLIENT,
            globalScaling=1,
        )
        p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/../../../assets/chairs/couch.urdf",
            [0.7, 0.9, 0],
            physicsClientId=self.CLIENT,
            globalScaling=0.7,
        )
        p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/../../../assets/chairs/chair.urdf",
            [2, -0.7, 0],
            baseOrientation=[0, 0, -1, 1],
            physicsClientId=self.CLIENT,
            globalScaling=1,
        )

        p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/../../../assets/kit_chair.urdf",
            [5, -0.7, 0],
            baseOrientation=[0, 0, 2, 1],
            physicsClientId=self.CLIENT,
            globalScaling=0.01,
        )
        p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/../../../assets/kit_chair.urdf",
            [4.5, 0.4, 0],
            baseOrientation=[0, 0, 1, 1],
            physicsClientId=self.CLIENT,
            globalScaling=0.01,
        )
        p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/../../../assets/dresser.urdf",
            [3.1, 1.0, 0.1],
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.CLIENT,
            globalScaling=0.15,
        )
        p.loadURDF(
            os.path.dirname(os.path.abspath(__file__))
            + "/../../../assets/lamp/lamp.urdf",
            [2.5, -0.7, 0.0],
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.CLIENT,
            globalScaling=0.8,
        )

    def _addObstacles(self):
        """Add obstacles"""
        if self.version == 0:
            self.version_0()
        elif self.version == 1:
            self.version_1()

