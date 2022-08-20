import math
import os
import numpy as np
from gym_pybullet_drones.envs.single_agent_rl.rewards.utils import Bounds

from gym_pybullet_drones.envs.single_agent_rl.terminations import Terminations


class FieldCoverageTerm(Terminations):
    """Checks terminals for field coverage"""

    def __init__(self, interval_sec, landing_zone_xyz):
        super().__init__()
        # Defined as [[x_high, y_high, z_high], [x_low, y_low, z_low]]
        self.SIM_FREQ = 240
        self.interval = interval_sec * self.SIM_FREQ
        self.curr_fuel = self.interval
        self.landing_zone_xyz = landing_zone_xyz
        self.target_position = self.landing_zone_xyz
        DRONE_MID_Z = 0.01347
        self.target_position[2] = 2 * self.target_position[2] + DRONE_MID_Z

    def _calculateTerm(self, state):
        self.curr_fuel -= 1
        position = state[0:3]
        pos_dist = np.linalg.norm(position - self.target_position)
        if pos_dist < 0.12:
            self.curr_fuel = self.interval
        return self.curr_fuel < 0
