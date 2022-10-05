import math
import numpy as np

from gym_pybullet_drones.envs.single_agent_rl.rewards.utils import (
    Bounds,
    within_bounds,
    POSITIVE_REWARD,
    NEGATIVE_REWARD,
    ZERO_REWARD,
)
from gym_pybullet_drones.envs.single_agent_rl.rewards.Reward import (
    DenseReward,
    SparseReward,
)


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

    def reset(self):
        self.is_landed = False
        self.max_dist_to_waypoint = 0
        self._new_point = True
        self.curr_waypoint = 0

    def _calculateReward(self, state, drone_id=0):
        position = state[0:3]
        target_position = self.waypoints[self.curr_waypoint]
        pos_dist = np.linalg.norm(position - target_position)
        if self._new_point:
            self.max_dist_to_waypoint = pos_dist
            self._new_point = False

        if math.isclose(pos_dist, 0, abs_tol=0.707):
            if self.max_waypoint > self.curr_waypoint + 1:
                self._new_point = True
                self.curr_waypoint += 1
            return POSITIVE_REWARD
        else:
            return POSITIVE_REWARD * (self.max_dist_to_waypoint - pos_dist)

    ################################################################################
