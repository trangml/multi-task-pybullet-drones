from typing import List, Optional
import os
import numpy as np
import gym_pybullet_drones.envs.single_agent_rl.rewards as rewards
from gym_pybullet_drones.envs.single_agent_rl.rewards.cross_obstacles.EnterAreaReward import (
    EnterAreaReward,
)
from gym_pybullet_drones.envs.single_agent_rl.rewards.cross_obstacles.IncreaseXReward import (
    IncreaseXReward,
)
import gym_pybullet_drones.envs.single_agent_rl.terminations as terminations
import pybullet as p
from gym_pybullet_drones.envs.BaseAviary import BaseAviary, DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    BaseSingleAgentAviary,
    ObservationType,
)
from gym_pybullet_drones.envs.single_agent_rl.rewards import getRewardDict
from gym_pybullet_drones.envs.single_agent_rl.terminations import getTermDict
from gym_pybullet_drones.envs.single_agent_rl.obstacles.ObstacleRoom import ObstacleRoom
from gym_pybullet_drones.envs.single_agent_rl.obstacles.LongObstacleRoom import (
    LongObstacleRoom,
)
from gym_pybullet_drones.envs.single_agent_rl.obstacles.IncrementalObstacleRoom import (
    IncrementalObstacleRoom,
)
from gym_pybullet_drones.envs.single_agent_rl.terminations.CollisionTerm import (
    CollisionTerm,
)
from gym_pybullet_drones.envs.single_agent_rl.terminations.Terminations import (
    BoundsTerm,
    OrientationTerm,
)


class LongCrossObstaclesAviary(BaseSingleAgentAviary):
    """Single agent RL problem: cross a room with obstacles"""

    ################################################################################

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        freq: int = 240,
        aggregate_phy_steps: int = 1,
        gui=False,
        record=False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.RPM,
        difficulty: int = 0,
        reward_components: List = [],
        term_components: List = [],
        bounds: List = [[5, 1, 1], [-1, -1, 0.1]],
    ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.bounds = bounds

        self.obstacles = []
        self.reward_components = []
        for reward_name in reward_components:
            if reward_components[reward_name]["scale"] != 0:
                r_class = getattr(rewards, reward_name)
                self.reward_components.append(r_class(**reward_components[reward_name]))
        if len(self.reward_components) == 0:
            self.reward_components.append(
                EnterAreaReward(scale=10, area=[[4, 5], [-1, 1]])
            )
            self.reward_components.append(IncreaseXReward(scale=0.1))

        self.term_components = []
        for term_name in term_components:
            t_class = getattr(terminations, term_name)
            args = term_components[term_name]
            if args is not None:
                self.term_components.append(t_class(**args))
            else:
                self.term_components.append(t_class())
        if len(self.term_components) == 0:
            self.term_components.append(BoundsTerm(bounds=self.bounds))
            self.term_components.append(OrientationTerm())

        super().__init__(
            drone_model=drone_model,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            freq=freq,
            aggregate_phy_steps=aggregate_phy_steps,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )

        # override base aviary episode length
        self.EPISODE_LEN_SEC = 10
        self.difficulty = difficulty
        self.obstacles.append(
            LongObstacleRoom(
                xyz=[0, 0, 0], physics=self.CLIENT, difficulty=self.difficulty
            )
        )
        self.reward_dict = getRewardDict(self.reward_components)
        self.term_components.append(
            CollisionTerm([[11, 12], [-1, 1]], self.DRONE_IDS[0], self.CLIENT)
        )
        self.term_dict = getTermDict(self.term_components)
        self.cum_reward_dict = getRewardDict(self.reward_components)
        self.truncated = False
        self.done = False

    ################################################################################

    def reset(self):
        self.truncated = False
        self.done = False
        for rwd in self.reward_components:
            rwd.reset()
        for term in self.term_components:
            term.reset()
        return super().reset()

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Extends the superclass method and add the obstacles to the environment.

        """
        super()._addObstacles()
        for obstacle in self.obstacles:
            obstacle._addObstacles()

    ################################################################################
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        cum_reward = 0
        state = self._getDroneStateVector(0)
        for reward_component, r_dict in zip(self.reward_components, self.reward_dict):
            r = reward_component.calculateReward(state)
            self.reward_dict[r_dict] = r
            self.cum_reward_dict[r_dict] += r
            cum_reward += r

        self.reward_dict["total"] = cum_reward
        self.cum_reward_dict["total"] += cum_reward
        return cum_reward

    ################################################################################

    def _computeDone(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC:
            # TODO: add to info if we timeout
            self.truncated = True
            self.done = True
        else:
            state = self._getDroneStateVector(0)
            done = False
            for term_component, t_dict in zip(self.term_components, self.term_dict):
                t = term_component.calculateTerm(state)
                self.term_dict[t_dict] = t
                done = done or t
            self.done = done

        return self.done

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        info = {}
        if self.truncated:
            info["TimeLimit.truncated"] = True
        if self.done:
            info["terminal_observation"] = True
        return info

    ################################################################################

    def _clipAndNormalizeState(self, state):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(
                state,
                clipped_pos_xy,
                clipped_pos_z,
                clipped_rp,
                clipped_vel_xy,
                clipped_vel_z,
            )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = (
            state[13:16] / np.linalg.norm(state[13:16])
            if np.linalg.norm(state[13:16]) != 0
            else state[13:16]
        )

        norm_and_clipped = np.hstack(
            [
                normalized_pos_xy,
                normalized_pos_z,
                state[3:7],
                normalized_rp,
                normalized_y,
                normalized_vel_xy,
                normalized_vel_z,
                normalized_ang_vel,
                state[16:20],
            ]
        ).reshape(20,)

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(
        self,
        state,
        clipped_pos_xy,
        clipped_pos_z,
        clipped_rp,
        clipped_vel_xy,
        clipped_vel_z,
    ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.

        """
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in NavigateMazeAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(
                    state[0], state[1]
                ),
            )
        if not (clipped_pos_z == np.array(state[2])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in NavigateMazeAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(
                    state[2]
                ),
            )
        if not (clipped_rp == np.array(state[7:9])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in NavigateMazeAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(
                    state[7], state[8]
                ),
            )
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in NavigateMazeAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(
                    state[10], state[11]
                ),
            )
        if not (clipped_vel_z == np.array(state[12])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in NavigateMazeAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(
                    state[12]
                ),
            )

