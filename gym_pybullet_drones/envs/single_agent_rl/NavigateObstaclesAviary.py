import os
import numpy as np
from gym import spaces
import pybullet as p

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
    BaseSingleAgentAviary,
)


class NavigateObstaclesAviary(BaseSingleAgentAviary):
    """Single agent RL problem: navigate through a obstacles."""

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

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Extends the superclass method and add the obstacles to the environment.

        """
        super()._addObstacles()
        boxStartOrientation = p.getQuaternionFromEuler([0, 1.57057, 0])
        for i in range(1, 4):
            for j in range(1, 4):
                boxStartPos = [i, j, 0.5]
                p.loadURDF(
                    "block.urdf",
                    boxStartPos,
                    boxStartOrientation,
                    physicsClientId=self.CLIENT,
                    useFixedBase=True,
                    globalScaling=10,
                )
        landing_zone = p.loadURDF(
            "cube.urdf",
            [3.5, 3.5, 0],
            physicsClientId=self.CLIENT,
            useFixedBase=True,
            globalScaling=0.2,
        )

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        target_dest = [3.5, 3.5, 0.125]
        dist = np.linalg.norm(np.asarray(state[0:3]) - np.asarray(target_dest))
        max_dist = 5
        if dist < 0.1:
            return 2240
        elif dist < 1:
            return 1
        # Penalize if out of bounds
        elif dist > max_dist:
            return -240
        elif state[0] > 4 or state[1] > 4 or state[2] > 1:
            return -240
        elif state[0] < -0.1 or state[1] < -0.1 or state[2] < 0.1:
            return -240
        # # penalize if not landed by the end of the episode
        elif self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC:
            return -240

        return 1 / dist

    ################################################################################

    def _computeDone(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)

        target_dest = [3.5, 3.5, 0.125]
        dist = np.linalg.norm(np.asarray(state[0:3]) - np.asarray(target_dest))
        max_dist = 5
        if dist < 0.1:
            return True
        elif dist > max_dist:
            return True
        elif state[0] > 4 or state[1] > 4 or state[2] > 1:
            return True
        elif state[0] < -0.1 or state[1] < -0.1 or state[2] < 0.1:
            return True
        elif self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {
            "answer": 42
        }  #### Calculated by the Deep Thought supercomputer in 7.5M years

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
        ).reshape(
            20,
        )

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
