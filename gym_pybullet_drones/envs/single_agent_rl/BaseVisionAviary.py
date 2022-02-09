import os
from datetime import datetime
from enum import Enum
import numpy as np
from gym import spaces
import pybullet as p
import pybullet_data

from gym_pybullet_drones.utils.utils import nnlsRPM
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl

from gym_pybullet_drones.envs.BaseAviary import (
    DroneModel,
    Physics,
    ImageType,
    BaseAviary,
)


class ActionType(Enum):
    """Action type enumeration class."""

    RPM = "rpm"  # RPMS
    DYN = "dyn"  # Desired thrust and torques
    PID = "pid"  # PID control
    VEL = "vel"  # Velocity input (using PID control)
    TUN = "tun"  # Tune the coefficients of a PID controller
    ONE_D_RPM = "one_d_rpm"  # 1D (identical input to all motors) with RPMs
    ONE_D_DYN = "one_d_dyn"  # 1D (identical input to all motors) with desired thrust and torques
    ONE_D_PID = "one_d_pid"  # 1D (identical input to all motors) with PID control


################################################################################


class BaseVisionAviary(BaseAviary):
    """Multi-drone environment class for control applications using vision."""

    ################################################################################

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 1,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        freq: int = 240,
        aggregate_phy_steps: int = 1,
        gui=False,
        record=False,
        obstacles=False,
        user_debug_gui=True,
        # obs: ObservationType=ObservationType.KIN,
        act: ActionType = ActionType.RPM,
    ):
        """Initialization of an aviary environment for control applications using vision.

        Attribute `vision_attributes` is automatically set to True when calling
        the superclass `__init__()` method.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
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
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.

        """
        self.ACT_TYPE = act
        self.EPISODE_LEN_SEC = 5

        self.completeEpisode = False
        self.min_dist = 100
        #### Create integrated controllers #########################
        if act in [
            ActionType.PID,
            ActionType.VEL,
            ActionType.TUN,
            ActionType.ONE_D_PID,
        ]:
            os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
                if act == ActionType.TUN:
                    self.TUNED_P_POS = np.array([0.4, 0.4, 1.25])
                    self.TUNED_I_POS = np.array([0.05, 0.05, 0.05])
                    self.TUNED_D_POS = np.array([0.2, 0.2, 0.5])
                    self.TUNED_P_ATT = np.array([70000.0, 70000.0, 60000.0])
                    self.TUNED_I_ATT = np.array([0.0, 0.0, 500.0])
                    self.TUNED_D_ATT = np.array([20000.0, 20000.0, 12000.0])
            elif drone_model == DroneModel.HB:
                self.ctrl = SimplePIDControl(drone_model=DroneModel.HB)
                if act == ActionType.TUN:
                    self.TUNED_P_POS = np.array([0.1, 0.1, 0.2])
                    self.TUNED_I_POS = np.array([0.0001, 0.0001, 0.0001])
                    self.TUNED_D_POS = np.array([0.3, 0.3, 0.4])
                    self.TUNED_P_ATT = np.array([0.3, 0.3, 0.05])
                    self.TUNED_I_ATT = np.array([0.0001, 0.0001, 0.0001])
                    self.TUNED_D_ATT = np.array([0.3, 0.3, 0.5])
            else:
                print(
                    "[ERROR] in BaseSingleAgentAviary.__init()__, no controller is available for the specified drone_model"
                )
        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            freq=freq,
            aggregate_phy_steps=aggregate_phy_steps,
            gui=gui,
            record=record,
            obstacles=True,  # Add obstacles for RGB observations and/or FlyThruGate
            user_debug_gui=False,  # Remove of RPM sliders from all single agent learning aviaries
            vision_attributes=True,
        )
        ### set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000 / 3600)
        #### Try _trajectoryTrackingRPMs exists IFF ActionType.T U N #
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000 / 3600)
        #### Try _trajectoryTrackingRPMs exists IFF ActionType.TUN #
        if act == ActionType.TUN and not (
            hasattr(self.__class__, "_trajectoryTrackingRPMs")
            and callable(getattr(self.__class__, "_trajectoryTrackingRPMs"))
        ):
            print(
                "[ERROR] in BaseSingleAgentAviary.__init__(), ActionType.TUN requires an implementation of _trajectoryTrackingRPMs in the instantiated subclass"
            )
            exit()

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        pass

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        dict[str, ndarray]
            A Dict of Box(4,) with NUM_DRONES entries,
            indexed by drone Id in string format.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([0.0, 0.0, 0.0, 0.0])
        act_upper_bound = np.array(
            [self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM]
        )
        return spaces.Dict(
            {
                str(i): spaces.Box(
                    low=act_lower_bound, high=act_upper_bound, dtype=np.float32
                )
                for i in range(self.NUM_DRONES)
            }
        )

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES), Box(H,W,4), Box(H,W), Box(H,W)}.

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array(
            [
                -np.inf,
                -np.inf,
                0.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -np.pi,
                -np.pi,
                -np.pi,
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        obs_upper_bound = np.array(
            [
                np.inf,
                np.inf,
                np.inf,
                1.0,
                1.0,
                1.0,
                1.0,
                np.pi,
                np.pi,
                np.pi,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                self.MAX_RPM,
                self.MAX_RPM,
                self.MAX_RPM,
                self.MAX_RPM,
            ]
        )
        return spaces.Dict(
            {
                str(i): spaces.Dict(
                    {
                        "state": spaces.Box(
                            low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32
                        ),
                        "neighbors": spaces.MultiBinary(self.NUM_DRONES),
                        "rgb": spaces.Box(
                            low=0,
                            high=255,
                            shape=(self.IMG_RES[1], self.IMG_RES[0], 4),
                            dtype=np.uint8,
                        ),
                        "dep": spaces.Box(
                            low=0.01,
                            high=1000.0,
                            shape=(self.IMG_RES[1], self.IMG_RES[0]),
                            dtype=np.float32,
                        ),
                        "seg": spaces.Box(
                            low=0,
                            high=100,
                            shape=(self.IMG_RES[1], self.IMG_RES[0]),
                            dtype=np.int,
                        ),
                    }
                )
                for i in range(self.NUM_DRONES)
            }
        )

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of key "state", see the implementation of `_getDroneStateVector()`,
        the value of key "neighbors" is the drone's own row of the adjacency matrix,
        "rgb", "dep", and "seg" are matrices containing POV camera captures.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES), Box(H,W,4), Box(H,W), Box(H,W)}.

        """
        adjacency_mat = self._getAdjacencyMatrix()
        obs = {}
        for i in range(self.NUM_DRONES):
            if self.step_counter % self.IMG_CAPTURE_FREQ == 0:
                self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
                #### Printing observation to PNG frames example ############
                if self.RECORD:
                    self._exportImage(
                        img_type=ImageType.RGB,  # ImageType.BW, ImageType.DEP, ImageType.SEG
                        img_input=self.rgb[i],
                        path=self.ONBOARD_IMG_PATH + "drone_" + str(i),
                        frame_num=int(self.step_counter / self.IMG_CAPTURE_FREQ),
                    )
            obs[str(i)] = {
                "state": self._getDroneStateVector(i),
                "neighbors": adjacency_mat[i, :],
                "rgb": self.rgb[i],
                "dep": self.dep[i],
                "seg": self.seg[i],
            }
        return obs

    ################################################################################

    def _preprocessAction(self, action):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : dict[str, ndarray]
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        clipped_action = np.zeros((self.NUM_DRONES, 4))
        for k, v in action.items():
            clipped_action[int(k), :] = np.clip(np.array(v), 0, self.MAX_RPM)
        return clipped_action

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        return -1

    ################################################################################

    def _computeDone(self):
        """Computes the current done value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {
            "answer": 42
        }  #### Calculated by the Deep Thought supercomputer in 7.5M years
