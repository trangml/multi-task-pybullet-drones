from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    BaseSingleAgentAviary,
)
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary
from gym_pybullet_drones.envs.single_agent_rl.TuneAviary import TuneAviary
from gym_pybullet_drones.envs.single_agent_rl.NavigateMazeAviary import (
    NavigateMazeAviary,
)
from gym_pybullet_drones.envs.single_agent_rl.NavigateObstaclesAviary import (
    NavigateObstaclesAviary,
)
from gym_pybullet_drones.envs.single_agent_rl.NavigateLandAviary import (
    NavigateLandAviary,
)
from gym_pybullet_drones.envs.single_agent_rl.FieldCoverageAviary import (
    FieldCoverageAviary,
)

from gym_pybullet_drones.envs.single_agent_rl.BaseVisionAviary import BaseVisionAviary
from gym_pybullet_drones.envs.single_agent_rl.LandVisionAviary import LandVisionAviary

ENV_MAP = {
    "hover-aviary-v0": HoverAviary,
    "maze-aviary-v0": NavigateMazeAviary,
    "obstacles-aviary-v0": NavigateObstaclesAviary,
    "land-aviary-v0": NavigateLandAviary,
    "field-aviary-v0": FieldCoverageAviary,
    "land-vision-aviary-v0": LandVisionAviary,
    "takeoff-aviary-v0": TakeoffAviary,
    "flythrugate-aviary-v0": FlyThruGateAviary,
    "tune-aviary-v0": TuneAviary,
}


def map_name_to_env(env_name: str) -> BaseSingleAgentAviary:
    """
    Maps the name of the environment to the corresponding Env class

    Parameters
    ----------
    env_name : str
       name of the environment/task

    Returns
    -------
    BaseSingleAgentAviary
        Callable environment
    """

    return ENV_MAP[env_name]

