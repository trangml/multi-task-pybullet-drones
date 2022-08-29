from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary
from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import LeaderFollowerAviary
from gym_pybullet_drones.envs.multi_agent_rl.MeetupAviary import MeetupAviary
from gym_pybullet_drones.envs.multi_agent_rl.MultiCrossObstaclesAviary import MultiCrossObstaclesAviary

ENV_MAP = {
    "flock-aviary-v0": FlockAviary,
    "leaderfollower-aviary-v0": LeaderFollowerAviary,
    "meetup-aviary-v0": MeetupAviary,
    "multicrossobs-aviary-v0": MultiCrossObstaclesAviary,
}


def map_name_to_multi_env(env_name: str) -> BaseMultiagentAviary:
    """
    Maps the name of the environment to the corresponding Env class

    Parameters
    ----------
    env_name : str
       name of the environment/task

    Returns
    -------
    BaseMultiagentAviary
        Callable environment
    """

    return ENV_MAP[env_name]
