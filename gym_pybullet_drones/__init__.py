from gym.envs.registration import register

register(
    id="ctrl-aviary-v0", entry_point="gym_pybullet_drones.envs:CtrlAviary",
)

register(
    id="dyn-aviary-v0", entry_point="gym_pybullet_drones.envs:DynAviary",
)

register(
    id="velocity-aviary-v0", entry_point="gym_pybullet_drones.envs:VelocityAviary",
)

register(
    id="vision-aviary-v0", entry_point="gym_pybullet_drones.envs:VisionAviary",
)


register(
    id="takeoff-aviary-v0",
    entry_point="gym_pybullet_drones.envs.single_agent_rl:TakeoffAviary",
)

register(
    id="hover-aviary-v0",
    entry_point="gym_pybullet_drones.envs.single_agent_rl:HoverAviary",
)

register(
    id="flythrugate-aviary-v0",
    entry_point="gym_pybullet_drones.envs.single_agent_rl:FlyThruGateAviary",
)

register(
    id="tune-aviary-v0",
    entry_point="gym_pybullet_drones.envs.single_agent_rl:TuneAviary",
)

register(
    id="maze-aviary-v0",
    entry_point="gym_pybullet_drones.envs.single_agent_rl:NavigateMazeAviary",
)

register(
    id="obstacles-aviary-v0",
    entry_point="gym_pybullet_drones.envs.single_agent_rl:NavigateObstaclesAviary",
)

register(
    id="cross-obstacles-aviary-v0",
    entry_point="gym_pybullet_drones.envs.single_agent_rl:CrossObstaclesAviary",
)

register(
    id="long-cross-obs-aviary-v0",
    entry_point="gym_pybullet_drones.envs.single_agent_rl:LongCrossObstaclesAviary",
)

register(
    id="field-aviary-v0",
    entry_point="gym_pybullet_drones.envs.single_agent_rl:FieldCoverageAviary",
)

register(
    id="land-aviary-v0",
    entry_point="gym_pybullet_drones.envs.single_agent_rl:NavigateLandAviary",
)

register(
    id="land-vision-aviary-v0",
    entry_point="gym_pybullet_drones.envs.single_agent_rl:LandVisionAviary",
)

register(
    id="flock-aviary-v0",
    entry_point="gym_pybullet_drones.envs.multi_agent_rl:FlockAviary",
)

register(
    id="leaderfollower-aviary-v0",
    entry_point="gym_pybullet_drones.envs.multi_agent_rl:LeaderFollowerAviary",
)

register(
    id="meetup-aviary-v0",
    entry_point="gym_pybullet_drones.envs.multi_agent_rl:MeetupAviary",
)

register(
    id="multicrossobs-aviary-v0",
    entry_point="gym_pybullet_drones.envs.multi_agent_rl:MultiCrossObstaclesAviary",
)

register(
    id="room-aviary-v0",
    entry_point="gym_pybullet_drones.envs.single_agent_rl:RoomAviary",
)
