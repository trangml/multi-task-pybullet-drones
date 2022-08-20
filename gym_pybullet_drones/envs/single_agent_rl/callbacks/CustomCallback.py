import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class CustomCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        # Log rewards
        reward_dict = self.training_env.envs[0].env.reward_dict
        cum_reward_dict = self.training_env.envs[0].env.cum_reward_dict
        for reward in reward_dict:
            self.logger.record(
                "reward/per_step_{}".format(reward),
                reward_dict[reward],  # this reward is per step
            )
            self.logger.record(
                "total_reward/total_{}".format(reward),
                cum_reward_dict[reward],  # reward per episode
            )
            self.logger.record(
                "avg_reward/avg_{}".format(reward),
                cum_reward_dict[reward]
                / self.training_env.envs[0].total_steps,  # Average reward per episode
            )

        # log observations
        obs = self.training_env.envs[0].env._getDroneStateVector(0)
        names = self.training_env.envs[0].env.getDroneObsLabels()
        for o, n in zip(obs, names):
            self.logger.record("obs/{}".format(n), o)

        return True
