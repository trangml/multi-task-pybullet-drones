import numpy as np

from stable_baselines3.common.callbacks import BaseCallback



class RewardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        reward_dict = self.training_env.envs[0].env.reward_dict
        cum_reward_dict = self.training_env.envs[0].env.cum_reward_dict
        for reward in reward_dict:
            self.logger.record(
                "reward/Per_Step_{}".format(reward), reward_dict[reward]
            )
            self.logger.record(
                "reward/Total_Episode_{}".format(reward), cum_reward_dict[reward]
            )
            self.logger.record(
                "reward/Average_Episode_{}".format(reward), cum_reward_dict[reward] /  self.training_env.envs[0].total_steps
            )
        return True
