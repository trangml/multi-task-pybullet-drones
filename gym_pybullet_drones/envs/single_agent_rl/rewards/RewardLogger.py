import numpy as np

from stable_baselines3.common.callbacks import BaseCallback



class RewardLoggerCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self, rewards_dict) -> bool:
        # Log scalar value (here a random variable)
        for reward in rewards_dict:
            self.logger.record(
                "reward/{}".format(reward), rewards_dict[reward]
            )
            self.logger.record('random_value', value)
        return True
