import numpy as np
import os

from stable_baselines3.common.callbacks import BaseCallback


class StopTrainingRunningAverageRewardThreshold(BaseCallback):
    """
    Stop the training once the last n evals mean reward is above a threshold

    It must be used with the ``EvalCallback``.

    :param reward_threshold:  Minimum expected reward per episode
        to stop training.
    :param eval_rollback_len: Number of last evals to consider
        to stop training.
    :param verbose:
    """

    def __init__(
        self, reward_threshold: float, eval_rollback_len: int = 1, verbose: int = 0
    ):
        super().__init__(verbose=verbose)
        self.eval_rollback_len = eval_rollback_len
        self.buffer = [0] * self.eval_rollback_len
        self.head = 0
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        assert self.parent is not None, (
            "``StopTrainingRunningAverageRewardThreshold`` callback must be used "
            "with an ``EvalCallback``"
        )
        self.buffer[self.head] = self.parent.last_mean_reward
        self.head = (self.head + 1) % self.eval_rollback_len
        # Convert np.bool_ to bool, otherwise callback() is False won't work
        running_ave = np.average(self.buffer)
        continue_training = bool(running_ave < self.reward_threshold)
        if self.verbose > 0 and not continue_training:
            print(
                f"Stopping training because the mean running reward {running_ave:.2f} "
                f" is above the threshold {self.reward_threshold}"
            )
        return continue_training
