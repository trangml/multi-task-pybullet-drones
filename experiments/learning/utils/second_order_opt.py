from typing import Any, Dict

import numpy as np
import optuna
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from torch import nn as nn


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    gradient_weight = trial.suggest_categorical(
        "gradient_weight", [0.0001, 0.001, 0.01, 0.1, 1, 10]
    )
    hessian_approx = trial.suggest_categorical(
        "hessian_approx", [0.0001, 0.001, 0.01, 0.1, 1, 10]
    )

    return {
        "ppo": {"gradient_weight": gradient_weight, "hessian_approx": hessian_approx}
    }


HYPERPARAMS_SAMPLER = {
    "ppo": sample_ppo_params,
}
